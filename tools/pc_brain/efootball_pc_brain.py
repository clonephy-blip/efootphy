#!/usr/bin/env python3
"""
efootball_pc_brain.py
Native Windows PC Co-Processor for eFootball Mobile (Method B).
Ultra-low latency real-time screen analysis, YOLO11n AI detection, team color K-Means clustering,
tactical pass suggestions, and normalized coordinate streaming to the Android AR HUD.
"""

import ctypes
import ctypes.wintypes
import json
import os
import select
import socket
import sys
import time
from pathlib import Path

import cv2
import mss
import numpy as np
import onnxruntime as ort
from PIL import ImageGrab
try:
    from tools.pc_brain.runtime_safety import is_known_team, resolve_field_model, validate_field_onnx_io
    from tools.pc_brain.transport_rtt import AckRttMeter
    from tools.pc_brain.field_postprocess import decode_field_detections, cluster_kit_colors
except ModuleNotFoundError as exc:
    if exc.name not in ("tools", "tools.pc_brain"):
        raise
    from runtime_safety import is_known_team, resolve_field_model, validate_field_onnx_io
    from transport_rtt import AckRttMeter
    from field_postprocess import decode_field_detections, cluster_kit_colors

# Class indices from YOLO11n field model
CLASS_BALL = 0
CLASS_CURSOR = 1
CLASS_CURSOR_SEC = 2
CLASS_GOALKEEPER = 3
CLASS_PLAYER = 4
CLASS_POWERBAR = 5
CLASS_REFEREE = 6


class WindowCapture:
    """High-speed screen grabber targeting the scrcpy game window or primary monitor."""

    def __init__(self, window_title="eFootball-Screen"):
        self.window_title = window_title
        self.user32 = ctypes.windll.user32
        # Explicit pointer-sized HWND signatures for 64-bit Windows.
        self.user32.FindWindowW.argtypes = (ctypes.wintypes.LPCWSTR, ctypes.wintypes.LPCWSTR)
        self.user32.FindWindowW.restype = ctypes.wintypes.HWND
        for name in ("IsWindow", "IsWindowVisible", "IsIconic"):
            fn = getattr(self.user32, name)
            fn.argtypes = (ctypes.wintypes.HWND,)
            fn.restype = ctypes.wintypes.BOOL
        self.user32.GetClientRect.argtypes = (ctypes.wintypes.HWND, ctypes.POINTER(ctypes.wintypes.RECT))
        self.user32.GetClientRect.restype = ctypes.wintypes.BOOL
        self.user32.ClientToScreen.argtypes = (ctypes.wintypes.HWND, ctypes.POINTER(ctypes.wintypes.POINT))
        self.user32.ClientToScreen.restype = ctypes.wintypes.BOOL
        self.user32.GetWindowTextLengthW.argtypes = (ctypes.wintypes.HWND,)
        self.user32.GetWindowTextLengthW.restype = ctypes.c_int
        self.user32.GetWindowTextW.argtypes = (ctypes.wintypes.HWND, ctypes.wintypes.LPWSTR, ctypes.c_int)
        self.user32.GetWindowTextW.restype = ctypes.c_int
        self._enum_callback_type = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.wintypes.HWND, ctypes.wintypes.LPARAM)
        self.user32.EnumWindows.argtypes = (self._enum_callback_type, ctypes.wintypes.LPARAM)
        self.user32.EnumWindows.restype = ctypes.wintypes.BOOL
        self.last_warn_time = 0.0
        try:
            self.sct = mss.mss()
        except Exception:
            self.sct = None

    def find_target_window(self):
        hwnd = self.user32.FindWindowW(None, self.window_title)
        if hwnd and self.user32.IsWindow(hwnd) and self.user32.IsWindowVisible(hwnd) and not self.user32.IsIconic(hwnd):
            return hwnd

        # Fallback: search all visible windows for eFootball or scrcpy
        matches = []
        def enum_cb(h, _):
            if self.user32.IsWindowVisible(h) and not self.user32.IsIconic(h):
                length = self.user32.GetWindowTextLengthW(h)
                if length > 0:
                    buff = ctypes.create_unicode_buffer(length + 1)
                    self.user32.GetWindowTextW(h, buff, length + 1)
                    title = buff.value
                    if "efootball" in title.lower() or "scrcpy" in title.lower():
                        matches.append((h, title))
            return True

        WNDENUMPROC = self._enum_callback_type
        self.user32.EnumWindows(WNDENUMPROC(enum_cb), 0)
        return matches[0][0] if matches else None

    def get_window_rect(self):
        hwnd = self.find_target_window()
        if not hwnd:
            return None

        rect = ctypes.wintypes.RECT()
        if not self.user32.GetClientRect(hwnd, ctypes.byref(rect)):
            return None
        pt = ctypes.wintypes.POINT(0, 0)
        if not self.user32.ClientToScreen(hwnd, ctypes.byref(pt)):
            return None

        width = rect.right - rect.left
        height = rect.bottom - rect.top
        if width <= 100 or height <= 100:
            return None

        return {
            "left": int(pt.x),
            "top": int(pt.y),
            "right": int(pt.x + width),
            "bottom": int(pt.y + height),
            "width": int(width),
            "height": int(height)
        }

    def grab(self):
        bbox = self.get_window_rect()
        frame = None
        is_window = False

        if bbox is not None:
            # 1. Try ultra-fast mss first (~6-10ms)
            if self.sct is not None:
                try:
                    sct_img = self.sct.grab(bbox)
                    bgra = np.array(sct_img)
                    frame = cv2.cvtColor(bgra, cv2.COLOR_BGRA2BGR)
                    is_window = True
                except Exception:
                    frame = None

            # 2. Fallback to Pillow ImageGrab
            if frame is None:
                try:
                    img = ImageGrab.grab(bbox=(bbox["left"], bbox["top"], bbox["right"], bbox["bottom"]))
                    frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                    is_window = True
                except Exception:
                    frame = None

        if frame is None:
            # Do not analyze unrelated desktop pixels or synthetic black frames.
            return None, False

        # Check for black screen
        if frame.mean() < 5.0:
            now = time.time()
            if now - self.last_warn_time >= 3.0:
                self.last_warn_time = now
                print("[Capture WARNING] Captured frame is pitch black! Check that scrcpy window is visible on screen.")

        return frame, is_window


class TeamKitClassifier:
    """Dynamically clusters players into Team 0 vs Team 1 based on jersey color."""

    def __init__(self):
        self.team0_center = None
        self.team1_center = None

    def extract_jersey_color(self, img_bgr, bbox):
        x1, y1, x2, y2 = bbox
        h = y2 - y1
        w = x2 - x1
        if w < 8 or h < 12:
            return None

        # Torso crop: top 25% to 58% of height, middle 60% of width
        ty1 = int(y1 + 0.25 * h)
        ty2 = int(y1 + 0.58 * h)
        tx1 = int(x1 + 0.20 * w)
        tx2 = int(x2 - 0.20 * w)

        crop = img_bgr[max(0, ty1):min(img_bgr.shape[0], ty2), max(0, tx1):min(img_bgr.shape[1], tx2)]
        if crop.size == 0 or crop.shape[0] < 3 or crop.shape[1] < 3:
            return None

        # Filter out green grass background
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        grass_mask = (hsv[:, :, 0] >= 35) & (hsv[:, :, 0] <= 85) & (hsv[:, :, 1] >= 40)
        jersey_pixels = crop[~grass_mask]

        if len(jersey_pixels) < 10:
            return None

        # Median reduces the influence of grass edges, numbers and HUD markers.
        return np.median(jersey_pixels, axis=0)

    def classify_players(self, img_bgr, player_boxes):
        if not player_boxes:
            return []
        colors = [self.extract_jersey_color(img_bgr, box) for box in player_boxes]
        prior = ((self.team0_center, self.team1_center)
                 if self.team0_center is not None and self.team1_center is not None
                 else None)
        teams, centers = cluster_kit_colors(colors, prior=prior)
        if centers is not None:
            self.team0_center, self.team1_center = centers
        return teams


class TacticalPassAnalyzer:
    """Calculates open passing lanes between teammates while avoiding opponents."""

    @staticmethod
    def analyze(players, ball_pos, frame_w, frame_h):
        if not players or ball_pos is None:
            return None, []

        ball_possessor = None
        min_dist = float("inf")
        thresh_px = 0.09 * frame_w

        if ball_pos:
            bx, by = ball_pos
            for p in players:
                if not is_known_team(p.get("team")):
                    continue
                fx = (p["x1"] + p["x2"]) / 2.0
                fy = p["y2"]
                d = np.hypot(fx - bx, fy - by)
                if d < min_dist and d < thresh_px:
                    min_dist = d
                    ball_possessor = p

        pass_lines = []
        if ball_possessor:
            ball_possessor["has_ball"] = True
            holder_team = ball_possessor["team"]
            hx = (ball_possessor["x1"] + ball_possessor["x2"]) / 2.0
            hy = ball_possessor["y2"]

            teammates = [p for p in players if p["team"] == holder_team and p != ball_possessor]
            opponents = [p for p in players if p["team"] != holder_team and p["team"] in (0, 1)]

            safe_passes = []
            for tm in teammates:
                tx = (tm["x1"] + tm["x2"]) / 2.0
                ty = tm["y2"]
                pass_dist = np.hypot(tx - hx, ty - hy)
                if pass_dist < 0.05 * frame_w or pass_dist > 0.65 * frame_w:
                    continue

                is_safe = True
                line_vec = np.array([tx - hx, ty - hy])
                line_len = pass_dist
                line_unit = line_vec / (line_len + 1e-6)

                corridor_px = 0.045 * frame_w
                for opp in opponents:
                    ox = (opp["x1"] + opp["x2"]) / 2.0
                    oy = opp["y2"]
                    to_opp = np.array([ox - hx, oy - hy])
                    proj = np.dot(to_opp, line_unit)

                    if 25 < proj < line_len - 25:
                        perp_dist = np.linalg.norm(to_opp - proj * line_unit)
                        if perp_dist < corridor_px:
                            is_safe = False
                            break

                if is_safe:
                    safe_passes.append({
                        "from_x": float(hx / frame_w),
                        "from_y": float(hy / frame_h),
                        "to_x": float(tx / frame_w),
                        "to_y": float(ty / frame_h),
                        "safe": True,
                        "dist": pass_dist
                    })

            # Pick top 3 cleanest passing options
            safe_passes.sort(key=lambda p: p["dist"])
            pass_lines = [{k: p[k] for k in ("from_x", "from_y", "to_x", "to_y", "safe")} for p in safe_passes[:3]]

        return ball_possessor, pass_lines


class EFootballPcBrain:
    def __init__(self, model_path, port=8888):
        self.model_path = model_path
        self.port = port
        self.sess = None
        self.input_name = None
        self.input_size = 320
        self.kit_classifier = TeamKitClassifier()
        self.cap = WindowCapture(window_title="eFootball-Screen")
        self.sock = None
        self.packet_sequence = 0
        self.transport_meter = AckRttMeter()
        self.prev_ball = None
        self.prev_ball_time = time.time()
        self.ball_persistence = 0
        self.last_known_ball = None
        self.last_ball_v = (0.0, 0.0)
        self.last_log_time = 0.0
        self._init_model()

    def _init_model(self):
        print(f"[PC Brain] Loading YOLO11n ONNX model: {self.model_path}")
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 2
        opts.inter_op_num_threads = 1
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.sess = ort.InferenceSession(str(self.model_path), sess_options=opts)
        self.input_size = validate_field_onnx_io(self.sess.get_inputs(), self.sess.get_outputs())
        self.input_name = self.sess.get_inputs()[0].name
        print(f"[PC Brain] Field ONNX validated: {self.input_size}x{self.input_size}, 7 classes; input: {self.input_name}")

    def connect_socket(self):
        if self.sock is not None:
            return True

        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            s.settimeout(0.3)
            s.connect(("127.0.0.1", self.port))
            s.settimeout(0.05)  # Avoid blocking video capture on a stalled socket.
            self.sock = s
            self.transport_meter.reset()
            print(f"[PC Brain] >>> CONNECTED TO PHONE HUD (Port {self.port}) <<<")
            return True
        except Exception:
            try: s.close()
            except Exception: pass
            self.sock = None
            return False

    def _disconnect_socket(self):
        if self.sock is not None:
            try:
                self.sock.close()
            except OSError:
                pass
        self.sock = None
        self.transport_meter.reset()

    def _receive_acks(self):
        """Nonblocking PC→Android receive/parser→PC RTT, on the PC clock only."""
        if self.sock is None:
            return
        try:
            while select.select([self.sock], [], [], 0)[0]:
                chunk = self.sock.recv(4096)
                if not chunk:
                    raise ConnectionError("HUD socket closed")
                self.transport_meter.feed(chunk)
        except (OSError, ConnectionError):
            self._disconnect_socket()

    def send_packet(self, packet_dict):
        if not self.connect_socket():
            return False
        try:
            seq = self.packet_sequence
            self.packet_sequence += 1
            payload = dict(packet_dict)
            payload["seq"] = seq
            last_rtt = self.transport_meter.last_rtt_ms
            payload["transport_rtt_ms"] = round(last_rtt, 3) if last_rtt is not None else -1.0
            line = json.dumps(payload, separators=(",", ":")) + "\n"
            self.transport_meter.record_sent(seq)
            self.sock.sendall(line.encode("utf-8"))
            self._receive_acks()
            return self.sock is not None
        except (OSError, ConnectionError):
            self._disconnect_socket()
            return False

    def run_detection(self, frame_bgr):
        orig_h, orig_w = frame_bgr.shape[:2]
        resized = cv2.resize(frame_bgr, (self.input_size, self.input_size))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        tensor = np.transpose(rgb, (2, 0, 1)).astype(np.float32) / 255.0
        tensor = np.expand_dims(tensor, axis=0)

        t0 = time.time()
        preds = self.sess.run(None, {self.input_name: tensor})[0]  # (1,11,2100) or (1,11,8400)
        infer_time_ms = (time.time() - t0) * 1000.0

        # Decode all seven classes with *class-local* NMS. The ball channel is
        # evaluated independently of non-ball argmax so a player/cursor proposal
        # does not silently discard a small overlapping ball candidate.
        # This improves proposal coverage, NOT yet verified ball precision.
        detected = decode_field_detections(
            preds, orig_w, orig_h, input_size=self.input_size,
            object_threshold=0.22, ball_threshold=0.08, iou_threshold=0.50
        )
        ball_box = detected[CLASS_BALL][0][0] if detected[CLASS_BALL] else None
        player_boxes = [box for box, _ in detected[CLASS_PLAYER]]
        gk_boxes = [box for box, _ in detected[CLASS_GOALKEEPER]]
        ref_boxes = [box for box, _ in detected[CLASS_REFEREE]]
        cursor_boxes = ([box for box, _ in detected[CLASS_CURSOR]] +
                        [box for box, _ in detected[CLASS_CURSOR_SEC]])

        # Team clustering for regular field players
        team_labels = self.kit_classifier.classify_players(frame_bgr, player_boxes)

        raw_players = []
        for box, team in zip(player_boxes, team_labels):
            raw_players.append({
                "x1": box[0], "y1": box[1], "x2": box[2], "y2": box[3],
                "team": int(team),
                "has_ball": False,
                "controlled": False
            })

        for box in gk_boxes:
            raw_players.append({
                "x1": box[0], "y1": box[1], "x2": box[2], "y2": box[3],
                "team": 2,  # Goalkeeper
                "has_ball": False,
                "controlled": False
            })

        for box in ref_boxes:
            raw_players.append({
                "x1": box[0], "y1": box[1], "x2": box[2], "y2": box[3],
                "team": 3,  # Referee
                "has_ball": False,
                "controlled": False
            })

        for cbox in cursor_boxes:
            ccx = (cbox[0] + cbox[2]) / 2.0
            ccy = cbox[3]
            for p in raw_players:
                if p["x1"] - 30 <= ccx <= p["x2"] + 30 and abs(p["y1"] - ccy) < 50:
                    p["controlled"] = True
                    break

        # Process Ball & Trajectory
        ball_data = None
        ball_pos = None
        if ball_box is not None:
            bx = (ball_box[0] + ball_box[2]) / 2.0
            by = (ball_box[1] + ball_box[3]) / 2.0
            ball_pos = (bx, by)
            now = time.time()
            dt = max(now - self.prev_ball_time, 0.001)
            vx, vy = 0.0, 0.0
            if self.prev_ball is not None:
                vx = (bx - self.prev_ball[0]) / dt
                vy = (by - self.prev_ball[1]) / dt
            self.prev_ball = (bx, by)
            self.prev_ball_time = now
            self.last_known_ball = (bx, by)
            self.last_ball_v = (vx, vy)
            self.ball_persistence = 5
            ball_data = {
                "x": float(bx / orig_w),
                "y": float(by / orig_h),
                "vx": float(vx / orig_w),
                "vy": float(vy / orig_h)
            }
        elif self.ball_persistence > 0 and self.last_known_ball is not None:
            now = time.time()
            dt = min(now - self.prev_ball_time, 0.2)
            self.ball_persistence -= 1
            bx = self.last_known_ball[0] + self.last_ball_v[0] * dt * 0.9
            by = self.last_known_ball[1] + self.last_ball_v[1] * dt * 0.9
            self.last_known_ball = (bx, by)
            self.prev_ball_time = now
            ball_pos = (bx, by)
            ball_data = {
                "x": float(bx / orig_w),
                "y": float(by / orig_h),
                "vx": float(self.last_ball_v[0] / orig_w * 0.9),
                "vy": float(self.last_ball_v[1] / orig_h * 0.9)
            }

        # Tactical Pass Lines
        # Hide pass hints if ball position is only an unconfirmed extrapolation.
        _, pass_lines = TacticalPassAnalyzer.analyze(raw_players, ball_pos if ball_box is not None else None, orig_w, orig_h)

        # Convert players to normalized coordinates [0.0, 1.0] for the phone HUD
        norm_players = []
        for p in raw_players:
            fx = ((p["x1"] + p["x2"]) / 2.0) / orig_w
            fy = p["y2"] / orig_h
            rx = ((p["x2"] - p["x1"]) * 0.70) / orig_w
            ry = ((p["y2"] - p["y1"]) * 0.25) / orig_h
            norm_players.append({
                "x1": float(p["x1"] / orig_w),
                "y1": float(p["y1"] / orig_h),
                "x2": float(p["x2"] / orig_w),
                "y2": float(p["y2"] / orig_h),
                "feet_x": float(fx),
                "feet_y": float(fy),
                "radius_x": float(rx),
                "radius_y": float(ry),
                "team": int(p["team"]),
                "has_ball": bool(p["has_ball"]),
                "controlled": bool(p["controlled"])
            })

        return ball_data, norm_players, pass_lines, infer_time_ms

    def start_loop(self):
        print("\n" + "=" * 65)
        print("   eFootball PC Brain Active (Method B - Native Windows)")
        print("=" * 65)
        print("[Status] Searching for 'eFootball-Screen' window...")
        print("[Status] Monitoring gameplay and sending HUD packets to phone...\n")

        fps_counter = 0
        fps_start = time.time()
        current_fps = 30.0

        while True:
            loop_t0 = time.time()

            # Grab screen frame
            frame_bgr, is_window = self.cap.grab()
            if frame_bgr is None:
                self.prev_ball = None
                self.last_known_ball = None
                self.ball_persistence = 0
                self.kit_classifier.team0_center = None
                self.kit_classifier.team1_center = None
                self.send_packet({"timestamp": int(time.time() * 1000), "fps": 0.0,
                                  "latency": 0.0, "status": "SCRCPY_UNAVAILABLE",
                                  "ball": None, "players": [], "pass_lines": []})
                time.sleep(0.2)
                continue

            # AI Inference & Tactical Analysis
            ball_data, players, pass_lines, infer_ms = self.run_detection(frame_bgr)

            fps_counter += 1
            now = time.time()
            if now - fps_start >= 1.0:
                current_fps = fps_counter / (now - fps_start)
                fps_counter = 0
                fps_start = now

            total_lat = (time.time() - loop_t0) * 1000.0

            # Periodic Terminal Diagnostics (every 1 second)
            if now - self.last_log_time >= 1.0:
                self.last_log_time = now
                team_a = sum(1 for p in players if p["team"] == 0)
                team_b = sum(1 for p in players if p["team"] == 1)
                ball_txt = f"({ball_data['x']:.2f}, {ball_data['y']:.2f})" if ball_data else "searching"
                target_txt = "eFootball-Screen" if is_window else "Desktop (fallback)"
                conn_txt = "CONNECTED" if self.sock else "waiting phone..."
                transport_txt = f"{self.transport_meter.last_rtt_ms:.1f}ms" if self.transport_meter.last_rtt_ms is not None else "--"
                print(f"[{conn_txt}] Src: {target_txt} | FPS: {current_fps:4.1f} | Infer: {infer_ms:4.1f}ms | USB RTT: {transport_txt} | Players: {len(players):2d} (A:{team_a:2d}, B:{team_b:2d}) | Ball: {ball_txt} | Passes: {len(pass_lines):2d}")

            # Build Packet for Phone
            packet = {
                "timestamp": int(time.time() * 1000),
                "fps": float(current_fps),
                "latency": float(total_lat),
                "status": f"AI Active ({infer_ms:.1f}ms)",
                "ball": ball_data,
                "players": players,
                "pass_lines": pass_lines
            }

            self.send_packet(packet)

            # Minimal sleep to yield CPU
            elapsed = time.time() - loop_t0
            sleep_needed = max(0.010 - elapsed, 0.001)
            time.sleep(sleep_needed)


def main():
    model_path = resolve_field_model(__file__, override=os.environ.get("EFOOTPHY_FIELD_MODEL"))

    brain = EFootballPcBrain(model_path=model_path, port=8888)
    try:
        brain.start_loop()
    except KeyboardInterrupt:
        print("\n[PC Brain] Stopped.")


if __name__ == "__main__":
    main()
