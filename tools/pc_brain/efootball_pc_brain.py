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
import socket
import sys
import time
from pathlib import Path

import cv2
import mss
import numpy as np
import onnxruntime as ort
from PIL import ImageGrab
from sklearn.cluster import KMeans

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
        self.last_warn_time = 0.0

    def find_target_window(self):
        hwnd = self.user32.FindWindowW(None, self.window_title)
        if hwnd and self.user32.IsWindow(hwnd) and self.user32.IsWindowVisible(hwnd):
            return hwnd

        # Fallback: search all visible windows for eFootball or scrcpy
        matches = []
        def enum_cb(h, _):
            if self.user32.IsWindowVisible(h):
                length = self.user32.GetWindowTextLengthW(h)
                if length > 0:
                    buff = ctypes.create_unicode_buffer(length + 1)
                    self.user32.GetWindowTextW(h, buff, length + 1)
                    title = buff.value
                    if "efootball" in title.lower() or "scrcpy" in title.lower():
                        matches.append((h, title))
            return True

        WNDENUMPROC = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_int, ctypes.c_int)
        self.user32.EnumWindows(WNDENUMPROC(enum_cb), 0)
        return matches[0][0] if matches else None

    def get_window_rect(self):
        hwnd = self.find_target_window()
        if not hwnd:
            return None

        rect = ctypes.wintypes.RECT()
        self.user32.GetClientRect(hwnd, ctypes.byref(rect))
        pt = ctypes.wintypes.POINT(0, 0)
        self.user32.ClientToScreen(hwnd, ctypes.byref(pt))

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
            # Try Pillow ImageGrab (handles DPI scaling and multi-monitors cleanly)
            try:
                img = ImageGrab.grab(bbox=(bbox["left"], bbox["top"], bbox["right"], bbox["bottom"]))
                frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                is_window = True
            except Exception:
                frame = None

        if frame is None:
            # Fallback: grab entire primary desktop
            try:
                img = ImageGrab.grab()
                frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            except Exception:
                try:
                    with mss.mss() as sct:
                        mon = sct.monitors[1] if len(sct.monitors) > 1 else sct.monitors[0]
                        sct_img = sct.grab(mon)
                        frame = cv2.cvtColor(np.array(sct_img), cv2.COLOR_BGRA2BGR)
                except Exception:
                    frame = np.zeros((360, 640, 3), dtype=np.uint8)

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
            jersey_pixels = crop.reshape(-1, 3)

        return np.mean(jersey_pixels, axis=0)

    def classify_players(self, img_bgr, player_boxes):
        if not player_boxes:
            return []

        colors = []
        valid_indices = []
        for i, box in enumerate(player_boxes):
            c = self.extract_jersey_color(img_bgr, box)
            if c is not None:
                colors.append(c)
                valid_indices.append(i)

        teams = [0] * len(player_boxes)
        if len(colors) >= 2:
            colors_arr = np.array(colors)
            kmeans = KMeans(n_clusters=2, n_init=3, max_iter=20, random_state=42)
            labels = kmeans.fit_predict(colors_arr)
            centers = kmeans.cluster_centers_

            if self.team0_center is None:
                self.team0_center = centers[0]
                self.team1_center = centers[1]
            else:
                dist_direct = np.linalg.norm(self.team0_center - centers[0]) + np.linalg.norm(self.team1_center - centers[1])
                dist_swapped = np.linalg.norm(self.team0_center - centers[1]) + np.linalg.norm(self.team1_center - centers[0])
                if dist_swapped < dist_direct:
                    labels = 1 - labels
                    centers = centers[[1, 0]]

                self.team0_center = 0.85 * self.team0_center + 0.15 * centers[0]
                self.team1_center = 0.85 * self.team1_center + 0.15 * centers[1]

            for idx, label in zip(valid_indices, labels):
                teams[idx] = int(label)

        return teams


class TacticalPassAnalyzer:
    """Calculates open passing lanes between teammates while avoiding opponents."""

    @staticmethod
    def analyze(players, ball_pos, frame_w, frame_h):
        if not players:
            return None, []

        ball_possessor = None
        min_dist = float("inf")
        thresh_px = 0.09 * frame_w

        if ball_pos:
            bx, by = ball_pos
            for p in players:
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

            for tm in teammates:
                tx = (tm["x1"] + tm["x2"]) / 2.0
                ty = tm["y2"]
                pass_dist = np.hypot(tx - hx, ty - hy)
                if pass_dist < 0.04 * frame_w:
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

                    if 20 < proj < line_len - 20:
                        perp_dist = np.linalg.norm(to_opp - proj * line_unit)
                        if perp_dist < corridor_px:
                            is_safe = False
                            break

                pass_lines.append({
                    "from_x": float(hx / frame_w),
                    "from_y": float(hy / frame_h),
                    "to_x": float(tx / frame_w),
                    "to_y": float(ty / frame_h),
                    "safe": bool(is_safe)
                })

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
        self.prev_ball = None
        self.prev_ball_time = time.time()
        self.last_log_time = 0.0
        self._init_model()

    def _init_model(self):
        print(f"[PC Brain] Loading YOLO11n ONNX model: {self.model_path}")
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 4
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.sess = ort.InferenceSession(str(self.model_path), sess_options=opts)
        self.input_name = self.sess.get_inputs()[0].name
        print(f"[PC Brain] Model loaded successfully! Input: {self.input_name}")

    def connect_socket(self):
        if self.sock is not None:
            return True

        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            s.settimeout(0.3)
            s.connect(("127.0.0.1", self.port))
            self.sock = s
            print(f"[PC Brain] >>> CONNECTED TO PHONE HUD (Port {self.port}) <<<")
            return True
        except Exception:
            try: s.close()
            except Exception: pass
            self.sock = None
            return False

    def send_packet(self, packet_dict):
        if not self.connect_socket():
            return False
        try:
            line = json.dumps(packet_dict) + "\n"
            self.sock.sendall(line.encode("utf-8"))
            return True
        except Exception:
            if self.sock:
                try: self.sock.close()
                except Exception: pass
            self.sock = None
            return False

    def run_detection(self, frame_bgr):
        orig_h, orig_w = frame_bgr.shape[:2]
        resized = cv2.resize(frame_bgr, (self.input_size, self.input_size))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        tensor = np.transpose(rgb, (2, 0, 1)).astype(np.float32) / 255.0
        tensor = np.expand_dims(tensor, axis=0)

        t0 = time.time()
        preds = self.sess.run(None, {self.input_name: tensor})[0]  # shape: (1, 11, 2100)
        infer_time_ms = (time.time() - t0) * 1000.0

        boxes_data = preds[0]  # shape: (11, 2100)
        trans = boxes_data.T
        cx = trans[:, 0]
        cy = trans[:, 1]
        w = trans[:, 2]
        h = trans[:, 3]
        scores = trans[:, 4:]  # 7 classes

        class_ids = np.argmax(scores, axis=1)
        confidences = np.max(scores, axis=1)

        scale_x = orig_w / float(self.input_size)
        scale_y = orig_h / float(self.input_size)

        ball_box = None
        max_ball_conf = 0.0
        player_boxes = []
        gk_boxes = []
        ref_boxes = []
        cursor_boxes = []

        for i in range(len(confidences)):
            conf = confidences[i]
            cid = class_ids[i]

            if cid == CLASS_BALL and conf >= 0.08:
                if conf > max_ball_conf:
                    max_ball_conf = conf
                    x1 = (cx[i] - w[i] / 2.0) * scale_x
                    y1 = (cy[i] - h[i] / 2.0) * scale_y
                    x2 = (cx[i] + w[i] / 2.0) * scale_x
                    y2 = (cy[i] + h[i] / 2.0) * scale_y
                    ball_box = (x1, y1, x2, y2)
            elif conf >= 0.25:
                x1 = (cx[i] - w[i] / 2.0) * scale_x
                y1 = (cy[i] - h[i] / 2.0) * scale_y
                x2 = (cx[i] + w[i] / 2.0) * scale_x
                y2 = (cy[i] + h[i] / 2.0) * scale_y
                box = [float(x1), float(y1), float(x2), float(y2)]

                if cid == CLASS_PLAYER:
                    player_boxes.append(box)
                elif cid == CLASS_GOALKEEPER:
                    gk_boxes.append(box)
                elif cid == CLASS_REFEREE:
                    ref_boxes.append(box)
                elif cid in (CLASS_CURSOR, CLASS_CURSOR_SEC):
                    cursor_boxes.append(box)

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
            ball_data = {
                "x": float(bx / orig_w),
                "y": float(by / orig_h),
                "vx": float(vx / orig_w),
                "vy": float(vy / orig_h)
            }

        # Tactical Pass Lines
        _, pass_lines = TacticalPassAnalyzer.analyze(raw_players, ball_pos, orig_w, orig_h)

        # Convert players to normalized coordinates [0.0, 1.0] for the phone HUD
        norm_players = []
        for p in raw_players:
            fx = ((p["x1"] + p["x2"]) / 2.0) / orig_w
            fy = p["y2"] / orig_h
            rx = ((p["x2"] - p["x1"]) * 0.70) / orig_w
            ry = ((p["y2"] - p["y1"]) * 0.25) / orig_h
            norm_players.append({
                "feet_x": float(fx),
                "feet_y": float(fy),
                "radius_x": float(rx),
                "radius_y": float(ry),
                "y1": float(p["y1"] / orig_h),
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
                print(f"[{conn_txt}] Src: {target_txt} | FPS: {current_fps:4.1f} | Infer: {infer_ms:4.1f}ms | Players: {len(players):2d} (A:{team_a:2d}, B:{team_b:2d}) | Ball: {ball_txt} | Passes: {len(pass_lines):2d}")

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

            # Save debug frame periodically so user can inspect what AI sees
            if fps_counter % 20 == 0:
                try:
                    dbg_img = frame_bgr.copy()
                    dh, dw = dbg_img.shape[:2]
                    for p in players:
                        fx = int(p["feet_x"] * dw)
                        fy = int(p["feet_y"] * dh)
                        color = (255, 220, 0) if p["team"] == 0 else (0, 60, 255) if p["team"] == 1 else (0, 255, 255)
                        cv2.circle(dbg_img, (fx, fy), max(int(dw * 0.02), 8), color, 2)
                    if ball_data:
                        bx = int(ball_data["x"] * dw)
                        by = int(ball_data["y"] * dh)
                        cv2.circle(dbg_img, (bx, by), 6, (0, 215, 255), -1)
                    dbg_path = Path(__file__).resolve().parent.parent.parent / "debug_pc_brain.jpg"
                    cv2.imwrite(str(dbg_path), dbg_img)
                except Exception:
                    pass

            # Cap loop at ~40 FPS to keep CPU cool
            elapsed = time.time() - loop_t0
            sleep_needed = max(0.020 - elapsed, 0.001)
            time.sleep(sleep_needed)


def main():
    model_path = Path("C:/Users/PC/.gemini/antigravity/scratch/efootphy/tools/models/field_yolo11n_320.onnx")
    if not model_path.exists():
        model_path = Path("tools/models/field_yolo11n_320.onnx")

    brain = EFootballPcBrain(model_path=model_path, port=8888)
    try:
        brain.start_loop()
    except KeyboardInterrupt:
        print("\n[PC Brain] Stopped.")


if __name__ == "__main__":
    main()
