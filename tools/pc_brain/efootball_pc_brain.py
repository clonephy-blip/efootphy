#!/usr/bin/env python3
"""
efootball_pc_brain.py
PC Co-Processor for eFootball Mobile (Method B).
Runs AI detection, team jersey K-Means clustering, ball tracking, and tactical pass suggestions.
Streams lightweight JSON HUD packets over USB ADB socket to the phone overlay at high speed.
"""

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
from sklearn.cluster import KMeans

# Class indices from YOLO11n field model
CLASS_BALL = 0
CLASS_CURSOR = 1
CLASS_CURSOR_SEC = 2
CLASS_GOALKEEPER = 3
CLASS_PLAYER = 4
CLASS_POWERBAR = 5
CLASS_REFEREE = 6


class TeamKitClassifier:
    """Dynamically clusters players into Team 0 vs Team 1 based on jersey color (HSV/RGB)."""

    def __init__(self):
        self.team0_center = None
        self.team1_center = None

    def extract_jersey_color(self, img_bgr, bbox):
        x1, y1, x2, y2 = bbox
        h = y2 - y1
        w = x2 - x1
        if w < 10 or h < 15:
            return None

        # Torso crop: top 25% to 58% of height, middle 60% of width
        ty1 = int(y1 + 0.25 * h)
        ty2 = int(y1 + 0.58 * h)
        tx1 = int(x1 + 0.20 * w)
        tx2 = int(x2 - 0.20 * w)

        crop = img_bgr[max(0, ty1):min(img_bgr.shape[0], ty2), max(0, tx1):min(img_bgr.shape[1], tx2)]
        if crop.size == 0 or crop.shape[0] < 4 or crop.shape[1] < 4:
            return None

        # Filter out green grass background
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        grass_mask = (hsv[:, :, 0] >= 35) & (hsv[:, :, 0] <= 85) & (hsv[:, :, 1] >= 40)
        jersey_pixels = crop[~grass_mask]

        if len(jersey_pixels) < 15:
            jersey_pixels = crop.reshape(-1, 3)

        # Average BGR of jersey
        mean_bgr = np.mean(jersey_pixels, axis=0)
        return mean_bgr

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

            # Track cluster center identity across frames for color stability
            if self.team0_center is None:
                self.team0_center = centers[0]
                self.team1_center = centers[1]
            else:
                dist_direct = np.linalg.norm(self.team0_center - centers[0]) + np.linalg.norm(self.team1_center - centers[1])
                dist_swapped = np.linalg.norm(self.team0_center - centers[1]) + np.linalg.norm(self.team1_center - centers[0])
                if dist_swapped < dist_direct:
                    labels = 1 - labels
                    centers = centers[[1, 0]]

                # Smooth update
                self.team0_center = 0.85 * self.team0_center + 0.15 * centers[0]
                self.team1_center = 0.85 * self.team1_center + 0.15 * centers[1]

            for idx, label in zip(valid_indices, labels):
                teams[idx] = int(label)

        return teams


class TacticalPassAnalyzer:
    """Calculates open passing lanes between teammates while avoiding opponents."""

    @staticmethod
    def analyze(players, ball_pos):
        if not players:
            return None, []

        ball_possessor = None
        min_dist = float("inf")

        # Find player closest to the ball
        if ball_pos:
            bx, by = ball_pos
            for p in players:
                fx = (p["x1"] + p["x2"]) / 2.0
                fy = p["y2"]
                d = np.hypot(fx - bx, fy - by)
                if d < min_dist and d < 80:  # within 80px
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
                if pass_dist < 40:
                    continue  # Too close

                # Check if any opponent is blocking this corridor
                is_safe = True
                line_vec = np.array([tx - hx, ty - hy])
                line_len = pass_dist
                line_unit = line_vec / (line_len + 1e-6)

                for opp in opponents:
                    ox = (opp["x1"] + opp["x2"]) / 2.0
                    oy = opp["y2"]
                    to_opp = np.array([ox - hx, oy - hy])
                    proj = np.dot(to_opp, line_unit)

                    if 20 < proj < line_len - 20:
                        perp_dist = np.linalg.norm(to_opp - proj * line_unit)
                        if perp_dist < 45:  # Opponent in corridor
                            is_safe = False
                            break

                pass_lines.append({
                    "from_x": float(hx),
                    "from_y": float(hy),
                    "to_x": float(tx),
                    "to_y": float(ty),
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
        self.sock = None
        self.sct = mss.mss()
        self.prev_ball = None
        self.prev_ball_time = time.time()
        self._init_model()

    def _init_model(self):
        print(f"Loading YOLO11n ONNX model from: {self.model_path}")
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 4
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.sess = ort.InferenceSession(str(self.model_path), sess_options=opts)
        self.input_name = self.sess.get_inputs()[0].name
        print(f"Model loaded successfully! Input: {self.input_name}")

    def connect_socket(self):
        if self.sock is not None:
            return True

        # Candidate target endpoints
        targets = [("127.0.0.1", 8888)]
        # If in WSL, resolve host Windows IP
        if Path("/etc/resolv.conf").exists():
            try:
                with open("/etc/resolv.conf") as f:
                    for line in f:
                        if line.strip().startswith("nameserver"):
                            host_ip = line.strip().split()[1]
                            targets.append((host_ip, 8889))
                            targets.append((host_ip, 8888))
                            break
            except Exception:
                pass

        for host, port in targets:
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                s.settimeout(0.5)
                s.connect((host, port))
                self.sock = s
                print(f"Connected to Phone HUD Server at {host}:{port} via USB ADB!")
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
        # Transpose to (2100, 11) -> 4 box coords + 7 class scores
        trans = boxes_data.T
        cx = trans[:, 0]
        cy = trans[:, 1]
        w = trans[:, 2]
        h = trans[:, 3]
        scores = trans[:, 4:]  # 7 classes

        class_ids = np.argmax(scores, axis=1)
        confidences = np.max(scores, axis=1)

        # Scale factors
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

            # Ball threshold is lower (0.08) because ball is tiny in full-field views
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

        # Team clustering for players
        team_labels = self.kit_classifier.classify_players(frame_bgr, player_boxes)

        # Construct final player list
        final_players = []
        for box, team in zip(player_boxes, team_labels):
            final_players.append({
                "x1": box[0], "y1": box[1], "x2": box[2], "y2": box[3],
                "team": int(team),
                "has_ball": False,
                "controlled": False
            })

        for box in gk_boxes:
            final_players.append({
                "x1": box[0], "y1": box[1], "x2": box[2], "y2": box[3],
                "team": 2,  # Goalkeeper
                "has_ball": False,
                "controlled": False
            })

        for box in ref_boxes:
            final_players.append({
                "x1": box[0], "y1": box[1], "x2": box[2], "y2": box[3],
                "team": 3,  # Referee
                "has_ball": False,
                "controlled": False
            })

        # Mark controlled player based on cursor position
        for cbox in cursor_boxes:
            ccx = (cbox[0] + cbox[2]) / 2.0
            ccy = cbox[3]  # tip of cursor
            for p in final_players:
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
            ball_data = {"x": float(bx), "y": float(by), "vx": float(vx), "vy": float(vy)}

        # Tactical Pass Lines
        _, pass_lines = TacticalPassAnalyzer.analyze(final_players, ball_pos)

        return ball_data, final_players, pass_lines, infer_time_ms

    def start_loop(self):
        print("\n🚀 eFootball PC Brain is actively monitoring gameplay...")
        print("💡 Make sure scrcpy is mirroring the game window on your screen.\n")

        fps_counter = 0
        fps_start = time.time()
        current_fps = 30.0

        monitor = self.sct.monitors[1] if len(self.sct.monitors) > 1 else self.sct.monitors[0]

        while True:
            loop_t0 = time.time()

            # Grab screen frame
            sct_img = self.sct.grab(monitor)
            frame_bgra = np.array(sct_img)
            frame_bgr = cv2.cvtColor(frame_bgra, cv2.COLOR_BGRA2BGR)

            # AI Inference & Analysis
            ball_data, players, pass_lines, infer_time_ms = self.run_detection(frame_bgr)

            fps_counter += 1
            if time.time() - fps_start >= 1.0:
                current_fps = fps_counter / (time.time() - fps_start)
                fps_counter = 0
                fps_start = time.time()

            total_lat = (time.time() - loop_t0) * 1000.0

            # Build Packet
            packet = {
                "timestamp": int(time.time() * 1000),
                "fps": float(current_fps),
                "latency": float(total_lat),
                "status": f"AI Active ({infer_time_ms:.1f}ms)",
                "ball": ball_data,
                "players": players,
                "pass_lines": pass_lines
            }

            self.send_packet(packet)

            # Target ~30-40 Hz update rate
            elapsed = time.time() - loop_t0
            sleep_needed = max(0.015 - elapsed, 0.001)
            time.sleep(sleep_needed)


def main():
    model_path = Path("/mnt/c/Users/PC/.gemini/antigravity/scratch/efootphy/tools/models/field_yolo11n_320.onnx")
    if not model_path.exists():
        model_path = Path("C:/Users/PC/.gemini/antigravity/scratch/efootphy/tools/models/field_yolo11n_320.onnx")

    brain = EFootballPcBrain(model_path=model_path, port=8888)
    try:
        brain.start_loop()
    except KeyboardInterrupt:
        print("\nStopped PC Brain.")


if __name__ == "__main__":
    main()
