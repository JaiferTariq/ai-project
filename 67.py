"""
╔══════════════════════════════════════════════════════════════════════════╗
║          DRIVER FATIGUE DETECTION SYSTEM  v2.0                          ║
║          AI Engine: MediaPipe FaceMesh + EAR + MAR + PERCLOS            ║
║                                                                          ║
║  Improvements over v1:                                                   ║
║   • MediaPipe Face Mesh (468 landmarks) replaces Haar Cascade           ║
║   • Eye Aspect Ratio (EAR) — Soukupová & Čech, 2016                    ║
║   • Mouth Aspect Ratio (MAR) for yawn detection                         ║
║   • PERCLOS metric (% eye closure over rolling 60-second window)        ║
║   • Head-pitch estimation for nod-off detection                         ║
║   • Per-session EAR calibration (adapts to each user)                   ║
║   • 4-level colour-coded alert system                                    ║
║   • Non-blocking threaded alarm (cross-platform)                        ║
║   • Pure-OpenCV real-time EAR graph (no matplotlib lag)                 ║
║   • Session statistics panel + auto-generated CSV log                   ║
║   • Screenshot (S), session reset (R), quit (Q)                         ║
╚══════════════════════════════════════════════════════════════════════════╝

Requirements:
    pip install opencv-python mediapipe numpy

Usage:
    python drowsiness_detector.py
"""

import cv2
import math
import os
import csv
import sys
import time
import platform
import threading
import subprocess
from datetime import datetime
from collections import deque

import numpy as np
import mediapipe as mp


# ═══════════════════════════════════════════════════════════════════════════
# LANDMARK INDICES  (MediaPipe 468-point model)
# ═══════════════════════════════════════════════════════════════════════════
# 6 EAR landmarks per eye: P1(left corner), P2, P3(right corner), P4, P5, P6
LEFT_EYE_EAR  = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_EAR = [33,  160, 158, 133, 153, 144]

# Full contours for drawing
LEFT_EYE_CTR  = [362,382,381,380,374,373,390,249,263,466,388,387,386,385,384,398]
RIGHT_EYE_CTR = [33,7,163,144,145,153,154,155,133,173,157,158,159,160,161,246]

# Mouth corners + top/bottom lips for MAR
MOUTH_TOP    = 13
MOUTH_BOTTOM = 14
MOUTH_LEFT   = 61
MOUTH_RIGHT  = 291

# Head-pose reference points
NOSE_TIP = 1
CHIN     = 152


# ═══════════════════════════════════════════════════════════════════════════
# TUNABLE CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════
EAR_THRESHOLD_DEFAULT = 0.22    # overridden by calibration
MAR_THRESHOLD         = 0.60    # above this → yawning
PERCLOS_WINDOW_SEC    = 60      # rolling window length (seconds)
PERCLOS_CAUTION       = 0.08    # 8 % closed → CAUTION
PERCLOS_WARNING       = 0.15    # 15 % → WARNING
PERCLOS_DANGER        = 0.25    # 25 % → DANGER
EYE_CLOSED_CAUTION    = 10      # consecutive frames → CAUTION
EYE_CLOSED_WARNING    = 20      # → WARNING
EYE_CLOSED_DANGER     = 35      # → DANGER (alarm)
YAWN_CONSEC_FRAMES    = 20      # frames yawning to count as one event
NOD_PITCH_THRESH      = 18.0    # degrees forward head drop
CALIB_DURATION        = 3.0     # seconds

ASSUMED_FPS           = 30      # used for PERCLOS window sizing

# Alert levels
ALERT_SAFE    = 0
ALERT_CAUTION = 1
ALERT_WARNING = 2
ALERT_DANGER  = 3

# Colours (BGR)
C_NEON    = (0, 255, 180)
C_GREEN   = (0, 220, 60)
C_YELLOW  = (0, 220, 220)
C_ORANGE  = (0, 140, 255)
C_RED     = (30, 30, 230)
C_WHITE   = (230, 230, 230)
C_DIM     = (100, 100, 130)
C_BG      = (14, 14, 30)
C_BORDER  = (0, 180, 100)

ALERT_COLORS  = [C_GREEN, C_YELLOW, C_ORANGE, C_RED]
ALERT_LABELS  = ["SAFE", "CAUTION", "WARNING", "DANGER"]


# ═══════════════════════════════════════════════════════════════════════════
# ALARM  (non-blocking threaded, cross-platform)
# ═══════════════════════════════════════════════════════════════════════════
_alarm_active = False
_alarm_thread = None

def _alarm_loop():
    system = platform.system()
    while _alarm_active:
        try:
            if system == "Darwin":
                subprocess.run(["afplay", "/System/Library/Sounds/Sosumi.aiff"],
                               capture_output=True, timeout=3)
            elif system == "Windows":
                import winsound
                winsound.Beep(1200, 600)
            else:  # Linux
                subprocess.run(
                    ["paplay", "/usr/share/sounds/freedesktop/stereo/alarm-clock-elapsed.oga"],
                    capture_output=True, timeout=3)
        except Exception:
            print("\a", end="", flush=True)
        time.sleep(0.9)

def start_alarm():
    global _alarm_active, _alarm_thread
    if not _alarm_active:
        _alarm_active = True
        _alarm_thread = threading.Thread(target=_alarm_loop, daemon=True)
        _alarm_thread.start()

def stop_alarm():
    global _alarm_active
    _alarm_active = False


# ═══════════════════════════════════════════════════════════════════════════
# METRIC CALCULATIONS
# ═══════════════════════════════════════════════════════════════════════════
def _pt(lm, idx, W, H):
    l = lm[idx]
    return (l.x * W, l.y * H)

def eye_aspect_ratio(lm, indices, W, H):
    """EAR = (‖P2–P6‖ + ‖P3–P5‖) / (2 · ‖P1–P4‖)"""
    p = [_pt(lm, i, W, H) for i in indices]
    A = math.dist(p[1], p[5])
    B = math.dist(p[2], p[4])
    C = math.dist(p[0], p[3])
    return (A + B) / (2.0 * C + 1e-7)

def mouth_aspect_ratio(lm, W, H):
    """MAR = vertical / horizontal mouth opening"""
    top    = _pt(lm, MOUTH_TOP,    W, H)
    bottom = _pt(lm, MOUTH_BOTTOM, W, H)
    left   = _pt(lm, MOUTH_LEFT,   W, H)
    right  = _pt(lm, MOUTH_RIGHT,  W, H)
    return math.dist(top, bottom) / (math.dist(left, right) + 1e-7)

def head_pitch_deg(lm, W, H):
    """Estimate forward head pitch from nose→chin vector (+ = drooping forward)."""
    nose = np.array([lm[NOSE_TIP].x * W, lm[NOSE_TIP].y * H])
    chin = np.array([lm[CHIN].x     * W, lm[CHIN].y     * H])
    vec  = chin - nose
    # positive pitch = chin lower than expected (head tipping forward)
    return math.degrees(math.atan2(vec[1], abs(vec[0]) + 1e-7)) - 90.0


# ═══════════════════════════════════════════════════════════════════════════
# DRAWING HELPERS
# ═══════════════════════════════════════════════════════════════════════════
def _put(frame, text, xy, scale=0.48, color=C_WHITE, thickness=1):
    cv2.putText(frame, text, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness,
                cv2.LINE_AA)

def draw_progress_bar(frame, x, y, w, h, value, max_val, fill_color,
                      bg=(35, 35, 55), border=(70, 70, 100), label=""):
    cv2.rectangle(frame, (x, y), (x + w, y + h), bg, -1)
    fill = int(w * min(max(value, 0) / max(max_val, 1e-7), 1.0))
    if fill > 0:
        cv2.rectangle(frame, (x, y), (x + fill, y + h), fill_color, -1)
    cv2.rectangle(frame, (x, y), (x + w, y + h), border, 1)
    if label:
        _put(frame, label, (x + 4, y + h - 3), 0.30, C_WHITE, 1)

def draw_ear_graph(frame, history, x, y, w, h, threshold, max_ear=0.45):
    """Fast pure-OpenCV EAR sparkline — no matplotlib overhead."""
    cv2.rectangle(frame, (x, y), (x + w, y + h), C_BG, -1)
    cv2.rectangle(frame, (x, y), (x + w, y + h), (50, 50, 80), 1)
    pts = list(history)
    if len(pts) < 2:
        return
    n = len(pts)
    # Threshold line
    ty = y + int(h * (1.0 - threshold / max_ear))
    ty = max(y + 1, min(ty, y + h - 1))
    cv2.line(frame, (x, ty), (x + w, ty), (60, 60, 160), 1)
    _put(frame, f"{threshold:.2f}", (x + w - 34, ty - 2), 0.28, (120, 120, 200), 1)
    # Curve
    for i in range(1, n):
        x1 = x + int((i - 1) / (n - 1) * w)
        x2 = x + int(i       / (n - 1) * w)
        y1 = y + int(h * (1.0 - min(pts[i - 1], max_ear) / max_ear))
        y2 = y + int(h * (1.0 - min(pts[i],     max_ear) / max_ear))
        c  = C_RED if pts[i] < threshold else C_NEON
        cv2.line(frame, (x1, y1), (x2, y2), c, 1, cv2.LINE_AA)

def draw_face_contours(frame, lm, W, H, ear, threshold):
    for contour_idx, color_open, color_closed in [
        (LEFT_EYE_CTR,  C_NEON, C_RED),
        (RIGHT_EYE_CTR, C_NEON, C_RED),
    ]:
        pts = np.array([(int(lm[i].x * W), int(lm[i].y * H)) for i in contour_idx])
        c   = color_open if ear >= threshold else color_closed
        cv2.polylines(frame, [pts], isClosed=True, color=c, thickness=1, lineType=cv2.LINE_AA)

def format_time(secs):
    m, s = divmod(int(secs), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


# ═══════════════════════════════════════════════════════════════════════════
# MAIN APPLICATION
# ═══════════════════════════════════════════════════════════════════════════
def main():
    # ── MediaPipe face mesh ──────────────────────────────────────────
    mp_mesh = mp.solutions.face_mesh
    face_mesh = mp_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    # ── Camera ───────────────────────────────────────────────────────
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Cannot open camera."); sys.exit(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    WIN = "Driver Fatigue Detection  |  v2.0  |  Q=Quit  R=Reset  S=Screenshot"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, 1280, 720)

    # ── Logging ──────────────────────────────────────────────────────
    os.makedirs("logs", exist_ok=True)
    log_path = f"logs/session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    log_f    = open(log_path, "w", newline="")
    csvw     = csv.writer(log_f)
    csvw.writerow(["Timestamp", "Event", "EAR", "MAR", "Head_Pitch_deg",
                   "PERCLOS_pct", "Consec_Closed_Frames", "Duration_s"])

    def log_event(event, ear, mar, pitch, perclos, consec, dur=""):
        csvw.writerow([datetime.now().strftime("%H:%M:%S"), event,
                       f"{ear:.4f}", f"{mar:.4f}", f"{pitch:.2f}",
                       f"{perclos*100:.2f}", consec, dur])
        log_f.flush()

    # ── Rolling state ────────────────────────────────────────────────
    perclos_buf   = deque(maxlen=int(ASSUMED_FPS * PERCLOS_WINDOW_SEC))
    ear_graph_buf = deque(maxlen=120)

    eye_closed_cnt  = 0
    yawn_cnt        = 0
    alert_level     = ALERT_SAFE
    alarm_on        = False
    alarm_start_t   = 0.0

    session_start   = time.time()
    total_alerts    = 0
    total_yawns     = 0

    # ── Calibration state ────────────────────────────────────────────
    calibrating    = True
    calib_start    = time.time()
    calib_ears     = []
    ear_threshold  = EAR_THRESHOLD_DEFAULT

    print("\n" + "═" * 66)
    print("  DRIVER FATIGUE DETECTION SYSTEM  v2.0")
    print("  AI: MediaPipe FaceMesh + EAR + MAR + PERCLOS + Head Pitch")
    print(f"  Log → {log_path}")
    print("  Q = quit    R = reset session    S = screenshot")
    print("═" * 66 + "\n")

    # ════════════════════════════════════════════════════════════════
    # MAIN LOOP
    # ════════════════════════════════════════════════════════════════
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame      = cv2.flip(frame, 1)
        H, W       = frame.shape[:2]
        SIDE_W     = 310                           # sidebar width
        CAM_W      = W - SIDE_W                    # camera area width
        now        = time.time()

        rgb        = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results    = face_mesh.process(rgb)

        # ── Per-frame metrics ────────────────────────────────────────
        ear  = 0.0
        mar  = 0.0
        pitch = 0.0
        face_ok = bool(results.multi_face_landmarks)

        if face_ok:
            lm    = results.multi_face_landmarks[0].landmark
            l_ear = eye_aspect_ratio(lm, LEFT_EYE_EAR,  W, H)
            r_ear = eye_aspect_ratio(lm, RIGHT_EYE_EAR, W, H)
            ear   = (l_ear + r_ear) / 2.0
            mar   = mouth_aspect_ratio(lm, W, H)
            pitch = head_pitch_deg(lm, W, H)
            draw_face_contours(frame, lm, W, H, ear, ear_threshold)

        perclos_buf.append(1 if (face_ok and ear < ear_threshold) else 0)
        ear_graph_buf.append(ear)
        perclos = sum(perclos_buf) / max(len(perclos_buf), 1)

        # ════════════════════════════════════════════════════════════
        # CALIBRATION PHASE
        # ════════════════════════════════════════════════════════════
        if calibrating:
            elapsed  = now - calib_start
            progress = min(elapsed / CALIB_DURATION, 1.0)
            if face_ok and ear > 0.15:
                calib_ears.append(ear)

            # Dark overlay over camera feed
            overlay = np.zeros_like(frame)
            frame   = cv2.addWeighted(frame, 0.35, overlay, 0.65, 0)

            cx, cy = CAM_W // 2, H // 2
            _put(frame, "CALIBRATING", (cx - 130, cy - 60), 1.4, C_NEON, 3)
            _put(frame, "Keep your eyes open & face the camera",
                 (cx - 200, cy - 20), 0.56, C_WHITE, 1)
            _put(frame, f"Sampling baseline EAR for {CALIB_DURATION:.0f} seconds...",
                 (cx - 195, cy + 12), 0.48, C_DIM, 1)

            bw = 420
            bx, by = cx - bw // 2, cy + 35
            draw_progress_bar(frame, bx, by, bw, 24, progress, 1.0, C_NEON)
            _put(frame, f"{int(progress * 100)}%", (cx - 12, by + 17), 0.5, C_BG, 2)

            if elapsed >= CALIB_DURATION:
                calibrating = False
                if len(calib_ears) >= 5:
                    base = float(np.percentile(calib_ears, 20))   # robust lower bound
                    ear_threshold = max(0.15, min(base * 0.80, 0.26))
                print(f"[CALIB] EAR threshold set to {ear_threshold:.4f}  "
                      f"(from {len(calib_ears)} samples)")

            cv2.imshow(WIN, frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            continue

        # ════════════════════════════════════════════════════════════
        # DROWSINESS LOGIC
        # ════════════════════════════════════════════════════════════
        eyes_closed = face_ok and (ear < ear_threshold)
        yawning     = face_ok and (mar > MAR_THRESHOLD)
        nodding     = face_ok and (pitch > NOD_PITCH_THRESH)

        # Eye closed counter
        if eyes_closed:
            eye_closed_cnt += 1
        else:
            eye_closed_cnt = 0

        # Yawn counter
        if yawning:
            yawn_cnt += 1
            if yawn_cnt == YAWN_CONSEC_FRAMES:
                total_yawns += 1
                log_event("YAWN", ear, mar, pitch, perclos, eye_closed_cnt)
                print(f"[YAWN] #{total_yawns}  MAR={mar:.3f}")
        else:
            yawn_cnt = 0

        # Alert level from eye-closed frames AND perclos
        if not face_ok:
            new_alert = ALERT_SAFE
        elif (eye_closed_cnt >= EYE_CLOSED_DANGER
              or perclos >= PERCLOS_DANGER):
            new_alert = ALERT_DANGER
        elif (eye_closed_cnt >= EYE_CLOSED_WARNING
              or perclos >= PERCLOS_WARNING):
            new_alert = ALERT_WARNING
        elif (eye_closed_cnt >= EYE_CLOSED_CAUTION
              or perclos >= PERCLOS_CAUTION):
            new_alert = ALERT_CAUTION
        else:
            new_alert = ALERT_SAFE

        alert_level = new_alert

        # Alarm management
        if alert_level == ALERT_DANGER and not alarm_on:
            alarm_on      = True
            alarm_start_t = now
            total_alerts += 1
            start_alarm()
            log_event("DROWSY_ALARM", ear, mar, pitch, perclos, eye_closed_cnt)
            print(f"[ALARM] #{total_alerts}  EAR={ear:.3f}  PERCLOS={perclos*100:.1f}%")

        if alert_level < ALERT_DANGER and alarm_on:
            dur = now - alarm_start_t
            stop_alarm()
            alarm_on = False
            log_event("RESOLVED", ear, mar, pitch, perclos, eye_closed_cnt, f"{dur:.1f}")
            print(f"[RESOLVED] Alarm lasted {dur:.1f}s")

        # ════════════════════════════════════════════════════════════
        # RENDER  — camera area alerts
        # ════════════════════════════════════════════════════════════
        if alert_level == ALERT_DANGER:
            # Pulsing red vignette
            pulse = 0.25 + 0.15 * abs(math.sin(now * 5))
            ov    = frame.copy()
            cv2.rectangle(ov, (0, 0), (CAM_W, H), (0, 0, 200), -1)
            cv2.addWeighted(ov, pulse, frame, 1 - pulse, 0, frame)

            # Central alert banner
            bh = 80
            cv2.rectangle(frame, (0, H // 2 - bh // 2),
                          (CAM_W, H // 2 + bh // 2), (0, 0, 160), -1)
            blink_c = C_RED if int(now * 5) % 2 == 0 else C_WHITE
            _put(frame, "!  DROWSINESS DETECTED  !",
                 (30, H // 2 + 14), 1.1, blink_c, 3)

            # Flashing border
            bc = C_RED if int(now * 4) % 2 == 0 else (20, 20, 140)
            cv2.rectangle(frame, (3, 3), (CAM_W - 3, H - 3), bc, 5)

        elif alert_level == ALERT_WARNING:
            cv2.rectangle(frame, (3, 3), (CAM_W - 3, H - 3), C_ORANGE, 3)
            _put(frame, "! STAY ALERT", (20, 55), 1.0, C_ORANGE, 2)

        elif alert_level == ALERT_CAUTION:
            cv2.rectangle(frame, (3, 3), (CAM_W - 3, H - 3), C_YELLOW, 2)

        else:
            cv2.rectangle(frame, (3, 3), (CAM_W - 3, H - 3), C_BORDER, 1)

        # Secondary notifications
        if yawning:
            _put(frame, "YAWN DETECTED", (20, H - 30), 0.85, C_YELLOW, 2)
        if nodding:
            _put(frame, "HEAD NOD", (20, H - 70), 0.85, C_ORANGE, 2)

        # ════════════════════════════════════════════════════════════
        # SIDEBAR
        # ════════════════════════════════════════════════════════════
        cv2.rectangle(frame, (CAM_W, 0), (W, H), C_BG, -1)
        cv2.line(frame, (CAM_W, 0), (CAM_W, H), C_BORDER, 1)

        sx  = CAM_W + 10
        sy  = 12
        sw  = SIDE_W - 18   # usable width inside sidebar

        # Title
        _put(frame, "FATIGUE MONITOR", (sx, sy + 18), 0.62, C_NEON, 2)
        cv2.line(frame, (sx, sy + 26), (W - 8, sy + 26), C_NEON, 1)
        sy += 42

        # Session time
        _put(frame, f"Session  {format_time(now - session_start)}",
             (sx, sy + 14), 0.50, C_WHITE, 1)
        sy += 32

        # Status badge
        al_c = ALERT_COLORS[alert_level]
        al_l = ALERT_LABELS[alert_level]
        cv2.rectangle(frame, (sx, sy), (sx + sw, sy + 28), al_c, -1)
        _put(frame, f"  STATUS: {al_l}", (sx + 4, sy + 20), 0.62, C_BG, 2)
        sy += 38

        # ── EAR bar ─────────────────────────────────────────────────
        _put(frame, f"Eye Aspect Ratio (EAR)  {ear:.3f}", (sx, sy + 12), 0.44, C_WHITE, 1)
        sy += 18
        ear_c = C_GREEN if ear >= ear_threshold else C_RED
        draw_progress_bar(frame, sx, sy, sw, 14, ear, 0.45, ear_c)
        # threshold tick
        tx = sx + int(sw * ear_threshold / 0.45)
        cv2.line(frame, (tx, sy), (tx, sy + 14), C_WHITE, 2)
        sy += 24

        # ── MAR bar ─────────────────────────────────────────────────
        _put(frame, f"Mouth Aspect Ratio (MAR)  {mar:.3f}", (sx, sy + 12), 0.44, C_WHITE, 1)
        sy += 18
        mar_c = C_YELLOW if mar >= MAR_THRESHOLD else C_GREEN
        draw_progress_bar(frame, sx, sy, sw, 14, mar, 1.0, mar_c)
        sy += 24

        # ── PERCLOS bar ─────────────────────────────────────────────
        _put(frame, f"PERCLOS (60s window)  {perclos*100:.1f}%", (sx, sy + 12), 0.44, C_WHITE, 1)
        sy += 18
        if perclos >= PERCLOS_DANGER:
            pc = C_RED
        elif perclos >= PERCLOS_WARNING:
            pc = C_ORANGE
        elif perclos >= PERCLOS_CAUTION:
            pc = C_YELLOW
        else:
            pc = C_GREEN
        draw_progress_bar(frame, sx, sy, sw, 14, perclos, PERCLOS_DANGER * 1.5, pc)
        sy += 24

        # ── Head pitch bar ──────────────────────────────────────────
        _put(frame, f"Head Pitch  {pitch:.1f}°  {'⚠ NOD' if nodding else ''}",
             (sx, sy + 12), 0.44, C_ORANGE if nodding else C_WHITE, 1)
        sy += 18
        draw_progress_bar(frame, sx, sy, sw, 14, max(pitch, 0), 40,
                          C_ORANGE if nodding else C_GREEN)
        sy += 28

        cv2.line(frame, (sx, sy), (W - 8, sy), (40, 40, 65), 1)
        sy += 12

        # ── EAR graph ───────────────────────────────────────────────
        _put(frame, "EAR TREND", (sx, sy + 12), 0.42, C_DIM, 1)
        sy += 16
        draw_ear_graph(frame, ear_graph_buf, sx, sy, sw, 75, ear_threshold)
        sy += 85

        cv2.line(frame, (sx, sy), (W - 8, sy), (40, 40, 65), 1)
        sy += 12

        # ── Session stats ────────────────────────────────────────────
        _put(frame, "SESSION STATS", (sx, sy + 12), 0.42, C_DIM, 1)
        sy += 22
        for label, val in [
            ("Drowsy alerts:", str(total_alerts)),
            ("Yawn events:",   str(total_yawns)),
            ("Eye closed:",    f"{eye_closed_cnt} frames"),
            ("Threshold:",     f"{ear_threshold:.3f} (calibrated)"),
        ]:
            _put(frame, label, (sx, sy + 12), 0.42, C_DIM, 1)
            _put(frame, val,   (sx + sw - len(val) * 8, sy + 12), 0.42, C_WHITE, 1)
            sy += 18
        sy += 6

        # Face-detected indicator
        fd_c = C_GREEN if face_ok else C_RED
        fd_l = "● FACE DETECTED" if face_ok else "● NO FACE"
        _put(frame, fd_l, (sx, sy + 12), 0.44, fd_c, 1)
        sy += 22

        # Log file name
        _put(frame, os.path.basename(log_path), (sx, sy + 12), 0.33, C_DIM, 1)

        # Bottom keybinds
        cv2.line(frame, (sx, H - 28), (W - 8, H - 28), (40, 40, 65), 1)
        _put(frame, "Q: Quit   R: Reset   S: Screenshot",
             (sx, H - 10), 0.36, C_DIM, 1)

        # ── Show ────────────────────────────────────────────────────
        cv2.imshow(WIN, frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break
        elif key == ord("r"):
            total_alerts = total_yawns = 0
            session_start = time.time()
            perclos_buf.clear()
            ear_graph_buf.clear()
            eye_closed_cnt = 0
            print("[RESET] Session counters reset.")
        elif key == ord("s"):
            fname = f"screenshot_{datetime.now().strftime('%H%M%S')}.png"
            cv2.imwrite(fname, frame)
            print(f"[SCREENSHOT] → {fname}")

    # ── Cleanup ───────────────────────────────────────────────────────
    stop_alarm()
    cap.release()
    face_mesh.close()
    cv2.destroyAllWindows()
    log_f.close()

    duration = time.time() - session_start
    print("\n" + "═" * 54)
    print("  SESSION COMPLETE")
    print(f"  Duration:      {format_time(duration)}")
    print(f"  Drowsy alerts: {total_alerts}")
    print(f"  Yawn events:   {total_yawns}")
    print(f"  Log saved:     {log_path}")
    print("═" * 54 + "\n")


if __name__ == "__main__":
    main()