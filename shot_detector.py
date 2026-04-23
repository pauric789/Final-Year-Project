from ultralytics import YOLO
import cv2
import cvzone
import mediapipe as mp
import numpy as np
import joblib
import pandas as pd
import threading
import subprocess
import shutil
from xgboost import XGBClassifier
from utils import detect_down, detect_up, in_hoop_region, clean_hoop_pos, clean_ball_pos, get_device, score, calculate_angle, calculate_distance


class ShotDetector:
    def __init__(self):

        self.model = YOLO("best.pt")
        self.class_names = ['Basketball', 'Basketball Hoop']
        self.device = get_device()

        # self.cap = cv2.VideoCapture(0) 
        self.cap = cv2.VideoCapture("test.mp4")
        self.ball_pos = []
        self.hoop_pos = []
        self.frame_count = 0
        self.frame = None

        self.makes = 0
        self.attempts = 0

        # Shot detection state
        self.up = False
        self.down = False
        self.up_frame = 0
        self.down_frame = 0

        # Delayed evaluation
        self.pending_shot = False
        self.shot_eval_frame = 0
        self.shot_delay = 15

        # Cooldown to prevent double-counting
        self.shot_cooldown = 0
        self.cooldown_frames = 45
        self.max_up_frames_without_down = 30
        self.early_prediction_delay_frames = 4
        self.early_prediction_done = False

        # Visual feedback
        self.fade_frames = 20
        self.fade_counter = 0
        self.overlay_color = (0, 0, 0)
        self.overlay_text = ""

        # XGBoost prediction display
        self.xgb_prediction_text = ""
        self.xgb_prediction_prob = None   # float 0-1
        self.xgb_display_counter = 0
        self.xgb_display_frames = 60      # how many frames to show prediction
        self._pending_metrics = {}

        # Audio prediction — uses PowerShell's built-in speech synthesis
        # instead of pyttsx3 (which breaks after one call when threaded).
        self.tts_enabled = True
        self.last_spoken_prediction = ""

        # Load XGBoost model and feature names
        try:
            self.xgb_model = XGBClassifier()
            self.xgb_model.load_model("xgboost_shot_model.json")
            self.xgb_feature_names = joblib.load("xgboost_feature_names.pkl")
            if not self.xgb_feature_names:
                booster_features = self.xgb_model.get_booster().feature_names
                self.xgb_feature_names = booster_features if booster_features else []
            print("XGBoost model loaded successfully.")
        except Exception as e:
            print(f"Warning: could not load XGBoost model: {e}")
            self.xgb_model = None
            self.xgb_feature_names = []

        # MediaPipe pose
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.pose_landmarks_needed = [
            self.mp_pose.PoseLandmark.LEFT_WRIST,
            self.mp_pose.PoseLandmark.RIGHT_WRIST,
            self.mp_pose.PoseLandmark.LEFT_ELBOW,
            self.mp_pose.PoseLandmark.RIGHT_ELBOW,
            self.mp_pose.PoseLandmark.LEFT_KNEE,
            self.mp_pose.PoseLandmark.RIGHT_KNEE,
            self.mp_pose.PoseLandmark.LEFT_SHOULDER,
            self.mp_pose.PoseLandmark.RIGHT_SHOULDER,
            self.mp_pose.PoseLandmark.LEFT_ANKLE,
            self.mp_pose.PoseLandmark.RIGHT_ANKLE,
            self.mp_pose.PoseLandmark.LEFT_HEEL,
            self.mp_pose.PoseLandmark.RIGHT_HEEL,
            self.mp_pose.PoseLandmark.LEFT_FOOT_INDEX,
            self.mp_pose.PoseLandmark.RIGHT_FOOT_INDEX,
            self.mp_pose.PoseLandmark.LEFT_HIP,
            self.mp_pose.PoseLandmark.RIGHT_HIP,
        ]
        self.pose_points = {}

        # Data tracking for per-shot metrics
        self.current_shot_frames = []

        # Track if currently tracking a shot
        self.tracking_shot = False
        self.shot_start_frame = 0

        self.run()

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self):
        while True:
            ret, self.frame = self.cap.read()
            if not ret:
                break

            results = self.model(self.frame, stream=True, device=self.device)

            frame_ball_pos = None
            frame_hoop_pos = None

            for r in results:
                boxes = r.boxes
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    w, h = x2 - x1, y2 - y1
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    current_class = self.class_names[cls]
                    center = (x1 + w // 2, y1 + h // 2)

                    if (conf > 0.01 or (in_hoop_region(center, self.hoop_pos) and conf > 0.01)) \
                            and current_class == "Basketball":
                        self.ball_pos.append((center, self.frame_count, w, h, conf))
                        frame_ball_pos = center
                        cvzone.cornerRect(self.frame, (x1, y1, w, h))

                    if conf > .1 and current_class == "Basketball Hoop":
                        self.hoop_pos.append((center, self.frame_count, w, h, conf))
                        frame_hoop_pos = center
                        cvzone.cornerRect(self.frame, (x1, y1, w, h))

            self.process_pose()
            self.clean_motion()

            if self.tracking_shot:
                frame_data = self._build_frame_data(frame_ball_pos, frame_hoop_pos)
                self.current_shot_frames.append(frame_data)

            self.shot_detection()
            self.display_score()
            self.display_xgb_prediction()

            if self.shot_cooldown > 0:
                self.shot_cooldown -= 1

            self.frame_count += 1
            cv2.imshow('Frame', self.frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()

    # ------------------------------------------------------------------
    # Frame data builder
    # ------------------------------------------------------------------

    def _build_frame_data(self, frame_ball_pos, frame_hoop_pos):
        """Build a dict of pose + ball/hoop state for the current frame."""
        data = {
            'frame': self.frame_count,
            'ball_x': frame_ball_pos[0] if frame_ball_pos else None,
            'ball_y': frame_ball_pos[1] if frame_ball_pos else None,
            'hoop_x': frame_hoop_pos[0] if frame_hoop_pos else None,
            'hoop_y': frame_hoop_pos[1] if frame_hoop_pos else None,
            'in_up_region': self.up,
        }
        for name, (x, y, vis) in self.pose_points.items():
            data[f'{name}_x'] = x
            data[f'{name}_y'] = y
            data[f'{name}_vis'] = vis
        return data

    # ------------------------------------------------------------------
    # XGBoost prediction
    # ------------------------------------------------------------------

    def predict_shot(self, metrics: dict):
        """Run XGBoost on release-point metrics and store result for display."""
        if self.xgb_model is None:
            return

        metrics = metrics or {}

        if not self.xgb_feature_names:
            booster_features = self.xgb_model.get_booster().feature_names
            if booster_features:
                self.xgb_feature_names = booster_features
            else:
                print("[XGBoost] Prediction skipped: feature names are unavailable.")
                return

        # Build a single-row DataFrame aligned to training feature names.
        # Missing features are filled with 0 (same as training-time NaN handling).
        row = {feat: metrics.get(feat, 0) for feat in self.xgb_feature_names}
        X = pd.DataFrame([row], columns=self.xgb_feature_names)

        try:
            prob = float(self.xgb_model.predict_proba(X)[0][1])   # P(make)
        except Exception as e:
            print(f"[XGBoost] Prediction failed: {e}")
            return

        self.xgb_prediction_prob = prob
        self.xgb_prediction_text = f"Pred: {'Make' if prob >= 0.5 else 'Miss'} ({prob * 100:.0f}%)"
        self.xgb_display_counter = self.xgb_display_frames

        print(f"[XGBoost] {self.xgb_prediction_text}")
        self.speak_prediction(prob)

    def speak_prediction(self, prob):
        """Speak prediction using eSpeak in a background thread."""
        if not self.tts_enabled:
            return

        label = "make" if prob >= 0.5 else "miss"
        phrase = label

        # Avoid repeating the exact same phrase frame-after-frame.
        if phrase == self.last_spoken_prediction:
            return
        self.last_spoken_prediction = phrase

        def _speak():
            try:
                tts_cmd = None
                candidates = [
                    "espeak-ng",
                    "espeak",
                    r"C:\Program Files\eSpeak NG\espeak-ng.exe",
                    r"C:\Program Files (x86)\eSpeak NG\espeak-ng.exe",
                    r"C:\Program Files\eSpeak\command_line\espeak.exe",
                    r"C:\Program Files (x86)\eSpeak\command_line\espeak.exe",
                ]

                for candidate in candidates:
                    resolved = shutil.which(candidate)
                    if resolved:
                        tts_cmd = resolved
                        break

                if tts_cmd is None:
                    print("[TTS] eSpeak not found. Install espeak-ng or espeak and ensure it is in PATH.")
                    return

                print(f"[TTS] Speaking '{phrase}' via: {tts_cmd}")
                subprocess.run([tts_cmd, "-a", "180", "-s", "160", phrase], check=False)
            except Exception as e:
                print(f"[TTS] Speech failed: {e}")

        threading.Thread(target=_speak, daemon=True).start()

    def display_xgb_prediction(self):
        """Draw the XGBoost prediction overlay while the display counter is active."""
        if self.xgb_display_counter <= 0 or not self.xgb_prediction_text:
            return

        prob = self.xgb_prediction_prob if self.xgb_prediction_prob is not None else 0.5
        alpha_factor = self.xgb_display_counter / self.xgb_display_frames

        # Colour gradient: green = confident make, red = confident miss, amber = uncertain
        if prob >= 0.65:
            color = (72, 220, 80)       # green
            bg_accent = (30, 80, 35)
        elif prob <= 0.35:
            color = (80, 80, 255)       # red
            bg_accent = (40, 30, 80)
        else:
            color = (60, 200, 255)      # amber
            bg_accent = (30, 70, 80)

        label = "MAKE" if prob >= 0.5 else "MISS"
        pct_text = f"{prob * 100:.0f}%"
        font = cv2.FONT_HERSHEY_SIMPLEX

        h_frame, w_frame = self.frame.shape[:2]

        # --- Pill badge centred at bottom ---
        badge_w, badge_h = 280, 70
        bx = (w_frame - badge_w) // 2
        by = h_frame - badge_h - 25

        overlay = self.frame.copy()
        cv2.rectangle(overlay, (bx, by), (bx + badge_w, by + badge_h), bg_accent, -1)
        cv2.addWeighted(overlay, 0.7 * alpha_factor,
                        self.frame, 1 - 0.7 * alpha_factor, 0, self.frame)
        cv2.rectangle(self.frame, (bx, by), (bx + badge_w, by + badge_h), color, 2)

        # Label text (e.g. "MAKE")
        cv2.putText(self.frame, label, (bx + 18, by + 45),
                    font, 1.3, color, 3)

        # Percentage text right-aligned
        (pw, _), _ = cv2.getTextSize(pct_text, font, 1.3, 3)
        cv2.putText(self.frame, pct_text, (bx + badge_w - pw - 18, by + 45),
                    font, 1.3, (255, 255, 255), 3)

        # Confidence bar at the bottom of the badge
        bar_pad = 12
        bar_y = by + badge_h - 14
        bar_w = badge_w - 2 * bar_pad
        bar_h = 6
        cv2.rectangle(self.frame, (bx + bar_pad, bar_y),
                      (bx + bar_pad + bar_w, bar_y + bar_h), (50, 50, 50), -1)
        cv2.rectangle(self.frame, (bx + bar_pad, bar_y),
                      (bx + bar_pad + int(bar_w * prob), bar_y + bar_h), color, -1)

        self.xgb_display_counter -= 1

    # ------------------------------------------------------------------
    # Pose processing
    # ------------------------------------------------------------------

    # Skeleton connections for drawing limb lines
    _SKELETON_CONNECTIONS = [
        ('LEFT_SHOULDER', 'LEFT_ELBOW'), ('LEFT_ELBOW', 'LEFT_WRIST'),
        ('RIGHT_SHOULDER', 'RIGHT_ELBOW'), ('RIGHT_ELBOW', 'RIGHT_WRIST'),
        ('LEFT_SHOULDER', 'RIGHT_SHOULDER'),
        ('LEFT_SHOULDER', 'LEFT_HIP'), ('RIGHT_SHOULDER', 'RIGHT_HIP'),
        ('LEFT_HIP', 'RIGHT_HIP'),
        ('LEFT_HIP', 'LEFT_KNEE'), ('LEFT_KNEE', 'LEFT_ANKLE'),
        ('RIGHT_HIP', 'RIGHT_KNEE'), ('RIGHT_KNEE', 'RIGHT_ANKLE'),
    ]

    def process_pose(self):
        rgb = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb)
        self.pose_points = {}

        if not results.pose_landmarks:
            return

        h, w = self.frame.shape[:2]
        for lm in self.pose_landmarks_needed:
            landmark = results.pose_landmarks.landmark[lm]
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            self.pose_points[lm.name] = (x, y, landmark.visibility)

        # Draw skeleton lines first (behind the dots)
        for a, b in self._SKELETON_CONNECTIONS:
            if a in self.pose_points and b in self.pose_points:
                pa = self.pose_points[a]
                pb = self.pose_points[b]
                if pa[2] > 0.4 and pb[2] > 0.4:
                    cv2.line(self.frame, (pa[0], pa[1]), (pb[0], pb[1]),
                             (200, 170, 50), 2, cv2.LINE_AA)

        # Draw joint dots on top
        for name, (x, y, vis) in self.pose_points.items():
            if vis > 0.4:
                cv2.circle(self.frame, (x, y), 6, (30, 30, 30), -1)
                cv2.circle(self.frame, (x, y), 4, (0, 220, 255), -1)

    def clean_motion(self):
        self.ball_pos = clean_ball_pos(self.ball_pos, self.frame_count)

        # Draw ball trail with fading opacity (most recent = brightest)
        num_trail = len(self.ball_pos)
        for i in range(num_trail):
            fade = max(0.25, (i + 1) / num_trail)
            radius = max(2, int(4 * fade))
            blue = int(80 + 175 * fade)
            green = int(50 * fade)
            cv2.circle(self.frame, self.ball_pos[i][0], radius,
                       (0, green, blue), -1, cv2.LINE_AA)

        if len(self.hoop_pos) > 1:
            self.hoop_pos = clean_hoop_pos(self.hoop_pos)
            # Draw a small crosshair on the hoop centre
            hx, hy = self.hoop_pos[-1][0]
            cv2.drawMarker(self.frame, (hx, hy), (0, 255, 200), cv2.MARKER_CROSS,
                           12, 2, cv2.LINE_AA)

    # ------------------------------------------------------------------
    # Shot metrics
    # ------------------------------------------------------------------

    def calculate_shot_metrics(self):
        if not self.current_shot_frames:
            return {}

        metrics = {}

        # Find release frame (first frame where ball is in up region)
        release_frame = None
        release_idx = -1
        for i, frame in enumerate(self.current_shot_frames):
            if frame['in_up_region']:
                release_frame = frame
                release_idx = i
                break

        # Find pre-release frame (5 frames before release)
        pre_release_frame = None
        if release_idx > 5:
            pre_release_frame = self.current_shot_frames[release_idx - 5]

        if release_frame:
            # Right elbow + shoulder angle at release
            if all(k in release_frame and release_frame[k] is not None for k in
                   ['RIGHT_WRIST_x', 'RIGHT_ELBOW_x', 'RIGHT_SHOULDER_x']):
                wrist    = np.array([release_frame['RIGHT_WRIST_x'],    release_frame['RIGHT_WRIST_y']])
                elbow    = np.array([release_frame['RIGHT_ELBOW_x'],    release_frame['RIGHT_ELBOW_y']])
                shoulder = np.array([release_frame['RIGHT_SHOULDER_x'], release_frame['RIGHT_SHOULDER_y']])
                metrics['right_elbow_angle']    = calculate_angle(wrist, elbow, shoulder)
                metrics['right_shoulder_angle'] = calculate_angle(shoulder, elbow, wrist)

            # Right wrist extension
            if all(k in release_frame and release_frame[k] is not None for k in
                   ['RIGHT_WRIST_y', 'RIGHT_ELBOW_y']):
                metrics['right_wrist_extension'] = release_frame['RIGHT_WRIST_y'] - release_frame['RIGHT_ELBOW_y']

            # Left arm
            if all(k in release_frame and release_frame[k] is not None for k in
                   ['LEFT_WRIST_x', 'LEFT_ELBOW_x', 'LEFT_SHOULDER_x']):
                l_wrist    = np.array([release_frame['LEFT_WRIST_x'],    release_frame['LEFT_WRIST_y']])
                l_elbow    = np.array([release_frame['LEFT_ELBOW_x'],    release_frame['LEFT_ELBOW_y']])
                l_shoulder = np.array([release_frame['LEFT_SHOULDER_x'], release_frame['LEFT_SHOULDER_y']])
                metrics['left_elbow_angle'] = calculate_angle(l_wrist, l_elbow, l_shoulder)

            # Right knee
            if all(k in release_frame and release_frame[k] is not None for k in
                   ['RIGHT_KNEE_x', 'RIGHT_ANKLE_x', 'RIGHT_HIP_x']):
                r_hip   = np.array([release_frame['RIGHT_HIP_x'],   release_frame['RIGHT_HIP_y']])
                r_knee  = np.array([release_frame['RIGHT_KNEE_x'],  release_frame['RIGHT_KNEE_y']])
                r_ankle = np.array([release_frame['RIGHT_ANKLE_x'], release_frame['RIGHT_ANKLE_y']])
                metrics['right_knee_angle'] = calculate_angle(r_hip, r_knee, r_ankle)

            # Left knee
            if all(k in release_frame and release_frame[k] is not None for k in
                   ['LEFT_KNEE_x', 'LEFT_ANKLE_x', 'LEFT_HIP_x']):
                l_hip   = np.array([release_frame['LEFT_HIP_x'],   release_frame['LEFT_HIP_y']])
                l_knee  = np.array([release_frame['LEFT_KNEE_x'],  release_frame['LEFT_KNEE_y']])
                l_ankle = np.array([release_frame['LEFT_ANKLE_x'], release_frame['LEFT_ANKLE_y']])
                metrics['left_knee_angle'] = calculate_angle(l_hip, l_knee, l_ankle)

            # Right hip
            if all(k in release_frame and release_frame[k] is not None for k in
                   ['RIGHT_SHOULDER_x', 'RIGHT_HIP_x', 'RIGHT_KNEE_x']):
                r_hip      = np.array([release_frame['RIGHT_HIP_x'],      release_frame['RIGHT_HIP_y']])
                r_shoulder = np.array([release_frame['RIGHT_SHOULDER_x'], release_frame['RIGHT_SHOULDER_y']])
                r_knee     = np.array([release_frame['RIGHT_KNEE_x'],     release_frame['RIGHT_KNEE_y']])
                metrics['right_hip_angle'] = calculate_angle(r_shoulder, r_hip, r_knee)

            # Alignment metrics
            if all(k in release_frame and release_frame[k] is not None for k in
                   ['LEFT_SHOULDER_y', 'RIGHT_SHOULDER_y']):
                metrics['shoulder_tilt'] = abs(release_frame['LEFT_SHOULDER_y'] - release_frame['RIGHT_SHOULDER_y'])

            if all(k in release_frame and release_frame[k] is not None for k in
                   ['LEFT_HIP_y', 'RIGHT_HIP_y']):
                metrics['hip_tilt'] = abs(release_frame['LEFT_HIP_y'] - release_frame['RIGHT_HIP_y'])

            if all(k in release_frame and release_frame[k] is not None for k in
                   ['LEFT_KNEE_x', 'RIGHT_KNEE_x']):
                metrics['knee_spread'] = abs(release_frame['LEFT_KNEE_x'] - release_frame['RIGHT_KNEE_x'])

            # Release height / position
            if release_frame.get('RIGHT_WRIST_y') is not None:
                metrics['release_height'] = release_frame['RIGHT_WRIST_y']
            if release_frame.get('RIGHT_WRIST_x') is not None:
                metrics['release_x_position'] = release_frame['RIGHT_WRIST_x']

            # Body vertical alignment
            if all(k in release_frame and release_frame[k] is not None for k in
                   ['RIGHT_SHOULDER_x', 'RIGHT_ANKLE_x']):
                metrics['body_vertical_alignment'] = abs(
                    release_frame['RIGHT_SHOULDER_x'] - release_frame['RIGHT_ANKLE_x'])

            # Ball-to-hoop metrics
            if release_frame.get('ball_x') is not None and release_frame.get('hoop_x') is not None:
                metrics['ball_hoop_distance_x'] = abs(release_frame['ball_x'] - release_frame['hoop_x'])
                metrics['ball_hoop_distance_y'] = abs(release_frame['ball_y'] - release_frame['hoop_y'])
                ball_arr = np.array([release_frame['ball_x'], release_frame['ball_y']])
                hoop_arr = np.array([release_frame['hoop_x'], release_frame['hoop_y']])
                metrics['ball_hoop_total_distance'] = calculate_distance(ball_arr, hoop_arr)
                if release_frame['ball_y'] != release_frame['hoop_y']:
                    metrics['release_angle_to_hoop'] = np.degrees(np.arctan2(
                        release_frame['hoop_y'] - release_frame['ball_y'],
                        release_frame['hoop_x'] - release_frame['ball_x']
                    ))

            # Pre-release comparisons
            if pre_release_frame:
                if all(k in pre_release_frame and pre_release_frame[k] is not None for k in
                       ['RIGHT_KNEE_x', 'RIGHT_ANKLE_x', 'RIGHT_HIP_x']):
                    pre_hip   = np.array([pre_release_frame['RIGHT_HIP_x'],   pre_release_frame['RIGHT_HIP_y']])
                    pre_knee  = np.array([pre_release_frame['RIGHT_KNEE_x'],  pre_release_frame['RIGHT_KNEE_y']])
                    pre_ankle = np.array([pre_release_frame['RIGHT_ANKLE_x'], pre_release_frame['RIGHT_ANKLE_y']])
                    pre_knee_angle = calculate_angle(pre_hip, pre_knee, pre_ankle)
                    if 'right_knee_angle' in metrics:
                        metrics['knee_extension'] = metrics['right_knee_angle'] - pre_knee_angle

                if pre_release_frame.get('RIGHT_HIP_y') is not None and release_frame.get('RIGHT_HIP_y') is not None:
                    metrics['jump_height'] = pre_release_frame['RIGHT_HIP_y'] - release_frame['RIGHT_HIP_y']

                if all(k in pre_release_frame and pre_release_frame[k] is not None for k in
                       ['RIGHT_WRIST_x', 'RIGHT_ELBOW_x', 'RIGHT_SHOULDER_x']):
                    pre_wrist    = np.array([pre_release_frame['RIGHT_WRIST_x'],    pre_release_frame['RIGHT_WRIST_y']])
                    pre_elbow    = np.array([pre_release_frame['RIGHT_ELBOW_x'],    pre_release_frame['RIGHT_ELBOW_y']])
                    pre_shoulder = np.array([pre_release_frame['RIGHT_SHOULDER_x'], pre_release_frame['RIGHT_SHOULDER_y']])
                    pre_elbow_angle = calculate_angle(pre_wrist, pre_elbow, pre_shoulder)
                    if 'right_elbow_angle' in metrics:
                        metrics['elbow_extension'] = metrics['right_elbow_angle'] - pre_elbow_angle

        return metrics

    # ------------------------------------------------------------------
    # Shot detection
    # ------------------------------------------------------------------

    def shot_detection(self):
        if len(self.hoop_pos) == 0 or len(self.ball_pos) == 0:
            return

        if self.shot_cooldown > 0:
            return

        # Detect ball in 'up' region (release point)
        if not self.up:
            self.up = detect_up(self.ball_pos, self.hoop_pos, min_consecutive=1)
            if self.up:
                self.up_frame = self.ball_pos[-1][1]
                self.tracking_shot = True
                self.shot_start_frame = self.frame_count
                self.current_shot_frames = []
                self.early_prediction_done = False

        # Early prediction path: show probability shortly after release,
        # before the ball reaches the rim.
        if self.up and not self.early_prediction_done:
            if self.frame_count - self.up_frame >= self.early_prediction_delay_frames:
                early_metrics = self.calculate_shot_metrics()
                self._pending_metrics = early_metrics if early_metrics else self._pending_metrics
                self.predict_shot(early_metrics)
                self.early_prediction_done = True

        # Detect ball in 'down' region — this is the moment of release confirmation,
        # so we run the XGBoost prediction here while we have a full arc of frames.
        if self.up and not self.down:
            self.down = detect_down(self.ball_pos, self.hoop_pos)
            if self.down:
                self.down_frame = self.ball_pos[-1][1]
                self._pending_metrics = self.calculate_shot_metrics()
                self.predict_shot(self._pending_metrics)

            # Fallback: if down is missed, force evaluation after a timeout.
            elif self.frame_count - self.up_frame >= self.max_up_frames_without_down:
                self.down = True
                self.down_frame = self.ball_pos[-1][1]
                self._pending_metrics = self.calculate_shot_metrics()
                self.predict_shot(self._pending_metrics)

        # Mark shot as pending for delayed make/miss evaluation
        if self.up and self.down and not self.pending_shot:
            self.pending_shot = True
            self.shot_eval_frame = self.frame_count

        # Evaluate make/miss after delay
        if self.pending_shot:
            if self.frame_count - self.shot_eval_frame >= self.shot_delay:
                # Fallback: if release-time prediction was missed, predict now.
                if self.xgb_display_counter <= 0:
                    fallback_metrics = self._pending_metrics if self._pending_metrics else self.calculate_shot_metrics()
                    self.predict_shot(fallback_metrics)

                self.attempts += 1
                is_make = score(self.ball_pos, self.hoop_pos)

                if is_make:
                    self.makes += 1
                    self.overlay_color = (0, 255, 0)
                    self.overlay_text = "Make"
                else:
                    self.overlay_color = (0, 0, 255)
                    self.overlay_text = "Miss"

                self.fade_counter = self.fade_frames

                # Reset state
                self.up = False
                self.down = False
                self.pending_shot = False
                self.shot_cooldown = self.cooldown_frames
                self.tracking_shot = False
                self.current_shot_frames = []
                self._pending_metrics = {}
                self.early_prediction_done = False
                self.last_spoken_prediction = ""

    # ------------------------------------------------------------------
    # Display helpers
    # ------------------------------------------------------------------

    def display_score(self):
        """Draw a translucent HUD bar at the top with score and shooting %."""
        h_frame, w_frame = self.frame.shape[:2]
        bar_h = 60

        # Semi-transparent dark bar across the top
        overlay = self.frame.copy()
        cv2.rectangle(overlay, (0, 0), (w_frame, bar_h), (15, 15, 15), -1)
        cv2.addWeighted(overlay, 0.65, self.frame, 0.35, 0, self.frame)

        # Thin accent line at the bottom of the bar
        cv2.line(self.frame, (0, bar_h), (w_frame, bar_h), (0, 180, 255), 2)

        font = cv2.FONT_HERSHEY_SIMPLEX

        # Score: "3 / 5" on the left
        score_text = f"{self.makes} / {self.attempts}"
        cv2.putText(self.frame, score_text, (20, 42),
                    font, 1.2, (255, 255, 255), 3, cv2.LINE_AA)

        # Shooting percentage on the right
        if self.attempts > 0:
            pct = self.makes / self.attempts * 100
            pct_text = f"{pct:.0f}%"
        else:
            pct_text = "--"
        (pw, _), _ = cv2.getTextSize(pct_text, font, 1.2, 3)
        cv2.putText(self.frame, pct_text, (w_frame - pw - 20, 42),
                    font, 1.2, (0, 220, 255), 3, cv2.LINE_AA)

        # "SHOT TRACKER" label centred
        label = "SHOT TRACKER"
        (lw, _), _ = cv2.getTextSize(label, font, 0.6, 2)
        cv2.putText(self.frame, label, ((w_frame - lw) // 2, 38),
                    font, 0.6, (140, 140, 140), 2, cv2.LINE_AA)

        # --- Make / Miss result badge (top-right, below bar) ---
        if self.overlay_text and self.fade_counter > 0:
            badge_text = self.overlay_text.upper()
            alpha = self.fade_counter / self.fade_frames

            badge_font_scale = 1.5
            badge_thickness = 3
            (bw, bh), _ = cv2.getTextSize(badge_text, font,
                                           badge_font_scale, badge_thickness)
            pad = 16
            bx = w_frame - bw - pad * 2 - 15
            by = bar_h + 15

            # Pill background
            pill_overlay = self.frame.copy()
            cv2.rectangle(pill_overlay, (bx, by),
                          (bx + bw + pad * 2, by + bh + pad * 2),
                          self.overlay_color, -1)
            cv2.addWeighted(pill_overlay, 0.55 * alpha,
                            self.frame, 1 - 0.55 * alpha, 0, self.frame)
            cv2.rectangle(self.frame, (bx, by),
                          (bx + bw + pad * 2, by + bh + pad * 2),
                          self.overlay_color, 2)

            # Text
            cv2.putText(self.frame, badge_text,
                        (bx + pad, by + bh + pad - 2),
                        font, badge_font_scale, (255, 255, 255),
                        badge_thickness, cv2.LINE_AA)

            self.fade_counter -= 1

if __name__ == "__main__":
    ShotDetector()