from ultralytics import YOLO
import cv2
import cvzone
import math
import mediapipe as mp
import numpy as np
import csv
import joblib
import pandas as pd
import threading
from xgboost import XGBClassifier
from datetime import datetime
from utils import detect_down, detect_up, in_hoop_region, clean_hoop_pos, clean_ball_pos, get_device, score, calculate_angle, calculate_distance


class ShotDetector:
    def __init__(self):

        self.model = YOLO("best.pt")
        self.class_names = ['Basketball', 'Basketball Hoop']
        self.device = get_device()

        self.cap = cv2.VideoCapture("cutmyvid.mp4")
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

        # Audio prediction (optional): uses local system voice via pyttsx3.
        self.tts_enabled = False
        self.tts_engine = None
        self.tts_lock = threading.Lock()
        self.last_spoken_prediction = ""

        try:
            import pyttsx3
            self.tts_engine = pyttsx3.init()
            self.tts_engine.setProperty("rate", 190)
            self.tts_enabled = True
            print("Text-to-speech enabled for prediction audio.")
        except Exception as e:
            print(f"Text-to-speech unavailable (install pyttsx3): {e}")

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

        # Data tracking for CSV export
        self.shot_data = []
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
        self.export_to_csv()

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
        """Speak prediction in a background thread so video loop stays responsive."""
        if not self.tts_enabled or self.tts_engine is None:
            return

        label = "make" if prob >= 0.5 else "miss"
        phrase = label

        # Avoid repeating the exact same phrase frame-after-frame.
        if phrase == self.last_spoken_prediction:
            return
        self.last_spoken_prediction = phrase

        def _speak():
            with self.tts_lock:
                try:
                    self.tts_engine.say(phrase)
                    self.tts_engine.runAndWait()
                except Exception as e:
                    print(f"[TTS] Speech failed: {e}")

        threading.Thread(target=_speak, daemon=True).start()

    def display_xgb_prediction(self):
        """Draw the XGBoost prediction overlay while the display counter is active."""
        if self.xgb_display_counter <= 0 or not self.xgb_prediction_text:
            return

        prob = self.xgb_prediction_prob if self.xgb_prediction_prob is not None else 0.5

        # Colour: green = confident make, red = confident miss, yellow = uncertain
        if prob >= 0.65:
            color = (0, 220, 0)
        elif prob <= 0.35:
            color = (0, 60, 255)
        else:
            color = (0, 200, 255)

        text = self.xgb_prediction_text
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 1.4
        thickness = 3
        (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)

        h_frame = self.frame.shape[0]
        x, y = 40, h_frame - 50
        pad = 10

        # Semi-transparent dark background pill
        overlay = self.frame.copy()
        cv2.rectangle(overlay,
                      (x - pad, y - th - pad),
                      (x + tw + pad, y + baseline + pad),
                      (20, 20, 20), -1)
        alpha_factor = self.xgb_display_counter / self.xgb_display_frames
        cv2.addWeighted(overlay, 0.55 * alpha_factor,
                        self.frame, 1 - 0.55 * alpha_factor, 0, self.frame)

        # Confidence bar below the text
        bar_x = x - pad
        bar_y = y + baseline + pad + 4
        bar_w = tw + 2 * pad
        bar_h = 8
        cv2.rectangle(self.frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (60, 60, 60), -1)
        cv2.rectangle(self.frame, (bar_x, bar_y),
                      (bar_x + int(bar_w * prob), bar_y + bar_h), color, -1)

        # Text with shadow for readability
        cv2.putText(self.frame, text, (x, y), font, scale, (0, 0, 0), thickness + 2)
        cv2.putText(self.frame, text, (x, y), font, scale, color, thickness)

        self.xgb_display_counter -= 1

    # ------------------------------------------------------------------
    # Pose processing
    # ------------------------------------------------------------------

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
            cv2.circle(self.frame, (x, y), 5, (255, 200, 0), -1)

    def clean_motion(self):
        self.ball_pos = clean_ball_pos(self.ball_pos, self.frame_count)
        for i in range(len(self.ball_pos)):
            cv2.circle(self.frame, self.ball_pos[i][0], 2, (0, 0, 255), 2)

        if len(self.hoop_pos) > 1:
            self.hoop_pos = clean_hoop_pos(self.hoop_pos)
            cv2.circle(self.frame, self.hoop_pos[-1][0], 2, (128, 128, 0), 2)

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

                # Reuse cached metrics from release time
                shot_metrics = self._pending_metrics if self._pending_metrics else self.calculate_shot_metrics()
                shot_record = {
                    'shot_number': self.attempts,
                    'result': 'make' if is_make else 'miss',
                    'xgb_make_prob': round(self.xgb_prediction_prob, 4) if self.xgb_prediction_prob is not None else None,
                    'start_frame': self.shot_start_frame,
                    'up_frame': self.up_frame,
                    'down_frame': self.down_frame,
                    'eval_frame': self.frame_count,
                    'duration_frames': self.frame_count - self.shot_start_frame,
                    'duration_seconds': (self.frame_count - self.shot_start_frame) / 30.0,
                }
                shot_record.update(shot_metrics)
                self.shot_data.append(shot_record)

                # Reset state
                self.up = False
                self.down = False
                self.pending_shot = False
                self.shot_cooldown = self.cooldown_frames
                self.tracking_shot = False
                self.current_shot_frames = []
                self._pending_metrics = {}
                self.early_prediction_done = False

    # ------------------------------------------------------------------
    # Display helpers
    # ------------------------------------------------------------------

    def display_score(self):
        text = str(self.makes) + " / " + str(self.attempts)
        cv2.putText(self.frame, text, (50, 125),
                    cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 6)
        cv2.putText(self.frame, text, (50, 125),
                    cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 0), 3)

        if self.overlay_text:
            (w, h), _ = cv2.getTextSize(self.overlay_text, cv2.FONT_HERSHEY_SIMPLEX, 3, 6)
            x = self.frame.shape[1] - w - 40
            y = 100
            cv2.putText(self.frame, self.overlay_text, (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 3, self.overlay_color, 6)

        if self.fade_counter > 0:
            alpha = 0.2 * (self.fade_counter / self.fade_frames)
            self.frame = cv2.addWeighted(
                self.frame, 1 - alpha,
                np.full_like(self.frame, self.overlay_color),
                alpha, 0
            )
            self.fade_counter -= 1

    # ------------------------------------------------------------------
    # CSV export
    # ------------------------------------------------------------------

    def export_to_csv(self):
        import os

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        if self.shot_data:
            shot_csv = 'all_shots.csv'
            file_exists = os.path.isfile(shot_csv)

            for shot in self.shot_data:
                shot['session_id'] = session_id
                shot['session_timestamp'] = timestamp

            with open(shot_csv, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=self.shot_data[0].keys())
                if not file_exists:
                    writer.writeheader()
                writer.writerows(self.shot_data)

            print(f"Shot data appended to {shot_csv}")
            print(f"Session: {session_id}")
            print(f"Shots this session: {len(self.shot_data)}")
            print(f"Makes: {self.makes}")
            if self.attempts > 0:
                print(f"Shooting percentage: {self.makes / self.attempts * 100:.1f}%")

        print("\n=== Export Complete ===")


if __name__ == "__main__":
    ShotDetector()