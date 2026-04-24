# import libraries
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

# main class
class ShotDetector:
    def __init__(self):

        # Yolo Model
        self.model = YOLO("best.pt")
        #  YOLO Classes 
        self.class_names = ['Basketball', 'Basketball Hoop']
        # Get GPU if available
        self.device = get_device()

        # Uncomment for webcam input comment out for video file input
        # self.cap = cv2.VideoCapture(0) 
        self.cap = cv2.VideoCapture("test.mp4")
        # record ball and hoop position in a list
        self.ball_pos = []
        self.hoop_pos = []
        # frame counter
        self.frame_count = 0
        # set frame to None for now
        self.frame = None
        
        # makes and attempts set to 0
        self.makes = 0
        self.attempts = 0

        # shot state tracking 
        # up flag set to False
        self.up = False
        # down set to False
        self.down = False
        # number up frames set to 0
        self.up_frame = 0
        # number down frames set to 0
        self.down_frame = 0

        # Delay evaluation by 15 frames to allow the ball to reach the hoop
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
        self.xgb_prediction_prob = None   
        self.xgb_display_counter = 0
        self.xgb_display_frames = 60      
        self._pending_metrics = {}

        # enable text-to-speech for predictions
        self.tts_enabled = True
        self.last_spoken_prediction = ""

        # Load XGBoost model and feature names
        try:
            self.xgb_model = XGBClassifier()
            # load the JSON model file
            self.xgb_model.load_model("xgboost_shot_model.json")
            # Load the pickle file containing the feature names
            self.xgb_feature_names = joblib.load("xgboost_feature_names.pkl")
            # if the pickle is empty, use feature names stored in the trained booster.
            if not self.xgb_feature_names:
                booster_features = self.xgb_model.get_booster().feature_names
                self.xgb_feature_names = booster_features if booster_features else []
        except Exception as e:
            print(f"Warning: could not load XGBoost model: {e}")
            self.xgb_model = None
            self.xgb_feature_names = []
        

        # MediaPipe pose
        self.mp_pose = mp.solutions.pose
        # tracking confidence set to 0.5 for both detection and tracking
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )# only track the landmarks needed for basketball
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
        # dictionary to store the pose points for the current frame
        self.pose_points = {}

        # List to store the frames of the current shot 
        self.current_shot_frames = []

        # seeing if a shot is being tracked or not
        self.tracking_shot = False
        self.shot_start_frame = 0
        # run the main loop
        self.run()

   
    # The main loop to process each frame, track ball and hoop, detect shots, calculate metrics, run XGBoost prediction, and display results
    def run(self):
        while True:
            # read a frame from the video capture
            ret, self.frame = self.cap.read()
            if not ret:
                break
            # run YOLO on the GPU 
            results = self.model(self.frame, stream=True, device=self.device)
            # Intialise frame ball and hoop position to None
            frame_ball_pos = None
            frame_hoop_pos = None
            # loop over the frames and draw bounding boxes and store ball and hoop positions
            for r in results:
                # get the bounding boxes for the current frame
                boxes = r.boxes
                for box in boxes: # loop over detected boxes
                    # get the coordinates of the bounding box
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    # calculate the width and height of the bounding box
                    w, h = x2 - x1, y2 - y1 
                    # get the confidence and class of the detected object
                    conf = float(box.conf[0])
                    # get the name of the class
                    cls = int(box.cls[0])
                    # current class name
                    current_class = self.class_names[cls]
                    # calculate the center of the bounding box
                    center = (x1 + w // 2, y1 + h // 2)
                    # low confidence threshold for easier detection 
                    if (conf > 0.01 or (in_hoop_region(center, self.hoop_pos) and conf > 0.01)) and current_class == "Basketball":
                        # append the center of the bounding box to the ball_pos list
                        self.ball_pos.append((center, self.frame_count, w, h, conf))
                        # set the frame_ball_pos to the center of the bounding box
                        frame_ball_pos = center
                        # draw a rectangle around the detected ball
                        cvzone.cornerRect(self.frame, (x1, y1, w, h))
                    # low confidence threshold for easier detection 
                    if conf > .1 and current_class == "Basketball Hoop":
                        # append the center of the bounding box to the hoop_pos list
                        self.hoop_pos.append((center, self.frame_count, w, h, conf))
                        # set the frame_hoop_pos to the center of the bounding box
                        frame_hoop_pos = center
                        # draw a rectangle around the detected hoop
                        cvzone.cornerRect(self.frame, (x1, y1, w, h))
            # call the pose processing and motion cleaning functions
            self.process_pose()
            self.clean_motion()
            # if tracking the shot 
            if self.tracking_shot:
                # save the frame data 
                frame_data = self._build_frame_data(frame_ball_pos, frame_hoop_pos)
                # append the frame data to the current shot frames list
                self.current_shot_frames.append(frame_data)
            # run shot detection, display score, and display XGBoost prediction 
            self.shot_detection()
            self.display_score()
            self.display_xgb_prediction()
            # decrease the shot cooldown if it is active
            if self.shot_cooldown > 0:
                self.shot_cooldown -= 1
            # increase the frame count
            self.frame_count += 1
            # show the frame
            cv2.imshow('Frame', self.frame)
            # press 'q' to quit the video
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        # release the video capture and close all windows
        self.cap.release()
        cv2.destroyAllWindows()

   
    # track all the frames data
    def _build_frame_data(self, frame_ball_pos, frame_hoop_pos):
        # build a dictionary that stores all the frames data 
        data = {
            # frame count
            'frame': self.frame_count,
            # ball positions 
            'ball_x': frame_ball_pos[0] if frame_ball_pos else None,
            'ball_y': frame_ball_pos[1] if frame_ball_pos else None,
            # hoop positions
            'hoop_x': frame_hoop_pos[0] if frame_hoop_pos else None,
            'hoop_y': frame_hoop_pos[1] if frame_hoop_pos else None,
            # up region tracking
            'in_up_region': self.up,
        }
        # loop over the pose points and add them to the data dictionary
        for name, (x, y, vis) in self.pose_points.items():
            data[f'{name}_x'] = x
            data[f'{name}_y'] = y
            data[f'{name}_vis'] = vis
            
        # return the data dictionary
        return data

    # XGBoost prediction 
    def predict_shot(self, metrics: dict):
        # skip prediction if the XGBoost model failed to load
        if self.xgb_model is None:
            return

        # if metrics is None, use an empty dictionary
        metrics = metrics or {}

        # if feature names are empty, try to pull them from the trained booster
        if not self.xgb_feature_names:
            booster_features = self.xgb_model.get_booster().feature_names
            if booster_features:
                self.xgb_feature_names = booster_features
            else:
                print("[XGBoost] Prediction skipped: feature names are unavailable.")
                return

        # get the features for the shot, use 0 if the metric is not available
        row = {feat: metrics.get(feat, 0) for feat in self.xgb_feature_names}
        # Wrap the row in single Data frame
        X = pd.DataFrame([row], columns=self.xgb_feature_names)
        # get the probability of the shot being a make 
        try:
            prob = float(self.xgb_model.predict_proba(X)[0][1])
        except Exception as e:
            print(f"[XGBoost] Prediction failed: {e}")
            return
        # store probability 
        self.xgb_prediction_prob = prob
        # > = 0.5 is a make, < 0.5 is a miss
        self.xgb_prediction_text = f"Pred: {'Make' if prob >= 0.5 else 'Miss'} ({prob * 100:.0f}%)"
        # Display the prediction
        self.xgb_display_counter = self.xgb_display_frames
        # print the prediction
        print(f"[XGBoost] {self.xgb_prediction_text}")
        # speak the prediction using text-to-speech
        self.speak_prediction(prob)

    # set up text to speech 
    def speak_prediction(self, prob):
        # if text to speech is disabled dont speak
        if not self.tts_enabled:
            return

      
        # assign label based on probability threshold
        label = "make" if prob >= 0.5 else "miss"
        # assign phrase to speak
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

    # Display the XGBoost prediciton
    def display_xgb_prediction(self):
        # only draw the badge while counter is active and text exists
        if self.xgb_display_counter <= 0 or not self.xgb_prediction_text:
            return

        
        # assign the probability to 0.5 if it is None to avoid errors
        prob = self.xgb_prediction_prob if self.xgb_prediction_prob is not None else 0.5
        alpha_factor = self.xgb_display_counter / self.xgb_display_frames

        # change the colour of the display based on probability thresholds 
        if prob >= 0.65:
            color = (72, 220, 80)       # green
            bg_accent = (30, 80, 35)
        elif prob <= 0.35:
            color = (80, 80, 255)       # red
            bg_accent = (40, 30, 80)
        else:
            color = (60, 200, 255)      # amber
            bg_accent = (30, 70, 80)
        # assign label based on probability threshold
        label = "MAKE" if prob >= 0.5 else "MISS"
        # format the probability as a percentage with no decimal places
        pct_text = f"{prob * 100:.0f}%"
        # Set font 
        font = cv2.FONT_HERSHEY_SIMPLEX
        # get the size of the text
        h_frame, w_frame = self.frame.shape[:2]

        # set the size and position of the badge
        badge_w, badge_h = 280, 70
        bx = (w_frame - badge_w) // 2
        by = h_frame - badge_h - 25
        # draw the badge
        overlay = self.frame.copy()
        cv2.rectangle(overlay, (bx, by), (bx + badge_w, by + badge_h), bg_accent, -1)
        cv2.addWeighted(overlay, 0.7 * alpha_factor,
                        self.frame, 1 - 0.7 * alpha_factor, 0, self.frame)
        cv2.rectangle(self.frame, (bx, by), (bx + badge_w, by + badge_h), color, 2)

        # Label text 
        cv2.putText(self.frame, label, (bx + 18, by + 45),
                    font, 1.3, color, 3)

        # set the size of the percentage text and position it on the badge
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
        # decrease the display counter
        self.xgb_display_counter -= 1

  

    # Connect the pose points with lines and draw circles on the joints
    _SKELETON_CONNECTIONS = [
        ('LEFT_SHOULDER', 'LEFT_ELBOW'), ('LEFT_ELBOW', 'LEFT_WRIST'),
        ('RIGHT_SHOULDER', 'RIGHT_ELBOW'), ('RIGHT_ELBOW', 'RIGHT_WRIST'),
        ('LEFT_SHOULDER', 'RIGHT_SHOULDER'),
        ('LEFT_SHOULDER', 'LEFT_HIP'), ('RIGHT_SHOULDER', 'RIGHT_HIP'),
        ('LEFT_HIP', 'RIGHT_HIP'),
        ('LEFT_HIP', 'LEFT_KNEE'), ('LEFT_KNEE', 'LEFT_ANKLE'),
        ('RIGHT_HIP', 'RIGHT_KNEE'), ('RIGHT_KNEE', 'RIGHT_ANKLE'),
    ]
    # process the pose and store points in a dictionary
    def process_pose(self):
        # Convert BGR to RGB
        rgb = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
        # results from MediaPipe pose processing
        results = self.pose.process(rgb)
        # store the pose points in a dictionary
        self.pose_points = {}

        # if no landmarks are found skip pose extraction for this frame
        if not results.pose_landmarks:
            return

        # Get the current frame height and width
        h, w = self.frame.shape[:2]
        # Loop over the needed pose landmarks
        for lm in self.pose_landmarks_needed:
            # store the x y coordinates and visibility of the landmark
            landmark = results.pose_landmarks.landmark[lm]
            # x is the x coordinate times the frame width
            x = int(landmark.x * w)
            # y is the y coordinate times the frame height
            y = int(landmark.y * h)
            # store the pose points 
            self.pose_points[lm.name] = (x, y, landmark.visibility)

        # Draw skeleton lines 
        for a, b in self._SKELETON_CONNECTIONS:
            # if both pose points are in dictionary and have visibility > 0.4 draw a line between them
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
    # clean the motion by removing outliers and drawing trails and hoop position
    def clean_motion(self):
        # Clean ball position
        self.ball_pos = clean_ball_pos(self.ball_pos, self.frame_count)

        # Draw ball trail with fading opacity 
        num_trail = len(self.ball_pos)
        # loop over ball positions 
        for i in range(num_trail):
            # fade the older positions 
            fade = max(0.25, (i + 1) / num_trail)
            # get the radius and colour based on the fade value
            radius = max(2, int(4 * fade))
            # fade the colour from bright blue to a dimmer blue as the trail gets older
            blue = int(80 + 175 * fade)
            green = int(50 * fade)
            # draw the circle 
            cv2.circle(self.frame, self.ball_pos[i][0], radius,
                       (0, green, blue), -1, cv2.LINE_AA)
        # if length of hoop positions is greater than 1
        if len(self.hoop_pos) > 1:
            # clean the hoop positions
            self.hoop_pos = clean_hoop_pos(self.hoop_pos)
            # get hoop center
            hx, hy = self.hoop_pos[-1][0]
            # draw a circle at the hoop center
            cv2.drawMarker(self.frame, (hx, hy), (0, 255, 200), cv2.MARKER_CROSS,
                           12, 2, cv2.LINE_AA)


    # calcute the shot metrics 
    def calculate_shot_metrics(self):
        # if no shot frames exist return empty metrics
        if not self.current_shot_frames:
            return {}
      
        # store metrics in a dictionary
        metrics = {}

        # set the release frame and index 
        release_frame = None
        release_idx = -1 
        # loop over the current shot frame 
        for i, frame in enumerate(self.current_shot_frames):
            #if ball in up region 
            if frame['in_up_region']:  
                # release frame is the frame where ball is in up region for the first time
                # save release frame 
                release_frame = frame  
                # save index of release frame 
                release_idx = i  
                break  

        # Find the 5 frames before release frame 
        pre_release_frame = None
        if release_idx > 5:
            pre_release_frame = self.current_shot_frames[release_idx - 5]
        # if in the release frame get the metrics for the shot
        if release_frame:
            # Right elbow + shoulder angles
            if all(k in release_frame and release_frame[k] is not None for k in
                   ['RIGHT_WRIST_x', 'RIGHT_ELBOW_x', 'RIGHT_SHOULDER_x']):
                # get the coordinates of the wrist, elbow, and shoulder
                wrist    = np.array([release_frame['RIGHT_WRIST_x'],    release_frame['RIGHT_WRIST_y']])
                elbow    = np.array([release_frame['RIGHT_ELBOW_x'],    release_frame['RIGHT_ELBOW_y']])
                shoulder = np.array([release_frame['RIGHT_SHOULDER_x'], release_frame['RIGHT_SHOULDER_y']])
                # call calculate angle function from utils.py to calculate the angles
                metrics['right_elbow_angle']    = calculate_angle(wrist, elbow, shoulder)
                metrics['right_shoulder_angle'] = calculate_angle(shoulder, elbow, wrist)

            # Right wrist extension 
            if all(k in release_frame and release_frame[k] is not None for k in
                   ['RIGHT_WRIST_y', 'RIGHT_ELBOW_y']):
                # get the difference between the y of the wrist and y of the elbow
                metrics['right_wrist_extension'] = release_frame['RIGHT_WRIST_y'] - release_frame['RIGHT_ELBOW_y']

            # Left arm
            if all(k in release_frame and release_frame[k] is not None for k in
                   ['LEFT_WRIST_x', 'LEFT_ELBOW_x', 'LEFT_SHOULDER_x']):
                # get coordinates of the wrist, elbow, and shoulder
                l_wrist    = np.array([release_frame['LEFT_WRIST_x'],    release_frame['LEFT_WRIST_y']])
                l_elbow    = np.array([release_frame['LEFT_ELBOW_x'],    release_frame['LEFT_ELBOW_y']])
                l_shoulder = np.array([release_frame['LEFT_SHOULDER_x'], release_frame['LEFT_SHOULDER_y']])
                # calculate angle 
                metrics['left_elbow_angle'] = calculate_angle(l_wrist, l_elbow, l_shoulder)

            # Right knee
            if all(k in release_frame and release_frame[k] is not None for k in
                   ['RIGHT_KNEE_x', 'RIGHT_ANKLE_x', 'RIGHT_HIP_x']):
                # get coordinates of the hip, knee, and ankle
                r_hip   = np.array([release_frame['RIGHT_HIP_x'],   release_frame['RIGHT_HIP_y']])
                r_knee  = np.array([release_frame['RIGHT_KNEE_x'],  release_frame['RIGHT_KNEE_y']])
                r_ankle = np.array([release_frame['RIGHT_ANKLE_x'], release_frame['RIGHT_ANKLE_y']])
                # calculate angle
                metrics['right_knee_angle'] = calculate_angle(r_hip, r_knee, r_ankle)

            # Left knee
            if all(k in release_frame and release_frame[k] is not None for k in
                   ['LEFT_KNEE_x', 'LEFT_ANKLE_x', 'LEFT_HIP_x']):
                # get coordinates of the hip, knee, and ankle
                l_hip   = np.array([release_frame['LEFT_HIP_x'],   release_frame['LEFT_HIP_y']])
                l_knee  = np.array([release_frame['LEFT_KNEE_x'],  release_frame['LEFT_KNEE_y']])
                l_ankle = np.array([release_frame['LEFT_ANKLE_x'], release_frame['LEFT_ANKLE_y']])
                # calculate angle
                metrics['left_knee_angle'] = calculate_angle(l_hip, l_knee, l_ankle)

            # Right hip
            if all(k in release_frame and release_frame[k] is not None for k in
                   ['RIGHT_SHOULDER_x', 'RIGHT_HIP_x', 'RIGHT_KNEE_x']):
                # get coordinates of the shoulder, hip, and knee
                r_hip      = np.array([release_frame['RIGHT_HIP_x'],      release_frame['RIGHT_HIP_y']])
                r_shoulder = np.array([release_frame['RIGHT_SHOULDER_x'], release_frame['RIGHT_SHOULDER_y']])
                r_knee     = np.array([release_frame['RIGHT_KNEE_x'],     release_frame['RIGHT_KNEE_y']])
                # calculate angle
                metrics['right_hip_angle'] = calculate_angle(r_shoulder, r_hip, r_knee)

            # Alignment metrics
            if all(k in release_frame and release_frame[k] is not None for k in
                   ['LEFT_SHOULDER_y', 'RIGHT_SHOULDER_y']):
                # shoulder tilt is the difference between the left and right shoulder y coordinates
                metrics['shoulder_tilt'] = abs(release_frame['LEFT_SHOULDER_y'] - release_frame['RIGHT_SHOULDER_y'])

            if all(k in release_frame and release_frame[k] is not None for k in
                   ['LEFT_HIP_y', 'RIGHT_HIP_y']):
                # hip tilt is the difference between the left and right hip y coordinates
                metrics['hip_tilt'] = abs(release_frame['LEFT_HIP_y'] - release_frame['RIGHT_HIP_y'])

            if all(k in release_frame and release_frame[k] is not None for k in
                   ['LEFT_KNEE_x', 'RIGHT_KNEE_x']):
                # knee spread is the difference between the left and right knee x coordinates
                metrics['knee_spread'] = abs(release_frame['LEFT_KNEE_x'] - release_frame['RIGHT_KNEE_x'])

            # Release height and position
            if release_frame.get('RIGHT_WRIST_y') is not None:
                # release height is the y coordinate of the right wrist
                metrics['release_height'] = release_frame['RIGHT_WRIST_y']
            if release_frame.get('RIGHT_WRIST_x') is not None:
                # release position is the x coordinate of the right wrist
                metrics['release_x_position'] = release_frame['RIGHT_WRIST_x']

            # Body vertical alignment
            if all(k in release_frame and release_frame[k] is not None for k in
                   ['RIGHT_SHOULDER_x', 'RIGHT_ANKLE_x']):
                # how straight the body is by looking at the x coordinates of the right shoulder and right ankle
                metrics['body_vertical_alignment'] = abs(
                    release_frame['RIGHT_SHOULDER_x'] - release_frame['RIGHT_ANKLE_x'])

            # Ball-to-hoop metrics
            if release_frame.get('ball_x') is not None and release_frame.get('hoop_x') is not None:
                # distance from hoop in x and y and total distance
                metrics['ball_hoop_distance_x'] = abs(release_frame['ball_x'] - release_frame['hoop_x'])
                metrics['ball_hoop_distance_y'] = abs(release_frame['ball_y'] - release_frame['hoop_y'])
                ball_arr = np.array([release_frame['ball_x'], release_frame['ball_y']])
                hoop_arr = np.array([release_frame['hoop_x'], release_frame['hoop_y']])
                metrics['ball_hoop_total_distance'] = calculate_distance(ball_arr, hoop_arr)
                # get the angle from the ball to hoop
                if release_frame['ball_y'] != release_frame['hoop_y']:
                    metrics['release_angle_to_hoop'] = np.degrees(np.arctan2(
                        release_frame['hoop_y'] - release_frame['ball_y'],
                        release_frame['hoop_x'] - release_frame['ball_x']
                    ))

            # Pre-release comparisons using the 5 frame before the release  
            if pre_release_frame:
                if all(k in pre_release_frame and pre_release_frame[k] is not None for k in
                       # using the right leg metrics calculate knee extension 
                       ['RIGHT_KNEE_x', 'RIGHT_ANKLE_x', 'RIGHT_HIP_x']):
                    pre_hip   = np.array([pre_release_frame['RIGHT_HIP_x'],   pre_release_frame['RIGHT_HIP_y']])
                    pre_knee  = np.array([pre_release_frame['RIGHT_KNEE_x'],  pre_release_frame['RIGHT_KNEE_y']])
                    pre_ankle = np.array([pre_release_frame['RIGHT_ANKLE_x'], pre_release_frame['RIGHT_ANKLE_y']])
                    pre_knee_angle = calculate_angle(pre_hip, pre_knee, pre_ankle)
                    if 'right_knee_angle' in metrics:
                        metrics['knee_extension'] = metrics['right_knee_angle'] - pre_knee_angle
                # using right hip calculate the jump height 
                if pre_release_frame.get('RIGHT_HIP_y') is not None and release_frame.get('RIGHT_HIP_y') is not None:
                    metrics['jump_height'] = pre_release_frame['RIGHT_HIP_y'] - release_frame['RIGHT_HIP_y']
                # using right elbow calculate elbow extension 
                if all(k in pre_release_frame and pre_release_frame[k] is not None for k in
                       ['RIGHT_WRIST_x', 'RIGHT_ELBOW_x', 'RIGHT_SHOULDER_x']):
                    pre_wrist    = np.array([pre_release_frame['RIGHT_WRIST_x'],    pre_release_frame['RIGHT_WRIST_y']])
                    pre_elbow    = np.array([pre_release_frame['RIGHT_ELBOW_x'],    pre_release_frame['RIGHT_ELBOW_y']])
                    pre_shoulder = np.array([pre_release_frame['RIGHT_SHOULDER_x'], pre_release_frame['RIGHT_SHOULDER_y']])
                    pre_elbow_angle = calculate_angle(pre_wrist, pre_elbow, pre_shoulder)
                    if 'right_elbow_angle' in metrics:
                        metrics['elbow_extension'] = metrics['right_elbow_angle'] - pre_elbow_angle
        # return the metrics
        return metrics

    
    # detect the shot by tracking its  ball position in relation to the hoop
    def shot_detection(self):
        # if we dont have hoop or ball detections yet, skip this frame
        if len(self.hoop_pos) == 0 or len(self.ball_pos) == 0:
            return

        # if cooldown is active, dont allow another shot count yet
        if self.shot_cooldown > 0:
            return

        # Detect ball in 'up' state  
        if not self.up:
            self.up = detect_up(self.ball_pos, self.hoop_pos, min_consecutive=1)
            if self.up:
                # if in up state save the frame number and start tracking the shot
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

   # Create an interface displaying the metrics 
    def display_score(self):
        # draw bar
        h_frame, w_frame = self.frame.shape[:2]
        bar_h = 60

        # draw semi-transparent background for the score bar
        overlay = self.frame.copy()
        cv2.rectangle(overlay, (0, 0), (w_frame, bar_h), (15, 15, 15), -1)
        cv2.addWeighted(overlay, 0.65, self.frame, 0.35, 0, self.frame)

        # Thin accent line at the bottom of the bar
        cv2.line(self.frame, (0, bar_h), (w_frame, bar_h), (0, 180, 255), 2)

        font = cv2.FONT_HERSHEY_SIMPLEX

        # Draw score
        score_text = f"{self.makes} / {self.attempts}"
        cv2.putText(self.frame, score_text, (20, 42),
                    font, 1.2, (255, 255, 255), 3, cv2.LINE_AA)

        # draw shoot percentage  
        if self.attempts > 0:
            pct = self.makes / self.attempts * 100
            pct_text = f"{pct:.0f}%"
        else:
            pct_text = "--"
        (pw, _), _ = cv2.getTextSize(pct_text, font, 1.2, 3)
        cv2.putText(self.frame, pct_text, (w_frame - pw - 20, 42),
                    font, 1.2, (0, 220, 255), 3, cv2.LINE_AA)

        # draw shot predictor title
        label = "SHOT PREDICTOR"
        (lw, _), _ = cv2.getTextSize(label, font, 0.6, 2)
        cv2.putText(self.frame, label, ((w_frame - lw) // 2, 38),
                    font, 0.6, (140, 140, 140), 2, cv2.LINE_AA)

        # Make or miss overlay 
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