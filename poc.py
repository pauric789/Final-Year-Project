from ultralytics import YOLO
import cv2
import cvzone
import math
import numpy as np
import mediapipe as mp
from utils import score, detect_down, detect_up, in_hoop_region, clean_hoop_pos, clean_ball_pos, get_device


class ShotDetector:
    def __init__(self):
        # Load the YOLO model created from main.py - change text to your relative path
        self.overlay_text = None
        self.model = YOLO("best.pt") 
        
        # Uncomment this line to accelerate inference. Note that this may cause errors in some setups.
        #self.model.half()
        
        self.class_names = ['Basketball', 'Basketball Hoop']
        self.device = get_device()
        # Uncomment line below to use webcam (I streamed to my iPhone using Iriun Webcam)
       # self.cap = cv2.VideoCapture(0)

        # Use video - replace text with your video path
        self.cap = cv2.VideoCapture("test.mp4")

        self.ball_pos = []  # array of tuples ((x_pos, y_pos), frame count, width, height, conf)
        self.hoop_pos = []  # array of tuples ((x_pos, y_pos), frame count, width, height, conf)

        self.frame_count = 0
        self.frame = None

        self.makes = 0
        self.attempts = 0

        # Used to detect shots (upper and lower region)
        self.up = False
        self.down = False
        self.up_frame = 0
        self.down_frame = 0

        # Used for green and red colors after make/miss
        self.fade_frames = 20
        self.fade_counter = 0
        self.overlay_color = (0, 0, 0)

        # MediaPipe pose initialization
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

        # Wrist tracking for simple "release" detection
        self.wrist_history = []  # list of (x_pix, y_pix, frame_count)
        self.release_counter = 0
        self.release_frames = 12
        self.release_threshold_px = 4  # upward motion threshold (pixels per few frames)

        self.run()

    def run(self):
        while True:
            ret, self.frame = self.cap.read()

            if not ret:
                # End of the video or an error occurred
                break

            # -- MediaPipe Pose Estimation (non-invasive, draws landmarks on frame) --
            rgb = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
            pose_res = self.pose.process(rgb)
            if pose_res.pose_landmarks:
                # draw pose landmarks
                self.mp_drawing.draw_landmarks(self.frame, pose_res.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)

                # extract wrist keypoints and choose the most visible one
                h, w, _ = self.frame.shape
                lm = pose_res.pose_landmarks.landmark
                r_w = lm[self.mp_pose.PoseLandmark.RIGHT_WRIST]
                l_w = lm[self.mp_pose.PoseLandmark.LEFT_WRIST]

                def _to_px(point):
                    return (int(point.x * w), int(point.y * h), float(getattr(point, "visibility", 0.0)))

                rx, ry, rv = _to_px(r_w)
                lx, ly, lv = _to_px(l_w)

                chosen_wrist = None
                if rv > lv and rv > 0.2:
                    chosen_wrist = (rx, ry)
                elif lv > 0.2:
                    chosen_wrist = (lx, ly)

                if chosen_wrist:
                    # draw wrist marker
                    cv2.circle(self.frame, chosen_wrist, 6, (0, 255, 255), -1)
                    self.wrist_history.append((chosen_wrist[0], chosen_wrist[1], self.frame_count))
                    if len(self.wrist_history) > 12:
                        self.wrist_history.pop(0)

                    # simple upward-motion release detection using recent history
                    if len(self.wrist_history) >= 4:
                        # average previous y over three frames vs current
                        prev_avg = sum(p[1] for p in self.wrist_history[-4:-1]) / 3.0
                        curr_y = self.wrist_history[-1][1]
                        vel = prev_avg - curr_y  # positive if wrist moved up
                        if vel > self.release_threshold_px:
                            # if hoop known, require proximity horizontally
                            if len(self.hoop_pos) > 0:
                                hx = self.hoop_pos[-1][0][0]
                                hoop_w = self.hoop_pos[-1][2]
                                if abs(chosen_wrist[0] - hx) < 2 * hoop_w:
                                    self.release_counter = self.release_frames
                            else:
                                self.release_counter = self.release_frames
            # -- end MediaPipe --

            results = self.model(self.frame, stream=True, device=self.device)

            for r in results:
                boxes = r.boxes
                for box in boxes:
                    # Bounding box
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    w, h = x2 - x1, y2 - y1

                    # Confidence
                    conf = math.ceil((box.conf[0] * 100)) / 100

                    # Class Name
                    cls = int(box.cls[0])
                    current_class = self.class_names[cls]

                    center = (int(x1 + w / 2), int(y1 + h / 2))

                    # Only create ball points if high confidence or near hoop
                    if (conf > .3 or (in_hoop_region(center, self.hoop_pos) and conf > 0.15)) and current_class == "Basketball":
                        self.ball_pos.append((center, self.frame_count, w, h, conf))
                        cvzone.cornerRect(self.frame, (x1, y1, w, h))

                    # Create hoop points if high confidence
                    if conf > .5 and current_class == "Basketball Hoop":
                        self.hoop_pos.append((center, self.frame_count, w, h, conf))
                        cvzone.cornerRect(self.frame, (x1, y1, w, h))

            self.clean_motion()
            self.shot_detection()
            #self.display_score()
            self.frame_count += 1

            cv2.imshow('Frame', self.frame)

            # Close if 'q' is clicked
            if cv2.waitKey(1) & 0xFF == ord('q'):  # higher waitKey slows video down, use 1 for webcam
                break

        self.cap.release()
        cv2.destroyAllWindows()

    def clean_motion(self):
        # Clean and display ball motion
        self.ball_pos = clean_ball_pos(self.ball_pos, self.frame_count)
        for i in range(0, len(self.ball_pos)):
            cv2.circle(self.frame, self.ball_pos[i][0], 2, (0, 0, 255), 2)

        # Clean hoop motion and display current hoop center
        if len(self.hoop_pos) > 1:
            self.hoop_pos = clean_hoop_pos(self.hoop_pos)
            cv2.circle(self.frame, self.hoop_pos[-1][0], 2, (128, 128, 0), 2)

    def shot_detection(self):
        if len(self.hoop_pos) > 0 and len(self.ball_pos) > 0:
            # Detecting when ball is in 'up' and 'down' area - ball can only be in 'down' area after it is in 'up'
            if not self.up:
                self.up = detect_up(self.ball_pos, self.hoop_pos)
                if self.up:
                    self.up_frame = self.ball_pos[-1][1]

            if self.up and not self.down:
                self.down = detect_down(self.ball_pos, self.hoop_pos)
                if self.down:
                    self.down_frame = self.ball_pos[-1][1]

            # If ball goes from 'up' area to 'down' area in that order, increase attempt and reset
            if self.frame_count % 10 == 0:
                if self.up and self.down and self.up_frame < self.down_frame:
                    self.attempts += 1
                    self.up = False
                    self.down = False

                    # If it is a make, put a green overlay and display "完美"
                    if score(self.ball_pos, self.hoop_pos):
                        self.makes += 1
                        self.overlay_color = (0, 255, 0)  # Green for make
                        #self.overlay_text = "Make"
                        self.fade_counter = self.fade_frames

                    else:
                        self.overlay_color = (255, 0, 0)  # Red for miss
                        #self.overlay_text = "Miss"
                        self.fade_counter = self.fade_frames

    def display_score(self):
        # Display release indicator if detected recently (minimal UI)
        if self.release_counter > 0:
            cv2.putText(self.frame, "Release", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)
            self.release_counter -= 1

        # Gradually fade out color after shot (keeps visual feedback without text)
        if self.fade_counter > 0:
            alpha = 0.2 * (self.fade_counter / self.fade_frames)
            self.frame = cv2.addWeighted(self.frame, 1 - alpha, np.full_like(self.frame, self.overlay_color), alpha, 0)
            self.fade_counter -= 1


if __name__ == "__main__":
    ShotDetector()

