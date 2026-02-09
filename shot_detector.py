from ultralytics import YOLO
import cv2
import cvzone
import math
import mediapipe as mp
import numpy as np
from utils import score, detect_down, detect_up, in_hoop_region, clean_hoop_pos, clean_ball_pos, get_device


class ShotDetector:
    def __init__(self):
        # Load the YOLO model created from main.py - change text to your relative path
        self.model = YOLO("best.pt") 
        
        # Uncomment this line to accelerate inference. Note that this may cause errors in some setups.
        # self.model.half()
        
        self.class_names = ['Basketball', 'Basketball Hoop']
        self.device = get_device()

        # Use video - replace text with your video path
        self.cap = cv2.VideoCapture("video_test_5.mp4")

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

        # delay evaluation so shot is not called mid-air
        self.pending_shot = False
        self.shot_eval_frame = 0
        self.shot_delay = 12  # frames to wait AFTER ball passes rim

        # Used for green and red colors after make/miss
        self.fade_frames = 20
        self.fade_counter = 0
        self.overlay_color = (0, 0, 0)

        # MediaPipe pose initialization
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        # Only keep needed landmarks from the report
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
        ]
        self.pose_points = {}



        self.run()

    def run(self):
        while True:
            ret, self.frame = self.cap.read()

            if not ret:
                # End of the video or an error occurred
                break

            results = self.model(self.frame, stream=True, device=self.device)

            for r in results:
                boxes = r.boxes
                for box in boxes:
                    # Bounding box
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    w, h = x2 - x1, y2 - y1

                    # Confidence
                    conf = float(box.conf[0])

                    # Class Name
                    cls = int(box.cls[0])
                    current_class = self.class_names[cls]

                    center = (x1 + w // 2, y1 + h // 2)

                    # Only create ball points if high confidence or near hoop
                    if (conf > .3 or (in_hoop_region(center, self.hoop_pos) and conf > 0.15)) \
                            and current_class == "Basketball":
                        self.ball_pos.append((center, self.frame_count, w, h, conf))
                        cvzone.cornerRect(self.frame, (x1, y1, w, h))

                    # Create hoop points if high confidence
                    if conf > .5 and current_class == "Basketball Hoop":
                        self.hoop_pos.append((center, self.frame_count, w, h, conf))
                        cvzone.cornerRect(self.frame, (x1, y1, w, h))
            self.process_pose()


            self.clean_motion()
            self.shot_detection()
            self.display_score()

            self.frame_count += 1
            cv2.imshow('Frame', self.frame)

            # Close if 'q' is clicked
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()
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
        # Clean and display ball motion
        self.ball_pos = clean_ball_pos(self.ball_pos, self.frame_count)
        for i in range(len(self.ball_pos)):
            cv2.circle(self.frame, self.ball_pos[i][0], 2, (0, 0, 255), 2)

        # Clean hoop motion and display current hoop center
        if len(self.hoop_pos) > 1:
            self.hoop_pos = clean_hoop_pos(self.hoop_pos)
            cv2.circle(self.frame, self.hoop_pos[-1][0], 2, (128, 128, 0), 2)

    def shot_detection(self):
        if len(self.hoop_pos) == 0 or len(self.ball_pos) == 0:
            return

        # Detecting when ball is in 'up' and 'down' area - ball can only be in 'down' area after it is in 'up'
        if not self.up:
            self.up = detect_up(self.ball_pos, self.hoop_pos)
            if self.up:
                self.up_frame = self.ball_pos[-1][1]

        if self.up and not self.down:
            self.down = detect_down(self.ball_pos, self.hoop_pos)
            if self.down:
                self.down_frame = self.ball_pos[-1][1]

        # 🔧 FIX: mark shot as pending instead of scoring immediately
        if self.up and self.down and not self.pending_shot:
            self.pending_shot = True
            self.shot_eval_frame = self.frame_count

        # 🔧 FIX: wait a few frames before evaluating make/miss
        if self.pending_shot:
            if self.frame_count - self.shot_eval_frame >= self.shot_delay:
                self.attempts += 1

                # If it is a make, put a green overlay
                if score(self.ball_pos, self.hoop_pos):
                    self.makes += 1
                    self.overlay_color = (0, 255, 0)
                    self.overlay_text = "Make"
                else:
                    self.overlay_color = (0, 0, 255)
                    self.overlay_text = "Miss"

                self.fade_counter = self.fade_frames

                # Reset shot state
                self.up = False
                self.down = False
                self.pending_shot = False

    def display_score(self):
        # Add score text
        text = str(self.makes) + " / " + str(self.attempts)
        cv2.putText(self.frame, text, (50, 125),
                    cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 6)
        cv2.putText(self.frame, text, (50, 125),
                    cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 0), 3)

        # Add overlay text for shot result if it exists
        if hasattr(self, 'overlay_text'):
            (w, h), _ = cv2.getTextSize(
                self.overlay_text, cv2.FONT_HERSHEY_SIMPLEX, 3, 6
            )
            x = self.frame.shape[1] - w - 40
            y = 100
            cv2.putText(self.frame, self.overlay_text, (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 3,
                        self.overlay_color, 6)

        # Gradually fade out color after shot
        if self.fade_counter > 0:
            alpha = 0.2 * (self.fade_counter / self.fade_frames)
            self.frame = cv2.addWeighted(
                self.frame, 1 - alpha,
                np.full_like(self.frame, self.overlay_color),
                alpha, 0
            )
            self.fade_counter -= 1


if __name__ == "__main__":
    ShotDetector()
