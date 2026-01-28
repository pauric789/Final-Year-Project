from ultralytics import YOLO
import cv2
import cvzone
import math
import numpy as np
import mediapipe as mp
from utils import score, detect_down, detect_up, in_hoop_region, clean_hoop_pos, clean_ball_pos, get_device


class ShotDetector:
    def __init__(self):
        self.overlay_text = "Waiting..."
        self.model = YOLO("best.pt")

        self.class_names = ['Basketball', 'Basketball Hoop']
        self.device = get_device()

        self.cap = cv2.VideoCapture("myowntest.mp4")

        self.ball_pos = []
        self.hoop_pos = []

        self.frame_count = 0
        self.frame = None

        self.makes = 0
        self.attempts = 0

        self.up = False
        self.down = False
        self.up_frame = 0
        self.down_frame = 0

        # Overlay visuals
        self.fade_frames = 20
        self.fade_counter = 0
        self.overlay_color = (0, 0, 0)

        # MediaPipe Pose
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        self.run()

    def run(self):
        while True:
            ret, self.frame = self.cap.read()
            if not ret:
                break

            # ---------- MediaPipe ----------
            rgb = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
            pose_res = self.pose.process(rgb)
            if pose_res.pose_landmarks:
                self.mp_drawing.draw_landmarks(
                    self.frame,
                    pose_res.pose_landmarks,
                    self.mp_pose.POSE_CONNECTIONS
                )

            # ---------- YOLO ----------
            results = self.model(self.frame, stream=True, device=self.device)

            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    w, h = x2 - x1, y2 - y1
                    conf = math.ceil((box.conf[0] * 100)) / 100
                    cls = int(box.cls[0])
                    current_class = self.class_names[cls]
                    center = (x1 + w // 2, y1 + h // 2)

                    if (
                        (conf > 0.3 or
                         (in_hoop_region(center, self.hoop_pos) and conf > 0.15))
                        and current_class == "Basketball"
                    ):
                        self.ball_pos.append((center, self.frame_count, w, h, conf))
                        cvzone.cornerRect(self.frame, (x1, y1, w, h))

                    if conf > 0.5 and current_class == "Basketball Hoop":
                        self.hoop_pos.append((center, self.frame_count, w, h, conf))
                        cvzone.cornerRect(self.frame, (x1, y1, w, h))

            self.clean_motion()
            self.shot_detection()
            self.display_score()

            self.frame_count += 1

            cv2.imshow("Frame", self.frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()

    def clean_motion(self):
        self.ball_pos = clean_ball_pos(self.ball_pos, self.frame_count)
        for p in self.ball_pos:
            cv2.circle(self.frame, p[0], 2, (0, 0, 255), 2)

        if len(self.hoop_pos) > 1:
            self.hoop_pos = clean_hoop_pos(self.hoop_pos)
            cv2.circle(self.frame, self.hoop_pos[-1][0], 2, (128, 128, 0), 2)

    def shot_detection(self):
        if len(self.hoop_pos) > 0 and len(self.ball_pos) > 0:

            if not self.up:
                self.up = detect_up(self.ball_pos, self.hoop_pos)
                if self.up:
                    self.up_frame = self.ball_pos[-1][1]

            if self.up and not self.down:
                self.down = detect_down(self.ball_pos, self.hoop_pos)
                if self.down:
                    self.down_frame = self.ball_pos[-1][1]

            if self.frame_count % 10 == 0:
                if self.up and self.down and self.up_frame < self.down_frame:
                    self.attempts += 1
                    self.up = False
                    self.down = False

                    if score(self.ball_pos, self.hoop_pos):
                        self.makes += 1
                        self.overlay_text = "Make"
                        self.overlay_color = (0, 255, 0)
                    else:
                        self.overlay_text = "Miss"
                        self.overlay_color = (0, 0, 255)

                    self.fade_counter = self.fade_frames

    def display_score(self):
        # Score counter
        text = f"{self.makes} / {self.attempts}"
        cv2.putText(self.frame, text, (50, 125),
                    cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 0), 10)
        cv2.putText(self.frame, text, (50, 125),
                    cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 4)

        # Make / Miss overlay
        if self.fade_counter > 0:
            (tw, th), _ = cv2.getTextSize(
                self.overlay_text, cv2.FONT_HERSHEY_SIMPLEX, 3, 6
            )
            tx = self.frame.shape[1] - tw - 40
            cv2.putText(
                self.frame,
                self.overlay_text,
                (tx, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                3,
                self.overlay_color,
                6
            )

            alpha = 0.2 * (self.fade_counter / self.fade_frames)
            overlay = np.full_like(self.frame, self.overlay_color)
            cv2.addWeighted(self.frame, 1 - alpha, overlay, alpha, 0, self.frame)

            self.fade_counter -= 1


if __name__ == "__main__":
    ShotDetector()
