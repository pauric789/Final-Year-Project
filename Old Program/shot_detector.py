from ultralytics import YOLO
import cv2
import cvzone
import math
import mediapipe as mp
import numpy as np
import csv
from datetime import datetime
from utils import detect_down, detect_up, in_hoop_region, clean_hoop_pos, clean_ball_pos, get_device, score


class ShotDetector:
    def __init__(self, video_path="pp.mp4", model_path="best.pt"):
        self.model = YOLO(model_path) 
        self.class_names = ['Basketball', 'Basketball Hoop']
        self.device = get_device()
        self.cap = cv2.VideoCapture(video_path)

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

        # Visual feedback
        self.fade_frames = 20
        self.fade_counter = 0
        self.overlay_color = (0, 0, 0)

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

        # Track if we're currently tracking a shot
        self.tracking_shot = False
        self.shot_start_frame = 0

        self.run()

    def run(self):
        while True:
            ret, self.frame = self.cap.read()
            if not ret:
                break

            results = self.model(self.frame, stream=True, device=self.device)

            # Track detections for this frame
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

                    if (conf > .1 or (in_hoop_region(center, self.hoop_pos) and conf > 0.1)) \
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
            
            self.shot_detection()
            self.display_score()

            if self.shot_cooldown > 0:
                self.shot_cooldown -= 1

            self.frame_count += 1
            cv2.imshow('Frame', self.frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()
        
        # Export all data to CSV
        self.export_to_csv()
        
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

    def calculate_angle(self, p1, p2, p3):
        """Calculate angle between three points"""
        v1 = p1 - p2
        v2 = p3 - p2
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
        return np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))
    
    def calculate_distance(self, p1, p2):
        """Calculate Euclidean distance between two points"""
        return np.linalg.norm(p1 - p2)

    def calculate_shot_metrics(self):
        
        if not self.current_shot_frames:
            return {}
        
        metrics = {}
        
        # Find release frame 
        release_frame = None
        release_idx = -1
        for i, frame in enumerate(self.current_shot_frames):
            if frame['in_up_region']:
                release_frame = frame
                release_idx = i
                break
        
        # Find pre-release frame 
        pre_release_frame = None
        if release_idx > 5:
            pre_release_frame = self.current_shot_frames[release_idx - 5]
        
        if release_frame:
            # Right arm metrics
            # Right elbow angle at release
            if all(k in release_frame and release_frame[k] is not None for k in 
                   ['RIGHT_WRIST_x', 'RIGHT_ELBOW_x', 'RIGHT_SHOULDER_x']):
                wrist = np.array([release_frame['RIGHT_WRIST_x'], release_frame['RIGHT_WRIST_y']])
                elbow = np.array([release_frame['RIGHT_ELBOW_x'], release_frame['RIGHT_ELBOW_y']])
                shoulder = np.array([release_frame['RIGHT_SHOULDER_x'], release_frame['RIGHT_SHOULDER_y']])
                metrics['right_elbow_angle'] = self.calculate_angle(wrist, elbow, shoulder)
            
            # Right shoulder angle 
            if all(k in release_frame and release_frame[k] is not None for k in 
                   ['RIGHT_WRIST_x', 'RIGHT_ELBOW_x', 'RIGHT_SHOULDER_x']):
                metrics['right_shoulder_angle'] = self.calculate_angle(shoulder, elbow, wrist)
            
            # Right wrist extension
            if all(k in release_frame and release_frame[k] is not None for k in 
                   ['RIGHT_WRIST_y', 'RIGHT_ELBOW_y']):
                metrics['right_wrist_extension'] = release_frame['RIGHT_WRIST_y'] - release_frame['RIGHT_ELBOW_y']
            
            # Left arm metrics
            if all(k in release_frame and release_frame[k] is not None for k in 
                   ['LEFT_WRIST_x', 'LEFT_ELBOW_x', 'LEFT_SHOULDER_x']):
                l_wrist = np.array([release_frame['LEFT_WRIST_x'], release_frame['LEFT_WRIST_y']])
                l_elbow = np.array([release_frame['LEFT_ELBOW_x'], release_frame['LEFT_ELBOW_y']])
                l_shoulder = np.array([release_frame['LEFT_SHOULDER_x'], release_frame['LEFT_SHOULDER_y']])
                metrics['left_elbow_angle'] = self.calculate_angle(l_wrist, l_elbow, l_shoulder)
            
            
            # Right knee angle
            if all(k in release_frame and release_frame[k] is not None for k in 
                   ['RIGHT_KNEE_x', 'RIGHT_ANKLE_x', 'RIGHT_HIP_x']):
                r_hip = np.array([release_frame['RIGHT_HIP_x'], release_frame['RIGHT_HIP_y']])
                r_knee = np.array([release_frame['RIGHT_KNEE_x'], release_frame['RIGHT_KNEE_y']])
                r_ankle = np.array([release_frame['RIGHT_ANKLE_x'], release_frame['RIGHT_ANKLE_y']])
                metrics['right_knee_angle'] = self.calculate_angle(r_hip, r_knee, r_ankle)
            
            # Left knee angle
            if all(k in release_frame and release_frame[k] is not None for k in 
                   ['LEFT_KNEE_x', 'LEFT_ANKLE_x', 'LEFT_HIP_x']):
                l_hip = np.array([release_frame['LEFT_HIP_x'], release_frame['LEFT_HIP_y']])
                l_knee = np.array([release_frame['LEFT_KNEE_x'], release_frame['LEFT_KNEE_y']])
                l_ankle = np.array([release_frame['LEFT_ANKLE_x'], release_frame['LEFT_ANKLE_y']])
                metrics['left_knee_angle'] = self.calculate_angle(l_hip, l_knee, l_ankle)
            
            # Hip angle 
            if all(k in release_frame and release_frame[k] is not None for k in 
                   ['RIGHT_SHOULDER_x', 'RIGHT_HIP_x', 'RIGHT_KNEE_x']):
                r_hip = np.array([release_frame['RIGHT_HIP_x'], release_frame['RIGHT_HIP_y']])
                shoulder = np.array([release_frame['RIGHT_SHOULDER_x'], release_frame['RIGHT_SHOULDER_y']])
                r_knee = np.array([release_frame['RIGHT_KNEE_x'], release_frame['RIGHT_KNEE_y']])
                metrics['right_hip_angle'] = self.calculate_angle(shoulder, r_hip, r_knee)
            
            
            # Shoulder alignment 
            if all(k in release_frame and release_frame[k] is not None for k in 
                   ['LEFT_SHOULDER_y', 'RIGHT_SHOULDER_y']):
                metrics['shoulder_tilt'] = abs(release_frame['LEFT_SHOULDER_y'] - release_frame['RIGHT_SHOULDER_y'])
            
            # Hip alignment
            if all(k in release_frame and release_frame[k] is not None for k in 
                   ['LEFT_HIP_y', 'RIGHT_HIP_y']):
                metrics['hip_tilt'] = abs(release_frame['LEFT_HIP_y'] - release_frame['RIGHT_HIP_y'])
            
            # Knee alignment 
            if all(k in release_frame and release_frame[k] is not None for k in 
                   ['LEFT_KNEE_x', 'RIGHT_KNEE_x']):
                metrics['knee_spread'] = abs(release_frame['LEFT_KNEE_x'] - release_frame['RIGHT_KNEE_x'])
            
            
            # Release height 
            if 'RIGHT_WRIST_y' in release_frame and release_frame['RIGHT_WRIST_y'] is not None:
                metrics['release_height'] = release_frame['RIGHT_WRIST_y']
            
            # Release point horizontal position
            if 'RIGHT_WRIST_x' in release_frame and release_frame['RIGHT_WRIST_x'] is not None:
                metrics['release_x_position'] = release_frame['RIGHT_WRIST_x']
            
            # Body vertical alignment (shoulder to ankle)
            if all(k in release_frame and release_frame[k] is not None for k in 
                   ['RIGHT_SHOULDER_x', 'RIGHT_ANKLE_x']):
                metrics['body_vertical_alignment'] = abs(release_frame['RIGHT_SHOULDER_x'] - release_frame['RIGHT_ANKLE_x'])
            
            
            if release_frame['ball_x'] is not None and release_frame['hoop_x'] is not None:
                # Horizontal distance to hoop at release
                metrics['ball_hoop_distance_x'] = abs(release_frame['ball_x'] - release_frame['hoop_x'])
                metrics['ball_hoop_distance_y'] = abs(release_frame['ball_y'] - release_frame['hoop_y'])
                
                # Total distance to hoop
                ball_pos = np.array([release_frame['ball_x'], release_frame['ball_y']])
                hoop_pos = np.array([release_frame['hoop_x'], release_frame['hoop_y']])
                metrics['ball_hoop_total_distance'] = self.calculate_distance(ball_pos, hoop_pos)
                
                # Release angle relative to hoop
                if release_frame['ball_y'] != release_frame['hoop_y']:
                    metrics['release_angle_to_hoop'] = np.degrees(np.arctan2(
                        release_frame['hoop_y'] - release_frame['ball_y'],
                        release_frame['hoop_x'] - release_frame['ball_x']
                    ))
            
            
            if pre_release_frame:
                # Knee extension (change in knee angle)
                if 'RIGHT_KNEE_x' in pre_release_frame and pre_release_frame['RIGHT_KNEE_x'] is not None:
                    if all(k in pre_release_frame and pre_release_frame[k] is not None for k in 
                           ['RIGHT_KNEE_x', 'RIGHT_ANKLE_x', 'RIGHT_HIP_x']):
                        pre_hip = np.array([pre_release_frame['RIGHT_HIP_x'], pre_release_frame['RIGHT_HIP_y']])
                        pre_knee = np.array([pre_release_frame['RIGHT_KNEE_x'], pre_release_frame['RIGHT_KNEE_y']])
                        pre_ankle = np.array([pre_release_frame['RIGHT_ANKLE_x'], pre_release_frame['RIGHT_ANKLE_y']])
                        pre_knee_angle = self.calculate_angle(pre_hip, pre_knee, pre_ankle)
                        
                        if 'right_knee_angle' in metrics:
                            metrics['knee_extension'] = metrics['right_knee_angle'] - pre_knee_angle
                
                # Jump height 
                if all(k in pre_release_frame and pre_release_frame[k] is not None for k in ['RIGHT_HIP_y']):
                    if 'RIGHT_HIP_y' in release_frame and release_frame['RIGHT_HIP_y'] is not None:
                        metrics['jump_height'] = pre_release_frame['RIGHT_HIP_y'] - release_frame['RIGHT_HIP_y']
                
                # Elbow extension speed
                if all(k in pre_release_frame and pre_release_frame[k] is not None for k in 
                       ['RIGHT_WRIST_x', 'RIGHT_ELBOW_x', 'RIGHT_SHOULDER_x']):
                    pre_wrist = np.array([pre_release_frame['RIGHT_WRIST_x'], pre_release_frame['RIGHT_WRIST_y']])
                    pre_elbow = np.array([pre_release_frame['RIGHT_ELBOW_x'], pre_release_frame['RIGHT_ELBOW_y']])
                    pre_shoulder = np.array([pre_release_frame['RIGHT_SHOULDER_x'], pre_release_frame['RIGHT_SHOULDER_y']])
                    pre_elbow_angle = self.calculate_angle(pre_wrist, pre_elbow, pre_shoulder)
                    
                    if 'right_elbow_angle' in metrics:
                        metrics['elbow_extension'] = metrics['right_elbow_angle'] - pre_elbow_angle
        
        return metrics

    def shot_detection(self):
        if len(self.hoop_pos) == 0 or len(self.ball_pos) == 0:
            return

        if self.shot_cooldown > 0:
            return

        # Detect ball in 'up' region
        if not self.up:
            self.up = detect_up(self.ball_pos, self.hoop_pos)
            if self.up:
                self.up_frame = self.ball_pos[-1][1]
                # Start tracking this shot
                self.tracking_shot = True
                self.shot_start_frame = self.frame_count
                self.current_shot_frames = []

        # Detect ball in 'down' region
        if self.up and not self.down:
            self.down = detect_down(self.ball_pos, self.hoop_pos)
            if self.down:
                self.down_frame = self.ball_pos[-1][1]

        # Mark shot as pending
        if self.up and self.down and not self.pending_shot:
            self.pending_shot = True
            self.shot_eval_frame = self.frame_count

        # Evaluate make/miss after delay
        if self.pending_shot:
            if self.frame_count - self.shot_eval_frame >= self.shot_delay:
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

                # Record shot data
                shot_metrics = self.calculate_shot_metrics()
                shot_record = {
                    'shot_number': self.attempts,
                    'result': 'make' if is_make else 'miss',
                    'start_frame': self.shot_start_frame,
                    'up_frame': self.up_frame,
                    'down_frame': self.down_frame,
                    'eval_frame': self.frame_count,
                    'duration_frames': self.frame_count - self.shot_start_frame,
                    'duration_seconds': (self.frame_count - self.shot_start_frame) / 30.0,
                }
                shot_record.update(shot_metrics)
                self.shot_data.append(shot_record)

                # Reset shot state
                self.up = False
                self.down = False
                self.pending_shot = False
                self.shot_cooldown = self.cooldown_frames
                self.tracking_shot = False
                self.current_shot_frames = []

    def display_score(self):
        # Score text
        text = str(self.makes) + " / " + str(self.attempts)
        cv2.putText(self.frame, text, (50, 125),
                    cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 6)
        cv2.putText(self.frame, text, (50, 125),
                    cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 0), 3)

        # Overlay text
        if hasattr(self, 'overlay_text'):
            (w, h), _ = cv2.getTextSize(
                self.overlay_text, cv2.FONT_HERSHEY_SIMPLEX, 3, 6
            )
            x = self.frame.shape[1] - w - 40
            y = 100
            cv2.putText(self.frame, self.overlay_text, (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 3,
                        self.overlay_color, 6)

        # Fade effect
        if self.fade_counter > 0:
            alpha = 0.2 * (self.fade_counter / self.fade_frames)
            self.frame = cv2.addWeighted(
                self.frame, 1 - alpha,
                np.full_like(self.frame, self.overlay_color),
                alpha, 0
            )
            self.fade_counter -= 1

    def export_to_csv(self):
        """Append all tracking data to single CSV files"""
        import os
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Export shot summary data (append to single file)
        if self.shot_data:
            shot_csv = 'all_shots.csv'
            file_exists = os.path.isfile(shot_csv)
            
            # Add session info to each shot record
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
            print(f"Shooting percentage: {self.makes/self.attempts*100:.1f}%")
        
        print("\n=== Export Complete ===")


if __name__ == "__main__":
    ShotDetector()
