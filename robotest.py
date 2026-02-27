#!/usr/bin/env python3
"""
Basketball Shot Detector with YOLO Player Detection + MediaPipe Pose
---------------------------------------------------------------------
- YOLO for player detection
- ByteTrack for multi-player tracking
- MediaPipe Pose on shooting player
- CV2 window display
- CSV export with detailed metrics
"""

import cv2
import numpy as np
import mediapipe as mp
import csv
from datetime import datetime
from ultralytics import YOLO
import cvzone
import os
import argparse
from collections import defaultdict

try:
    from supervision import ByteTrack, Detections
    SUPERVISION_AVAILABLE = True
except ImportError:
    SUPERVISION_AVAILABLE = False
    print("Warning: Supervision not installed. Install with: pip install supervision")


# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    # Video and Models
    VIDEO_PATH = "ncaa.mp4"
    MODEL_PATH = "best copy.pt"  # Single YOLO model for everything
    
    # Class IDs from the model:
    # 0=ball, 1=ball-in-basket, 2=number, 3=player, 4=player-in-possession,
    # 5=player-jump-shot, 6=player-layup-dunk, 7=player-shot-block, 8=referee, 9=rim
    BALL_CLASS_ID = 0
    BALL_IN_BASKET_CLASS_ID = 1
    RIM_CLASS_ID = 9
    PLAYER_CLASS_IDS = [3, 4, 5, 6, 7]
    
    # Output
    OUTPUT_CSV = "all_shots.csv"
    SHOW_WINDOW = True
    
    # Shot Event Class IDs
    JUMP_SHOT_CLASS_ID = 5
    LAYUP_DUNK_CLASS_ID = 6
    
    # Detection Parameters
    DETECTION_CONFIDENCE = 0.2
    IOU_THRESHOLD = 0.7
    BALL_CONFIDENCE = 0.1
    HOOP_CONFIDENCE = 0.1
    PLAYER_CONFIDENCE = 0.5
    
    # Tracking
    SHOOTER_PROXIMITY_THRESHOLD = 300  # pixels - distance to ball to identify shooter


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def calculate_angle(point1, point2, point3):
    """Calculate angle between three points"""
    if point1 is None or point2 is None or point3 is None:
        return None
    
    a = np.array(point1)
    b = np.array(point2)
    c = np.array(point3)
    
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    
    if angle > 180.0:
        angle = 360 - angle
    
    return angle


class ShotEventTracker:
    """
    Tracks shot events using model class detections (jump-shot, layup-dunk, ball-in-basket).
    Also uses ball-near-rim proximity as a fallback scoring method.
    """
    def __init__(self, reset_time_frames=50, minimum_frames_between_starts=15,
                 cooldown_frames_after_made=15):
        self.reset_time_frames = reset_time_frames
        self.minimum_frames_between_starts = minimum_frames_between_starts
        self.cooldown_frames_after_made = cooldown_frames_after_made
        
        self.shot_in_progress = False
        self.shot_start_frame = None
        self.last_start_frame = -999
        self.cooldown_until = -1
        self.ball_near_rim = False  # fallback scoring
    
    def update(self, frame_index, has_jump_shot, has_layup_dunk, has_ball_in_basket,
               ball_near_rim=False):
        """Update tracker with current frame detections. Returns list of event dicts."""
        events = []
        
        # Skip if in cooldown
        if frame_index < self.cooldown_until:
            return events
        
        has_shot_action = has_jump_shot or has_layup_dunk
        
        # Track ball near rim during shot
        if self.shot_in_progress and ball_near_rim:
            self.ball_near_rim = True
        
        # If shot is in progress and we still see shot action, extend the timeout window
        # (the actual release may happen well after the initial detection)
        if self.shot_in_progress and has_shot_action:
            self.shot_start_frame = frame_index
        
        # Detect shot START
        if has_shot_action and not self.shot_in_progress:
            if frame_index - self.last_start_frame >= self.minimum_frames_between_starts:
                self.shot_in_progress = True
                self.shot_start_frame = frame_index
                self.last_start_frame = frame_index
                self.ball_near_rim = False
                shot_type = "jump-shot" if has_jump_shot else "layup-dunk"
                events.append({"event": "START", "frame": frame_index, "type": shot_type})
        
        # Check for MADE or MISSED
        if self.shot_in_progress:
            is_made = has_ball_in_basket or self.ball_near_rim
            
            if is_made:
                events.append({"event": "MADE", "frame": frame_index,
                             "start_frame": self.shot_start_frame})
                self.shot_in_progress = False
                self.shot_start_frame = None
                self.ball_near_rim = False
                self.cooldown_until = frame_index + self.cooldown_frames_after_made
            elif frame_index - self.shot_start_frame >= self.reset_time_frames:
                events.append({"event": "MISSED", "frame": frame_index,
                             "start_frame": self.shot_start_frame})
                self.shot_in_progress = False
                self.shot_start_frame = None
                self.ball_near_rim = False
        
        return events


def clean_hoop_pos(hoop_positions):
    """Clean hoop position tracking (hoop should be stable)"""
    if not hoop_positions:
        return None
    avg_x = sum(h[0][0] for h in hoop_positions) / len(hoop_positions)
    avg_y = sum(h[0][1] for h in hoop_positions) / len(hoop_positions)
    return (int(avg_x), int(avg_y))


def get_device():
    """Get available device for YOLO"""
    try:
        import torch
        return 0 if torch.cuda.is_available() else 'cpu'
    except:
        return 'cpu'


# ============================================================================
# MULTI-PLAYER SHOT DETECTOR WITH YOLO + MEDIAPIPE
# ============================================================================

class MultiPlayerShotDetector:
    def __init__(self, video_path, model_path=None,
                 output_csv="all_shots.csv", show_window=True):
        
        self.video_path = video_path
        self.output_csv = output_csv
        self.show_window = show_window
        
        print(f"Initializing Multi-Player Shot Detector...")
        print(f"Video: {video_path}")
        
        # Initialize single YOLO model for all detections
        model_path = model_path or Config.MODEL_PATH
        self.model = YOLO(model_path)
        self.device = get_device()
        print(f"Model: {model_path}")
        print(f"Classes: {self.model.names}")
        print(f"Using device: {self.device}")
        
        # Initialize video capture
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        if self.fps == 0:
            self.fps = 30.0
        
        total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"Video: {total_frames} frames @ {self.fps:.2f} fps")
        
        # Initialize ByteTrack for multi-player tracking
        if not SUPERVISION_AVAILABLE:
            raise ImportError("Supervision not installed. Run: pip install supervision")
        
        self.tracker = ByteTrack(
            track_activation_threshold=0.4,
            lost_track_buffer=30,
            minimum_matching_threshold=0.8,
            frame_rate=self.fps
        )
        print("ByteTrack multi-player tracker initialized")
        
        # Initialize MediaPipe Pose
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        print("MediaPipe Pose initialized")
        
        # Ball and hoop tracking
        self.ball_pos = []
        self.hoop_pos = []
        self.frame_count = 0
        self.frame = None
        
        # Player tracking
        self.players = defaultdict(lambda: {
            'bbox': None,
            'last_seen': 0,
            'pose_landmarks': None,
            'is_shooter': False,
            'shot_attempts': 0,
            'shot_makes': 0,
        })
        
        # Shot event tracker (notebook approach)
        self.shot_event_tracker = ShotEventTracker(
            reset_time_frames=int(self.fps * 5.0),
            minimum_frames_between_starts=int(self.fps * 3.0),
            cooldown_frames_after_made=int(self.fps * 2.0),
        )
        self.current_shooter_id = None
        self.shot_start_frame = 0
        print(f"ShotEventTracker initialized (reset={int(self.fps * 5.0)}f, min_between={int(self.fps * 3.0)}f)")
        
        # Stats
        self.total_makes = 0
        self.total_attempts = 0
        self.shot_data = []
        self.player_shot_data = defaultdict(list)
        
        # Visual feedback
        self.fade_frames = 20
        self.fade_counter = 0
        self.overlay_color = (0, 0, 0)
        self.overlay_text = ""
        
        # Pose at release
        self.release_pose = None
        
        print("Initialization complete!\n")
    
    def identify_shooter(self, tracked_players, ball_pos):
        """Identify which player is shooting based on proximity to ball"""
        if ball_pos is None or len(tracked_players.xyxy) == 0:
            return None
        
        min_distance = float('inf')
        shooter_id = None
        
        for i in range(len(tracked_players.xyxy)):
            bbox = tracked_players.xyxy[i]
            track_id = tracked_players.tracker_id[i] if tracked_players.tracker_id is not None else None
            
            if track_id is None:
                continue
            
            # Calculate center of player bbox
            player_center_x = (bbox[0] + bbox[2]) / 2
            player_center_y = (bbox[1] + bbox[3]) / 2
            
            # Calculate distance to ball
            distance = np.sqrt((player_center_x - ball_pos[0])**2 + 
                             (player_center_y - ball_pos[1])**2)
            
            if distance < min_distance:
                min_distance = distance
                shooter_id = int(track_id)
        
        # Only consider as shooter if reasonably close
        if min_distance < Config.SHOOTER_PROXIMITY_THRESHOLD:
            return shooter_id
        
        return None
    
    def process_pose_for_player(self, player_bbox):
        """Extract pose landmarks for a specific player"""
        x1, y1, x2, y2 = map(int, player_bbox)
        
        # Expand bbox for better pose detection
        padding = 20
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(self.frame.shape[1], x2 + padding)
        y2 = min(self.frame.shape[0], y2 + padding)
        
        # Crop player region
        player_crop = self.frame[y1:y2, x1:x2]
        
        if player_crop.size == 0:
            return None
        
        # Run pose estimation
        rgb = cv2.cvtColor(player_crop, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb)
        
        if not results.pose_landmarks:
            return None
        
        # Convert landmarks to full frame coordinates
        landmarks = results.pose_landmarks.landmark
        crop_h, crop_w = player_crop.shape[:2]
        
        def get_point(landmark_id):
            lm = landmarks[landmark_id]
            if lm.visibility < 0.5:
                return None
            # Convert from crop coordinates to full frame coordinates
            x = int(lm.x * crop_w) + x1
            y = int(lm.y * crop_h) + y1
            return (x, y)
        
        pose_data = {
            'right_wrist': get_point(self.mp_pose.PoseLandmark.RIGHT_WRIST),
            'left_wrist': get_point(self.mp_pose.PoseLandmark.LEFT_WRIST),
            'right_elbow': get_point(self.mp_pose.PoseLandmark.RIGHT_ELBOW),
            'left_elbow': get_point(self.mp_pose.PoseLandmark.LEFT_ELBOW),
            'right_shoulder': get_point(self.mp_pose.PoseLandmark.RIGHT_SHOULDER),
            'left_shoulder': get_point(self.mp_pose.PoseLandmark.LEFT_SHOULDER),
            'right_hip': get_point(self.mp_pose.PoseLandmark.RIGHT_HIP),
            'left_hip': get_point(self.mp_pose.PoseLandmark.LEFT_HIP),
            'right_knee': get_point(self.mp_pose.PoseLandmark.RIGHT_KNEE),
            'left_knee': get_point(self.mp_pose.PoseLandmark.LEFT_KNEE),
            'right_ankle': get_point(self.mp_pose.PoseLandmark.RIGHT_ANKLE),
            'left_ankle': get_point(self.mp_pose.PoseLandmark.LEFT_ANKLE),
        }
        
        return pose_data
    
    def draw_pose_on_frame(self, pose_data, color=(0, 255, 255)):
        """Draw pose landmarks on the full frame"""
        if pose_data is None:
            return
        
        # Draw connections
        connections = [
            ('right_shoulder', 'right_elbow'),
            ('right_elbow', 'right_wrist'),
            ('left_shoulder', 'left_elbow'),
            ('left_elbow', 'left_wrist'),
            ('right_shoulder', 'left_shoulder'),
            ('right_shoulder', 'right_hip'),
            ('left_shoulder', 'left_hip'),
            ('right_hip', 'left_hip'),
            ('right_hip', 'right_knee'),
            ('right_knee', 'right_ankle'),
            ('left_hip', 'left_knee'),
            ('left_knee', 'left_ankle'),
        ]
        
        for start_key, end_key in connections:
            start_point = pose_data.get(start_key)
            end_point = pose_data.get(end_key)
            if start_point and end_point:
                cv2.line(self.frame, start_point, end_point, color, 2)
        
        # Draw landmarks
        for key, point in pose_data.items():
            if point:
                cv2.circle(self.frame, point, 4, (0, 0, 255), -1)
    
    def calculate_shot_metrics(self, pose_data, ball_pos, hoop_pos):
        """Calculate detailed shot metrics from pose data"""
        if pose_data is None:
            return {}
        
        metrics = {}
        
        # Right elbow angle
        if all(pose_data.get(k) for k in ['right_shoulder', 'right_elbow', 'right_wrist']):
            metrics['right_elbow_angle'] = calculate_angle(
                pose_data['right_shoulder'],
                pose_data['right_elbow'],
                pose_data['right_wrist']
            )
            metrics['right_shoulder_angle'] = metrics['right_elbow_angle']
        
        # Left elbow angle
        if all(pose_data.get(k) for k in ['left_shoulder', 'left_elbow', 'left_wrist']):
            metrics['left_elbow_angle'] = calculate_angle(
                pose_data['left_shoulder'],
                pose_data['left_elbow'],
                pose_data['left_wrist']
            )
        
        # Knee angles
        if all(pose_data.get(k) for k in ['right_hip', 'right_knee', 'right_ankle']):
            metrics['right_knee_angle'] = calculate_angle(
                pose_data['right_hip'],
                pose_data['right_knee'],
                pose_data['right_ankle']
            )
        
        if all(pose_data.get(k) for k in ['left_hip', 'left_knee', 'left_ankle']):
            metrics['left_knee_angle'] = calculate_angle(
                pose_data['left_hip'],
                pose_data['left_knee'],
                pose_data['left_ankle']
            )
        
        # Hip angle
        if all(pose_data.get(k) for k in ['right_shoulder', 'right_hip', 'right_knee']):
            metrics['right_hip_angle'] = calculate_angle(
                pose_data['right_shoulder'],
                pose_data['right_hip'],
                pose_data['right_knee']
            )
        
        # Shoulder tilt
        if pose_data.get('right_shoulder') and pose_data.get('left_shoulder'):
            shoulder_diff = abs(pose_data['right_shoulder'][1] - pose_data['left_shoulder'][1])
            metrics['shoulder_tilt'] = int(shoulder_diff / 10)
        
        # Hip tilt
        if pose_data.get('right_hip') and pose_data.get('left_hip'):
            hip_diff = abs(pose_data['right_hip'][1] - pose_data['left_hip'][1])
            metrics['hip_tilt'] = int(hip_diff / 10)
        
        # Knee spread
        if pose_data.get('right_knee') and pose_data.get('left_knee'):
            knee_spread = abs(pose_data['right_knee'][0] - pose_data['left_knee'][0])
            metrics['knee_spread'] = int(knee_spread)
        
        # Release height and position
        if ball_pos:
            metrics['release_height'] = int(ball_pos[1])
            metrics['release_x_position'] = int(ball_pos[0])
        
        # Body vertical alignment
        if pose_data.get('right_shoulder') and pose_data.get('right_hip'):
            alignment = abs(pose_data['right_shoulder'][0] - pose_data['right_hip'][0])
            metrics['body_vertical_alignment'] = int(alignment)
        
        # Ball to hoop distance
        if ball_pos and hoop_pos:
            metrics['ball_hoop_distance_x'] = abs(int(ball_pos[0] - hoop_pos[0]))
            metrics['ball_hoop_distance_y'] = abs(int(ball_pos[1] - hoop_pos[1]))
            metrics['ball_hoop_total_distance'] = int(np.sqrt(
                (ball_pos[0] - hoop_pos[0])**2 + (ball_pos[1] - hoop_pos[1])**2
            ))
            
            angle = np.arctan2(hoop_pos[1] - ball_pos[1], hoop_pos[0] - ball_pos[0])
            metrics['release_angle_to_hoop'] = abs(angle * 180 / np.pi)
        
        # Wrist extension (placeholder)
        metrics['right_wrist_extension'] = -53
        
        return metrics
    
    def process_frame(self):
        """Process a single frame"""
        # Run single model for all detections
        results = self.model(self.frame, stream=True, device=self.device, verbose=False)
        
        frame_ball_pos = None
        frame_hoop_pos = None
        player_detections_list = []
        has_jump_shot = False
        has_layup_dunk = False
        has_ball_in_basket = False
        jump_shot_bboxes = []
        shooter_pose = None
        
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                w, h = x2 - x1, y2 - y1
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                center = (x1 + w // 2, y1 + h // 2)
                
                # Ball detection
                if conf > Config.BALL_CONFIDENCE and cls == Config.BALL_CLASS_ID:
                    self.ball_pos.append((center, self.frame_count, w, h, conf))
                    frame_ball_pos = center
                    cvzone.cornerRect(self.frame, (x1, y1, w, h), colorR=(0, 255, 0))
                    cv2.circle(self.frame, center, 5, (0, 255, 0), -1)
                
                # Ball-in-basket detection
                if conf > Config.BALL_CONFIDENCE and cls == Config.BALL_IN_BASKET_CLASS_ID:
                    has_ball_in_basket = True
                    cv2.putText(self.frame, "BALL IN BASKET", (x1, y1 - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    cvzone.cornerRect(self.frame, (x1, y1, w, h), colorR=(0, 255, 0))
                
                # Jump shot detection
                if conf > Config.DETECTION_CONFIDENCE and cls == Config.JUMP_SHOT_CLASS_ID:
                    has_jump_shot = True
                    jump_shot_bboxes.append([float(x1), float(y1), float(x2), float(y2)])
                    cv2.putText(self.frame, "JUMP SHOT", (x1, y2 + 20),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
                
                # Layup/dunk detection
                if conf > Config.DETECTION_CONFIDENCE and cls == Config.LAYUP_DUNK_CLASS_ID:
                    has_layup_dunk = True
                    jump_shot_bboxes.append([float(x1), float(y1), float(x2), float(y2)])
                    cv2.putText(self.frame, "LAYUP/DUNK", (x1, y2 + 20),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
                
                # Rim/hoop detection
                if conf > Config.HOOP_CONFIDENCE and cls == Config.RIM_CLASS_ID:
                    self.hoop_pos.append((center, self.frame_count, w, h, conf))
                    frame_hoop_pos = center
                    cvzone.cornerRect(self.frame, (x1, y1, w, h), colorR=(255, 0, 0))
                    cv2.circle(self.frame, center, 5, (255, 0, 0), -1)
                
                # Player detections (collected for tracking)
                if conf > Config.PLAYER_CONFIDENCE and cls in Config.PLAYER_CLASS_IDS:
                    player_detections_list.append([float(x1), float(y1), float(x2), float(y2), conf])
        
        # Track players with ByteTrack
        if len(player_detections_list) > 0:
            detections_array = np.array(player_detections_list)
            detections = Detections(
                xyxy=detections_array[:, :4],
                confidence=detections_array[:, 4]
            )
            tracked_players = self.tracker.update_with_detections(detections)
        else:
            tracked_players = Detections.empty()
        
        # Run MediaPipe pose on jump-shot/layup player (the shooter)
        if jump_shot_bboxes:
            shot_bbox = jump_shot_bboxes[0]
            pose_data = self.process_pose_for_player(shot_bbox)
            if pose_data:
                self.draw_pose_on_frame(pose_data, color=(0, 255, 255))
                shooter_pose = pose_data
                self.release_pose = pose_data
        
        # Update player data
        if len(tracked_players.xyxy) > 0:
            for i in range(len(tracked_players.xyxy)):
                bbox = tracked_players.xyxy[i]
                track_id = tracked_players.tracker_id[i] if tracked_players.tracker_id is not None else None
                
                if track_id is None:
                    continue
                
                track_id = int(track_id)
                
                # Update player info
                self.players[track_id]['bbox'] = bbox
                self.players[track_id]['last_seen'] = self.frame_count
                
                # Draw player bounding box
                x1, y1, x2, y2 = map(int, bbox)
                color = (255, 0, 255) if track_id == self.current_shooter_id else (255, 255, 0)
                cv2.rectangle(self.frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(self.frame, f"Player {track_id}", (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # --- Shot Event Detection (notebook approach) ---
        # Debug: log when shot-related classes or ball/rim are detected
        if has_jump_shot or has_layup_dunk or has_ball_in_basket:
            print(f"  [F{self.frame_count}] jump_shot={has_jump_shot} layup={has_layup_dunk} ball_in_basket={has_ball_in_basket} (in_progress={self.shot_event_tracker.shot_in_progress})")
        if frame_ball_pos:
            print(f"  [F{self.frame_count}] Ball detected at {frame_ball_pos}")
        if frame_hoop_pos:
            print(f"  [F{self.frame_count}] Rim detected at {frame_hoop_pos}")
        
        # Check if ball is near the rim (fallback scoring)
        # Use historical positions: check latest ball against latest rim across recent frames
        ball_near_rim = False
        
        # Get recent ball position (current frame or last 15 frames)
        recent_ball = frame_ball_pos
        if not recent_ball:
            for bp, bf, bw, bh, bc in reversed(self.ball_pos):
                if self.frame_count - bf <= 15:
                    recent_ball = bp
                    break
        
        # Get recent hoop position (current frame or last 30 frames — hoop is static)
        recent_hoop = frame_hoop_pos
        if not recent_hoop:
            for hp, hf, hw, hh, hc in reversed(self.hoop_pos):
                if self.frame_count - hf <= 30:
                    recent_hoop = hp
                    break
        
        if recent_ball and recent_hoop:
            dist = np.sqrt((recent_ball[0] - recent_hoop[0])**2 + 
                          (recent_ball[1] - recent_hoop[1])**2)
            # Ball must be near rim AND at/below rim height (ball dropping through, not approaching)
            # In image coords, y increases downward, so ball_y >= rim_y means ball is at or below rim
            ball_at_or_below_rim = recent_ball[1] >= recent_hoop[1] - 20  # 20px tolerance
            if dist < 150 and ball_at_or_below_rim:
                ball_near_rim = True
                if self.shot_event_tracker.shot_in_progress:
                    print(f"  [F{self.frame_count}] Ball near rim! dist={dist:.0f}px (ball={recent_ball}, rim={recent_hoop})")
            elif dist < 150 and self.shot_event_tracker.shot_in_progress:
                print(f"  [F{self.frame_count}] Ball near rim but still above (dist={dist:.0f}px, ball_y={recent_ball[1]}, rim_y={recent_hoop[1]})")
        
        events = self.shot_event_tracker.update(
            frame_index=self.frame_count,
            has_jump_shot=has_jump_shot,
            has_layup_dunk=has_layup_dunk,
            has_ball_in_basket=has_ball_in_basket,
            ball_near_rim=ball_near_rim,
        )
        
        for event in events:
            if event["event"] == "START":
                self.shot_start_frame = event["frame"]
                print(f"Frame {self.frame_count}: Shot START ({event['type']})")
                
                # Identify shooter from jump-shot/layup bbox proximity to ball
                if frame_ball_pos and jump_shot_bboxes:
                    # Use jump-shot bbox center as shooter location
                    shot_bbox = jump_shot_bboxes[0]
                    shot_center = ((shot_bbox[0] + shot_bbox[2]) / 2,
                                   (shot_bbox[1] + shot_bbox[3]) / 2)
                    self.current_shooter_id = self.identify_shooter(tracked_players, shot_center)
                elif frame_ball_pos:
                    self.current_shooter_id = self.identify_shooter(tracked_players, frame_ball_pos)
                
                if self.current_shooter_id:
                    print(f"  Shooter identified: Player {self.current_shooter_id}")
                    # Capture pose at shot start
                    if self.players[self.current_shooter_id]['pose_landmarks']:
                        self.release_pose = self.players[self.current_shooter_id]['pose_landmarks']
            
            elif event["event"] == "MADE":
                self.total_attempts += 1
                self.total_makes += 1
                self.overlay_color = (0, 255, 0)
                self.overlay_text = "MAKE"
                self.fade_counter = self.fade_frames
                print(f"Frame {self.frame_count}: MAKE! ({self.total_makes}/{self.total_attempts})")
                self._record_shot(is_make=True, event=event)
            
            elif event["event"] == "MISSED":
                self.total_attempts += 1
                self.overlay_color = (0, 0, 255)
                self.overlay_text = "MISS"
                self.fade_counter = self.fade_frames
                print(f"Frame {self.frame_count}: MISS ({self.total_makes}/{self.total_attempts})")
                self._record_shot(is_make=False, event=event)
        
        return frame_ball_pos, frame_hoop_pos, shooter_pose
    
    def _record_shot(self, is_make, event):
        """Record a shot event to stats and CSV data"""
        shooter_id = self.current_shooter_id
        
        if shooter_id:
            print(f"  By: Player {shooter_id}")
        
        # Update player stats
        if shooter_id is not None:
            self.players[shooter_id]['shot_attempts'] += 1
            if is_make:
                self.players[shooter_id]['shot_makes'] += 1
        
        # Calculate metrics
        ball_at_release = self.ball_pos[-1][0] if self.ball_pos else None
        hoop_at_release = clean_hoop_pos(self.hoop_pos)
        
        metrics = self.calculate_shot_metrics(self.release_pose, ball_at_release, hoop_at_release)
        
        # Create shot record
        start_frame = event.get('start_frame', self.shot_start_frame)
        shot_record = {
            'shot_number': self.total_attempts,
            'player_id': shooter_id if shooter_id else -1,
            'result': 'make' if is_make else 'miss',
            'start_frame': start_frame,
            'end_frame': self.frame_count,
            'duration_frames': self.frame_count - start_frame,
            'duration_seconds': (self.frame_count - start_frame) / self.fps,
        }
        shot_record.update(metrics)
        
        session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        shot_record['session_id'] = session_id
        shot_record['session_timestamp'] = session_timestamp
        
        self.shot_data.append(shot_record)
        
        if shooter_id is not None:
            self.player_shot_data[shooter_id].append(shot_record)
        
        # Reset shooter
        self.release_pose = None
        self.current_shooter_id = None
    
    def display_overlay(self):
        """Display score and feedback overlay"""
        # Overall score
        text = f"Overall: {self.total_makes} / {self.total_attempts}"
        cv2.putText(self.frame, text, (50, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 4)
        cv2.putText(self.frame, text, (50, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 2)
        
        # Per-player stats (top 3)
        y_offset = 140
        active_players = sorted(
            [(pid, data) for pid, data in self.players.items() if data['shot_attempts'] > 0],
            key=lambda x: x[1]['shot_attempts'],
            reverse=True
        )[:3]
        
        for player_id, player_data in active_players:
            makes = player_data['shot_makes']
            attempts = player_data['shot_attempts']
            pct = (makes / attempts * 100) if attempts > 0 else 0
            player_text = f"P{player_id}: {makes}/{attempts} ({pct:.0f}%)"
            cv2.putText(self.frame, player_text, (50, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(self.frame, player_text, (50, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
            y_offset += 35
        
        # Frame counter
        frame_text = f"Frame: {self.frame_count}"
        cv2.putText(self.frame, frame_text, (50, self.frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Make/Miss text
        if self.overlay_text:
            (w, h), _ = cv2.getTextSize(self.overlay_text, cv2.FONT_HERSHEY_SIMPLEX, 2, 5)
            x = self.frame.shape[1] - w - 40
            y = 80
            cv2.putText(self.frame, self.overlay_text, (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, self.overlay_color, 5)
        
        # Fade effect
        if self.fade_counter > 0:
            alpha = 0.2 * (self.fade_counter / self.fade_frames)
            overlay = np.full_like(self.frame, self.overlay_color, dtype=np.uint8)
            self.frame = cv2.addWeighted(self.frame, 1 - alpha, overlay, alpha, 0)
            self.fade_counter -= 1
    
    def export_to_csv(self):
        """Export shot data to CSV"""
        if not self.shot_data:
            print("No shots to export")
            return
        
        file_exists = os.path.isfile(self.output_csv)
        
        with open(self.output_csv, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.shot_data[0].keys())
            if not file_exists:
                writer.writeheader()
            writer.writerows(self.shot_data)
        
        print(f"\n{'='*60}")
        print(f"Shot data exported to {self.output_csv}")
        print(f"Session: {self.shot_data[0]['session_id']}")
        print(f"Total shots: {len(self.shot_data)}")
        print(f"Overall: {self.total_makes}/{self.total_attempts}", end="")
        if self.total_attempts > 0:
            print(f" ({self.total_makes/self.total_attempts*100:.1f}%)")
        else:
            print()
        
        # Per-player stats
        print(f"\nPer-Player Stats:")
        for player_id, shots in self.player_shot_data.items():
            makes = sum(1 for s in shots if s['result'] == 'make')
            attempts = len(shots)
            pct = (makes / attempts * 100) if attempts > 0 else 0
            print(f"  Player {player_id}: {makes}/{attempts} ({pct:.1f}%)")
        
        print(f"{'='*60}\n")
    
    def run(self):
        """Main processing loop"""
        print("Starting multi-player shot detection...")
        print("Press 'q' to quit")
        print("="*60)
        
        while True:
            ret, self.frame = self.cap.read()
            if not ret:
                break
            
            # Process frame (includes shot event detection)
            frame_ball_pos, frame_hoop_pos, shooter_pose = self.process_frame()
            
            # Display overlay
            self.display_overlay()
            
            # Show frame
            if self.show_window:
                cv2.imshow('Multi-Player Basketball Shot Detector', self.frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("\nQuitting...")
                    break
            
            self.frame_count += 1
            
            if self.frame_count % 100 == 0:
                print(f"Processed {self.frame_count} frames... (shots: {self.total_makes}/{self.total_attempts})")
        # Cleanup
        self.cap.release()
        if self.show_window:
            cv2.destroyAllWindows()
        
        # Export data
        self.export_to_csv()
        
        print("\nProcessing complete!")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Multi-Player Basketball Shot Detector')
    parser.add_argument('--video', type=str, default=Config.VIDEO_PATH, help='Path to video file')
    parser.add_argument('--model', type=str, default=Config.MODEL_PATH, help='Path to YOLO model')
    parser.add_argument('--output', type=str, default=Config.OUTPUT_CSV, help='Output CSV file')
    parser.add_argument('--no-window', action='store_true', help='Disable CV2 window display')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.video):
        print(f"Error: Video file not found: {args.video}")
        return
    
    if not os.path.exists(args.model):
        print(f"Error: Model file not found: {args.model}")
        return
    
    # Create detector
    detector = MultiPlayerShotDetector(
        video_path=args.video,
        model_path=args.model,
        output_csv=args.output,
        show_window=not args.no_window
    )
    
    # Run
    detector.run()


if __name__ == "__main__":
    main()