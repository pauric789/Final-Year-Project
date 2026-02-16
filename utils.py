import math
import numpy as np
import torch

def get_device():
    """Automatically select devices -> mps（Mac） -> cpu"""
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    return device


def score(ball_pos, hoop_pos):
    
    if len(ball_pos) == 0 or len(hoop_pos) == 0:
        return False

    rim_center_x = hoop_pos[-1][0][0]
    rim_center_y = hoop_pos[-1][0][1] - 0.5 * hoop_pos[-1][3]
    
    # Scoring window - slightly more lenient
    rim_half_width = 0.55 * hoop_pos[-1][2]  
    rim_band = 0.5 * hoop_pos[-1][3]  

    # Check all recent points
    recent_points = ball_pos[-30:] if len(ball_pos) >= 30 else ball_pos
    
    for center, frame_num, _, _, _ in recent_points:
        x, y = center
        if (rim_center_x - rim_half_width) < x < (rim_center_x + rim_half_width):
            if (rim_center_y - rim_band) < y < (rim_center_y + rim_band):
                return True

    return False


def detect_down(ball_pos, hoop_pos):
    """Ball is below the bottom of the hoop"""
    y = hoop_pos[-1][0][1] + 0.5 * hoop_pos[-1][3]
    if ball_pos[-1][0][1] > y:
        return True
    return False


def detect_up(ball_pos, hoop_pos):
 
    # Narrower horizontal range 
    x1 = hoop_pos[-1][0][0] - 2 * hoop_pos[-1][2]
    x2 = hoop_pos[-1][0][0] + 2 * hoop_pos[-1][2]
    
    # Higher minimum 
    y1 = hoop_pos[-1][0][1] - 2.5 * hoop_pos[-1][3]  
    y2 = hoop_pos[-1][0][1] - 0.3 * hoop_pos[-1][3]  

    if x1 < ball_pos[-1][0][0] < x2 and y1 < ball_pos[-1][0][1] < y2:
        return True
    return False


def in_hoop_region(center, hoop_pos):
    """Checks if center point is near the hoop"""
    if len(hoop_pos) < 1:
        return False
    x = center[0]
    y = center[1]

    x1 = hoop_pos[-1][0][0] - 1 * hoop_pos[-1][2]
    x2 = hoop_pos[-1][0][0] + 1 * hoop_pos[-1][2]
    y1 = hoop_pos[-1][0][1] - 1 * hoop_pos[-1][3]
    y2 = hoop_pos[-1][0][1] + 0.5 * hoop_pos[-1][3]

    if x1 < x < x2 and y1 < y < y2:
        return True
    return False


def clean_ball_pos(ball_pos, frame_count):
    """Removes inaccurate data points"""
    if len(ball_pos) > 1:
        w1 = ball_pos[-2][2]
        h1 = ball_pos[-2][3]
        w2 = ball_pos[-1][2]
        h2 = ball_pos[-1][3]

        x1 = ball_pos[-2][0][0]
        y1 = ball_pos[-2][0][1]
        x2 = ball_pos[-1][0][0]
        y2 = ball_pos[-1][0][1]

        f1 = ball_pos[-2][1]
        f2 = ball_pos[-1][1]
        f_dif = f2 - f1

        dist = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        max_dist = 4 * math.sqrt((w1) ** 2 + (h1) ** 2)

        # Ball should not move 4x its diameter within 5 frames
        if (dist > max_dist) and (f_dif < 5):
            ball_pos.pop()

        # Ball should be relatively square
        elif (w2*1.4 < h2) or (h2*1.4 < w2):
            ball_pos.pop()

    # Keep longer history for better scoring
    if len(ball_pos) > 0:
        if frame_count - ball_pos[0][1] > 50:  # Increased from 30
            ball_pos.pop(0)

    return ball_pos


def clean_hoop_pos(hoop_pos):
    """Prevents jumping from one hoop to another"""
    if len(hoop_pos) > 1:
        x1 = hoop_pos[-2][0][0]
        y1 = hoop_pos[-2][0][1]
        x2 = hoop_pos[-1][0][0]
        y2 = hoop_pos[-1][0][1]

        w1 = hoop_pos[-2][2]
        h1 = hoop_pos[-2][3]
        w2 = hoop_pos[-1][2]
        h2 = hoop_pos[-1][3]

        f1 = hoop_pos[-2][1]
        f2 = hoop_pos[-1][1]
        f_dif = f2 - f1

        dist = math.sqrt((x2-x1)**2 + (y2-y1)**2)
        max_dist = 0.5 * math.sqrt(w1 ** 2 + h1 ** 2)

        # Hoop should not move 0.5x its diameter within 5 frames
        if dist > max_dist and f_dif < 5:
            hoop_pos.pop()

        # Hoop should be relatively square
        if (w2*1.3 < h2) or (h2*1.3 < w2):
            hoop_pos.pop()

    # Remove old points
    if len(hoop_pos) > 25:
        hoop_pos.pop(0)

    return hoop_pos
