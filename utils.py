import math
import numpy as np
import torch
from enum import Enum


def calculate_angle(p1, p2, p3):
    """Calculate angle between three points"""
    v1 = p1 - p2
    v2 = p3 - p2
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
    return np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))


def calculate_distance(p1, p2):
    """Calculate Euclidean distance between two points"""
    return np.linalg.norm(p1 - p2)

def get_device():
    """Automatically select devices -> mps（Mac） -> cpu"""
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    return device


# ------------------------------------------------------------------
# Shot State Machine
# ------------------------------------------------------------------

class ShotState(Enum):
    """Explicit states for shot detection — replaces loose boolean flags."""
    IDLE = "idle"
    BALL_RISING = "rising"
    BALL_DESCENDING = "descending"
    EVALUATING = "evaluating"
    COOLDOWN = "cooldown"


# ------------------------------------------------------------------
# Ball velocity
# ------------------------------------------------------------------

def _ball_velocity(ball_pos, n=3):
    """Average velocity (dx, dy) over the last *n* tracked ball positions.
    Returns (0, 0) when there is not enough data.
    """
    if len(ball_pos) < 2:
        return (0, 0)
    pts = ball_pos[-n:] if len(ball_pos) >= n else ball_pos
    dx_sum = 0
    dy_sum = 0
    count = 0
    for i in range(1, len(pts)):
        f_diff = pts[i][1] - pts[i - 1][1]
        if f_diff == 0:
            continue
        dx_sum += (pts[i][0][0] - pts[i - 1][0][0]) / f_diff
        dy_sum += (pts[i][0][1] - pts[i - 1][0][1]) / f_diff
        count += 1
    if count == 0:
        return (0, 0)
    return (dx_sum / count, dy_sum / count)


# ------------------------------------------------------------------
# Trajectory fitting
# ------------------------------------------------------------------

def fit_trajectory(ball_pos, min_points=6):
    """Fit a parabola to recent ball positions and return arc parameters.

    A basketball shot follows a parabolic arc.  By fitting y = at² + bt + c
    we can validate that the ball is actually following a shot arc (R² check)
    and extract useful features like launch angle and apex height.

    Returns a dict with trajectory info, or None if not enough data.
    """
    if len(ball_pos) < min_points:
        return None

    pts = ball_pos[-20:] if len(ball_pos) >= 20 else ball_pos

    frames = np.array([bp[1] for bp in pts], dtype=float)
    xs = np.array([bp[0][0] for bp in pts], dtype=float)
    ys = np.array([bp[0][1] for bp in pts], dtype=float)

    try:
        coeffs_y = np.polyfit(frames, ys, 2)
        coeffs_x = np.polyfit(frames, xs, 1)
    except (np.linalg.LinAlgError, ValueError):
        return None

    a, b, c = coeffs_y

    # a > 0 means parabolic arc (gravity pulling down in image coords)
    if a <= 0.01:
        return None

    # Apex frame (where dy/dt = 0)
    apex_frame = -b / (2 * a)
    apex_y = float(np.polyval(coeffs_y, apex_frame))
    apex_x = float(np.polyval(coeffs_x, apex_frame))

    # R² goodness of fit
    y_pred = np.polyval(coeffs_y, frames)
    ss_res = np.sum((ys - y_pred) ** 2)
    ss_tot = np.sum((ys - np.mean(ys)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

    # Launch angle at first tracked point
    t0 = frames[0]
    dy_dt = 2 * a * t0 + b
    dx_dt = coeffs_x[0]
    launch_angle = float(np.degrees(np.arctan2(-dy_dt, dx_dt))) if dx_dt != 0 else 0.0

    return {
        'coeffs_y': coeffs_y,
        'coeffs_x': coeffs_x,
        'apex': (apex_x, apex_y),
        'apex_frame': float(apex_frame),
        'r_squared': float(r_squared),
        'launch_angle': launch_angle,
    }


# ------------------------------------------------------------------
# Release detection (ball-player separation)
# ------------------------------------------------------------------

def detect_release(ball_pos, pose_points, prev_ball_in_hands):
    """Detect a shot release by ball–hand separation.

    Returns (ball_in_hands: bool, is_release: bool).

    A release is when the ball was near the shooter's wrist last frame
    but is no longer near it this frame, AND the ball is moving upward.
    """
    if not ball_pos or not pose_points:
        return prev_ball_in_hands, False

    bx, by = ball_pos[-1][0]
    ball_w, ball_h = ball_pos[-1][2], ball_pos[-1][3]
    ball_radius = (ball_w + ball_h) / 4.0

    ball_in_hands = False
    for wrist_key in ('RIGHT_WRIST', 'LEFT_WRIST'):
        if wrist_key in pose_points:
            wx, wy, vis = pose_points[wrist_key]
            if vis > 0.5:
                dist = math.sqrt((bx - wx) ** 2 + (by - wy) ** 2)
                if dist < ball_radius * 2.5:
                    ball_in_hands = True
                    break

    # Release = was in hands, now isn't, and ball is heading upward
    is_release = False
    if prev_ball_in_hands and not ball_in_hands:
        _, vy = _ball_velocity(ball_pos, n=3)
        if vy < 0:  # negative = upward in image coords
            is_release = True

    return ball_in_hands, is_release


# ------------------------------------------------------------------
# Score detection (net-zone lingering + pass-through fallback)
# ------------------------------------------------------------------

def score(ball_pos, hoop_pos, min_frames_in_net=3):
    """Detect a made basket.

    Primary method — *net-zone lingering*:  a made shot results in the
    ball spending several frames directly below the rim while remaining
    horizontally centred (it is inside / passing through the net).  A miss
    bounces away horizontally and will not linger.

    Fallback — *pass-through*:  if the net-zone check is inconclusive,
    look for two chronologically ordered positions where the ball is above
    the rim centre first and below it second, both horizontally within the
    rim width.
    """
    if len(ball_pos) == 0 or len(hoop_pos) == 0:
        return False

    rim_cx = hoop_pos[-1][0][0]
    rim_bottom_y = hoop_pos[-1][0][1] + 0.5 * hoop_pos[-1][3]
    rim_half_w = 0.4 * hoop_pos[-1][2]
    net_depth = 1.5 * hoop_pos[-1][3]

    recent_points = ball_pos[-30:] if len(ball_pos) >= 30 else ball_pos

    # --- Primary: net-zone lingering ---
    net_frames = 0
    for center, frame_num, _, _, _ in recent_points:
        bx, by = center
        if (abs(bx - rim_cx) < rim_half_w
                and rim_bottom_y < by < rim_bottom_y + net_depth):
            net_frames += 1

    if net_frames >= min_frames_in_net:
        return True

    # --- Fallback: pass-through detection ---
    rim_center_y = hoop_pos[-1][0][1] - 0.5 * hoop_pos[-1][3]
    y_tolerance = 0.6 * hoop_pos[-1][3]

    rim_points = []
    for center, frame_num, _, _, _ in recent_points:
        x, y = center
        if abs(x - rim_cx) < rim_half_w:
            if abs(y - rim_center_y) < y_tolerance:
                rim_points.append((x, y, frame_num))

    if len(rim_points) >= 2:
        for i in range(len(rim_points)):
            for j in range(i + 1, len(rim_points)):
                if rim_points[i][2] < rim_points[j][2]:
                    if rim_points[i][1] < rim_center_y < rim_points[j][1]:
                        return True

    return False


def detect_down(ball_pos, hoop_pos):
    """Ball is below the bottom of the hoop **and** is moving downward.

    Adding a velocity check prevents a single noisy detection below the
    rim from falsely triggering 'down'.
    """
    if len(ball_pos) < 2 or len(hoop_pos) == 0:
        return False

    y_threshold = hoop_pos[-1][0][1] + 0.5 * hoop_pos[-1][3]

    # Must be below the hoop bottom
    if ball_pos[-1][0][1] <= y_threshold:
        return False

    # Must have downward velocity (positive dy in image coords)
    _, vy = _ball_velocity(ball_pos, n=3)
    if vy <= 0:
        return False

    return True


def detect_up(ball_pos, hoop_pos, min_consecutive=2):
    """Ball is in the 'up' region above the hoop.

    Improvements over the original:
      - Requires *min_consecutive* recent ball positions inside the region
        to filter out single-frame noise.
      - Checks that the ball is moving upward (negative dy) so a
        stationary/descending detection is not treated as a release.
    """
    if len(ball_pos) < min_consecutive or len(hoop_pos) == 0:
        return False

    # Define the 'up' zone
    hoop_cx = hoop_pos[-1][0][0]
    hoop_cy = hoop_pos[-1][0][1]
    hoop_w = hoop_pos[-1][2]
    hoop_h = hoop_pos[-1][3]

    x1 = hoop_cx - 2 * hoop_w
    x2 = hoop_cx + 2 * hoop_w
    y1 = hoop_cy - 2.5 * hoop_h
    y2 = hoop_cy - 0.3 * hoop_h

    # Check that the last `min_consecutive` positions are all inside the zone
    for bp in ball_pos[-min_consecutive:]:
        bx, by = bp[0]
        if not (x1 < bx < x2 and y1 < by < y2):
            return False

    # Ball should be heading upward (negative y velocity)
    _, vy = _ball_velocity(ball_pos, n=3)
    if vy >= 0:
        return False

    return True


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


# ------------------------------------------------------------------
# Hoop-relative normalisation
# ------------------------------------------------------------------

def normalise_to_hoop(point, hoop_center, hoop_width, hoop_height):
    """Convert pixel coordinates to hoop-relative units.

    (0, 0) = hoop centre.  1 unit horizontally = hoop width,
    1 unit vertically = hoop height.
    """
    dx = (point[0] - hoop_center[0]) / (hoop_width + 1e-6)
    dy = (point[1] - hoop_center[1]) / (hoop_height + 1e-6)
    return (dx, dy)


def normalise_metrics(metrics, hoop_pos):
    """Add hoop-relative normalised versions of distance-based metrics.

    Originals are kept unchanged (so the existing XGBoost model still
    works on raw pixel features).  Normalised copies are stored with a
    '_norm' suffix.
    """
    if not hoop_pos or not metrics:
        return metrics

    scale = (hoop_pos[-1][2] + hoop_pos[-1][3]) / 2.0
    if scale < 1:
        return metrics

    distance_keys = [
        'ball_hoop_distance_x', 'ball_hoop_distance_y',
        'ball_hoop_total_distance', 'shoulder_tilt', 'hip_tilt',
        'knee_spread', 'release_height', 'release_x_position',
        'body_vertical_alignment', 'right_wrist_extension',
        'jump_height',
    ]

    normalised = dict(metrics)
    for k in distance_keys:
        if k in normalised and normalised[k] is not None:
            normalised[f'{k}_norm'] = normalised[k] / scale

    return normalised


# ------------------------------------------------------------------
# Ball position cleaning
# ------------------------------------------------------------------

def clean_ball_pos(ball_pos, frame_count):
    """Removes inaccurate data points.

    Improvements:
      - Original only compared the last two points, so an outlier
        sandwiched between valid detections would survive.
      - Now also performs a *look-back* check: if we have >=3 recent
        points and the middle one is far from both neighbours, remove it.
      - Aspect-ratio filter remains unchanged.
    """
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
        max_dist = 4 * math.sqrt(w1 ** 2 + h1 ** 2)

        # Ball should not move 4x its diameter within 5 frames
        if (dist > max_dist) and (f_dif < 5):
            ball_pos.pop()

        # Ball should be relatively square
        elif (w2 * 1.4 < h2) or (h2 * 1.4 < w2):
            ball_pos.pop()

    # Look-back outlier removal: check 2nd-to-last point against its
    # neighbours.  If it is far from *both* the point before it and the
    # point after it (the current last), it is likely noise.
    if len(ball_pos) >= 3:
        for idx in (-2,):   # check the second-to-last point
            p_prev = ball_pos[idx - 1]
            p_curr = ball_pos[idx]
            p_next = ball_pos[idx + 1] if idx + 1 < 0 else ball_pos[-1]
            d_prev = math.sqrt((p_curr[0][0] - p_prev[0][0]) ** 2 +
                               (p_curr[0][1] - p_prev[0][1]) ** 2)
            d_next = math.sqrt((p_curr[0][0] - p_next[0][0]) ** 2 +
                               (p_curr[0][1] - p_next[0][1]) ** 2)
            ref_size = math.sqrt(p_curr[2] ** 2 + p_curr[3] ** 2)
            if d_prev > 4 * ref_size and d_next > 4 * ref_size:
                ball_pos.pop(idx)
                break

    # Keep longer history for better scoring
    if len(ball_pos) > 0:
        if frame_count - ball_pos[0][1] > 50:
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
