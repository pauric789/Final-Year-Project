# import libraries
import math
import numpy as np
import torch

# calculate angle between three points 
def calculate_angle(p1, p2, p3):
    # v1 is the vector from p2 to p1
    v1 = p1 - p2
    # V2 is the vector from p2 to p3
    v2 = p3 - p2
    # Calculate the dot of v1 and v2
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
    # return the angle in degrees
    return np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))

# caluclate the distance using the Euclidean distance formula
def calculate_distance(p1, p2):
    # return the distance between p1 and p2
    return np.linalg.norm(p1 - p2)

# get the GPU if available, otherwise return CPU
def get_device():
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    return device

# the score function 
def score(ball_pos, hoop_pos, min_frames_in_net=3):
    # if no ball or hoop positions return False
    if len(ball_pos) == 0 or len(hoop_pos) == 0:
        return False
    # calculate the center of the rim 
    rim_cx = hoop_pos[-1][0][0]
    # calculate the bottom of the net
    rim_bottom_y = hoop_pos[-1][0][1] + 0.5 * hoop_pos[-1][3]#
    # calculate the half width and depth of the net
    rim_half_w = 0.4 * hoop_pos[-1][2]
    # calculate the depth of the net
    net_depth = 1.5 * hoop_pos[-1][3]
    # get the last 30 points of the ball position 
    recent_points = ball_pos[-30:] if len(ball_pos) >= 30 else ball_pos
    # initialize the number of frames the ball is in the net
    net_frames = 0
    # loop over the recent points 
    for center, frame_num, _, _, _ in recent_points:
        # get the x and y coordinates of the center of the ball
        bx, by = center
        # check if the ball is within the horizontal bounds 
        if (abs(bx - rim_cx) < rim_half_w
                and rim_bottom_y < by < rim_bottom_y + net_depth):
            # increment the number of frames in the net
            net_frames += 1
    # if the number of frames in the net is greater than or equal to the minimum required return score 
    if net_frames >= min_frames_in_net:
        return True
    # calculate under rim points 
    rim_center_y = hoop_pos[-1][0][1] - 0.5 * hoop_pos[-1][3]
    # get the y tolerance for detecting rim points
    y_tolerance = 0.6 * hoop_pos[-1][3]

    # store the points that are near the rim 
    rim_points = []
    # loop over the recent points
    for center, frame_num, _, _, _ in recent_points:
        # get the center of the ball
        x, y = center
        # check if the ball is near the rim
        if abs(x - rim_cx) < rim_half_w:
            # check if the ball is above or below the rim
            if abs(y - rim_center_y) < y_tolerance:
                # add the point to the list of rim points
                rim_points.append((x, y, frame_num))
    # if there are at least 2 rim points
    if len(rim_points) >= 2:
        # loop over the rim points
        for i in range(len(rim_points)):
            # loop over the remaining rim points
            for j in range(i + 1, len(rim_points)):
                # check if the second point is above the first
                if rim_points[i][2] < rim_points[j][2]:
                    # check if the rim center is between the two points
                    if rim_points[i][1] < rim_center_y < rim_points[j][1]:
                        # return score
                        return True
    # return false if no scoring conditions are met
    return False

# detect if the ball is in the "down" state
def detect_down(ball_pos, hoop_pos):
    # if there are not enough ball positions or no hoop positions return False
    if len(ball_pos) < 2 or len(hoop_pos) == 0:
        return False
    # calculate the y threshold for being below the hoop
    y_threshold = hoop_pos[-1][0][1] + 0.5 * hoop_pos[-1][3]
    # if ball is not below the y threshold yet return False
    if ball_pos[-1][0][1] <= y_threshold:
        return False
    # get the last 3 points of the ball position
    pts = ball_pos[-3:] if len(ball_pos) >= 3 else ball_pos
    # intialize the vertical velocity sum
    dy_sum = 0
    # intialize the sample count
    count = 0
    # loop over the points
    for i in range(1, len(pts)):
        # get the frame difference
        frame_diff = pts[i][1] - pts[i - 1][1]
        # if frame difference is 0 skip
        if frame_diff == 0:
            continue
        # calculate the the dy sum
        dy_sum += (pts[i][0][1] - pts[i - 1][0][1]) / frame_diff
        # increment the count
        count += 1
    # calculate the average vertical velocity
    vy = (dy_sum / count) if count else 0
    # if the average vertical velocity is less than or equal to 0 return False
    if vy <= 0:
        return False
    # if conditions are met return True
    return True

# detect if ball is in the "up" state
def detect_up(ball_pos, hoop_pos, min_consecutive=2):
    # if there isn't enough ball positions or no hoop positions return False
    if len(ball_pos) < min_consecutive or len(hoop_pos) == 0:
        return False
    # get the center, width, and height of the hoop
    hoop_cx = hoop_pos[-1][0][0]
    hoop_cy = hoop_pos[-1][0][1]
    hoop_w = hoop_pos[-1][2]
    hoop_h = hoop_pos[-1][3]
    # define the region above the hoop 
    x1 = hoop_cx - 2 * hoop_w
    x2 = hoop_cx + 2 * hoop_w
    y1 = hoop_cy - 2.5 * hoop_h
    y2 = hoop_cy - 0.3 * hoop_h
    # loop over the last few ball positions
    for bp in ball_pos[-min_consecutive:]:
        # get the x and y coordinates of the ball
        bx, by = bp[0]
        # check if the ball is within the defined region
        if not (x1 < bx < x2 and y1 < by < y2):
            # return False if the ball is not in the region
            return False
    # get the last 3 points of the ball position
    pts = ball_pos[-3:] if len(ball_pos) >= 3 else ball_pos
    # intialize the vertical velocity sum
    dy_sum = 0
    # intialize the sample count
    count = 0
    # loop over the points
    for i in range(1, len(pts)):
        # get the frame difference
        frame_diff = pts[i][1] - pts[i - 1][1]
        # if frame difference is 0 skip
        if frame_diff == 0:
            continue
        # calculate the dy sum
        dy_sum += (pts[i][0][1] - pts[i - 1][0][1]) / frame_diff
        # increment the count
        count += 1
    # calculate the average vertical velocity
    vy = (dy_sum / count) if count else 0
    # if the average vertical velocity is greater than or equal to 0 return False
    if vy >= 0:
        return False
    # if conditions are met return True
    return True

# check if the center of the ball is in the hoop region
def in_hoop_region(center, hoop_pos):
    # if there are no hoop positions return False
    if len(hoop_pos) < 1:
        return False
    # get the x and y coordinates of the center of the ball
    x = center[0]
    y = center[1]
    # define the region around the hoop
    x1 = hoop_pos[-1][0][0] - 1 * hoop_pos[-1][2]
    x2 = hoop_pos[-1][0][0] + 1 * hoop_pos[-1][2]
    y1 = hoop_pos[-1][0][1] - 1 * hoop_pos[-1][3]
    y2 = hoop_pos[-1][0][1] + 0.5 * hoop_pos[-1][3]
    # check if the center of the ball is in the region
    if x1 < x < x2 and y1 < y < y2:
        # return True if the center of the ball is in the region
        return True
    # if the center of the ball is not in the region return False
    return False

# clean the ball position data by removing inaccurate data points
def clean_ball_pos(ball_pos, frame_count):
    # if there are more than 1 ball positions 
    if len(ball_pos) > 1:
        # get the last positions and dimensions of the ball
        w1 = ball_pos[-2][2]
        h1 = ball_pos[-2][3]
        w2 = ball_pos[-1][2]
        h2 = ball_pos[-1][3]
        # get the x and y coordinates of the last two ball positions
        x1 = ball_pos[-2][0][0]
        y1 = ball_pos[-2][0][1]
        x2 = ball_pos[-1][0][0]
        y2 = ball_pos[-1][0][1]

        # calculate the distance between the last two ball positions
        f1 = ball_pos[-2][1]
        f2 = ball_pos[-1][1]
        f_dif = f2 - f1
        # calculate the distance between the last two ball positions
        dist = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        max_dist = 4 * math.sqrt(w1 ** 2 + h1 ** 2)
        # if the distance between the last two ball positions is greater than the maximum allowed distance 
        # and the frame difference is less than 5, remove the last ball position
        if (dist > max_dist) and (f_dif < 5):   
            ball_pos.pop()
        # else if the width and height of the last ball position are not in a reasonable ratio remove the last ball position
        elif (w2 * 1.4 < h2) or (h2 * 1.4 < w2):
            ball_pos.pop()
    # if the length of ball positions is greater than or equal to 3
    if len(ball_pos) >= 3:
        # the idx of the second to last ball position
        idx = -2
        # get the previous, current, and next ball positions
        p_prev = ball_pos[idx - 1]
        p_curr = ball_pos[idx]
        p_next = ball_pos[idx + 1] if idx + 1 < 0 else ball_pos[-1]
        # calculate the distance between the previous, current, and next ball positions
        d_prev = math.sqrt((p_curr[0][0] - p_prev[0][0]) ** 2 +
                           (p_curr[0][1] - p_prev[0][1]) ** 2)
        d_next = math.sqrt((p_curr[0][0] - p_next[0][0]) ** 2 +
                           (p_curr[0][1] - p_next[0][1]) ** 2)
        ref_size = math.sqrt(p_curr[2] ** 2 + p_curr[3] ** 2)
        # if the previous and next ball positions are both far from the current ball position remove the current ball position
        if d_prev > 4 * ref_size and d_next > 4 * ref_size:
            ball_pos.pop(idx)
    # if the length of the ball positions is greater than 0
    if len(ball_pos) > 0:
        # if the current frame count minus the oldest frame numbeer is greater than 50
        if frame_count - ball_pos[0][1] > 50:
            # remove the first ball position
            ball_pos.pop(0)
    # return the cleaned ball positions
    return ball_pos

# clean the hoop positions 
def clean_hoop_pos(hoop_pos):
    # if there are more than 1 hoop positions
    if len(hoop_pos) > 1:
        # get the x and y coordinates of the last two hoop positions
        x1 = hoop_pos[-2][0][0]
        y1 = hoop_pos[-2][0][1]
        x2 = hoop_pos[-1][0][0]
        y2 = hoop_pos[-1][0][1]
        # get the width and height of the last two hoop positions
        w1 = hoop_pos[-2][2]
        h1 = hoop_pos[-2][3]
        w2 = hoop_pos[-1][2]
        h2 = hoop_pos[-1][3]
        # calculate the distance between the last two hoop positions
        f1 = hoop_pos[-2][1]
        f2 = hoop_pos[-1][1]
        f_dif = f2 - f1
        # calculate the distance between the last two hoop positions
        dist = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)   
        max_dist = 0.5 * math.sqrt(w1 ** 2 + h1 ** 2)
        # if the distance between the last two hoop positions is greater than the maximum allowed distance 
        if dist > max_dist and f_dif < 5:
            hoop_pos.pop()
        # else if the width and height of the last hoop position are not in a reasonable ratio remove the last hoop position
        if (w2 * 1.3 < h2) or (h2 * 1.3 < w2):
            hoop_pos.pop()
    # if the length of the hoop positions is greater than 25
    if len(hoop_pos) > 25:
        # remove the first hoop position
        hoop_pos.pop(0)
    # return the cleaned hoop positions
    return hoop_pos
