# Import necessary libraries
import cv2 
import numpy as np
import math

# Function to preprocess the image to detect yellow and white lanes
def preprocessing(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    gblur = cv2.GaussianBlur(gray,(5,5),0)
    white_mask = cv2.threshold(gblur,200,255,cv2.THRESH_BINARY)[1]
    lower_yellow = np.array([0,100,100])
    upper_yellow = np.array([210,255,255])
    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    mask = cv2.bitwise_or(white_mask, yellow_mask)
    return mask

# Function that defines the polygon region of interest
def regionOfInterest(img, polygon):
    mask = np.zeros_like(img)
    x1, y1 = polygon[0]
    x2, y2 = polygon[1]
    x3, y3 = polygon[2]
    x4, y4 = polygon[3]
    pts = np.array([[x1,y1], [x2,y2], [x3,y3], [x4,y4]], np.int32)
    cv2.fillPoly(mask, [pts], 255)
    masked_img = cv2.bitwise_and(img, mask)
    return masked_img

# Function to warp the image for curved lane detection
def warp(img, source_points, destination_points, destn_size):
    matrix = cv2.getPerspectiveTransform(source_points, destination_points)
    warped_img = cv2.warpPerspective(img, matrix, destn_size)
    return warped_img

# Function that fits curves to the lanes
def fitCurve(img):
    histogram = np.sum(img[img.shape[0]//2:,:], axis=0)
    midpoint = int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint
    nwindows = 50
    margin = 100
    minpix = 50
    window_height = int(img.shape[0]/nwindows)
    y, x = img.nonzero()
    leftx_current = leftx_base
    rightx_current = rightx_base
    left_lane_indices = []
    right_lane_indices = []
    
    for window in range(nwindows):
        win_y_low = img.shape[0] - (window+1)*window_height
        win_y_high = img.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin

        good_left_indices = ((y >= win_y_low) & (y < win_y_high) & (x >= win_xleft_low) & (x < win_xleft_high)).nonzero()[0]
        good_right_indices  = ((y >= win_y_low) & (y < win_y_high) & (x >= win_xright_low) & (x < win_xright_high)).nonzero()[0]
        left_lane_indices.append(good_left_indices)
        right_lane_indices.append(good_right_indices)
        if len(good_left_indices) > minpix:
            leftx_current = int(np.mean(x[good_left_indices]))
        if len(good_right_indices) > minpix:
            rightx_current = int(np.mean(x[good_right_indices]))
        
    left_lane_indices = np.concatenate(left_lane_indices)
    right_lane_indices = np.concatenate(right_lane_indices)
    leftx = x[left_lane_indices]
    lefty = y[left_lane_indices]
    rightx = x[right_lane_indices]
    righty = y[right_lane_indices]
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    return left_fit, right_fit

# Function that determines whether the lane is straight or curved
def classifyLane(left_fit, right_fit):
    left_curve_radius = (1 + (2*left_fit[0])**2)**1.5 / np.abs(2*left_fit[0])
    right_curve_radius = (1 + (2*right_fit[0])**2)**1.5 / np.abs(2*right_fit[0])
    avg_curve_radius = (left_curve_radius + right_curve_radius) / 2

    # Classification based on curvature radius
    if avg_curve_radius > 1000:  # Adjust this threshold as necessary
        return 'straight'
    else:
        return 'curved'

# Function to draw straight lines on the image
def drawStraightLines(img, masked_img):
    lines = cv2.HoughLinesP(masked_img, 1, np.pi/180, 50, minLineLength=100, maxLineGap=50)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 5)
    return img

# Main video processing loop for real-time display
video = cv2.VideoCapture("data/combined_lane_detection.mp4")

while True:
    isTrue, frame = video.read()
    if not isTrue:
        break
    
    processed_img = preprocessing(frame)
    height, width = processed_img.shape
    polygon = [(int(width*0.15), height), (int(width*0.45), int(height*0.62)), (int(width*0.55), int(height*0.62)), (int(width*0.95), height)]
    masked_img = regionOfInterest(processed_img, polygon)

    # Fit curves to lanes
    left_fit, right_fit = fitCurve(masked_img)

    # Classify lane type (straight or curved)
    lane_type = classifyLane(left_fit, right_fit)

    if lane_type == 'straight':
        result_frame = drawStraightLines(frame, masked_img)
    else:
        # If curved, apply curve detection and overlay
        source_points = np.float32([[int(width*0.49), int(height*0.62)], [int(width*0.58), int(height*0.62)], [int(width*0.15), int(height)], [int(0.95*width), int(height)]])
        destination_points = np.float32([[0,0], [400,0], [0, 960], [400, 960]])
        warped_img = warp(masked_img, source_points, destination_points, (400, 960))
        left_fit, right_fit = fitCurve(warped_img)
        result_frame = frame  # Add visualization for curved lines here

    # Display the result frame in a window
    cv2.imshow('Lane Detection', result_frame)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
