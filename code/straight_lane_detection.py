#Import necessary libraries
import cv2 
import numpy as np
import math
import copy

#Function to perform image preprocessing
def preprocessing(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gblur = cv2.GaussianBlur(gray,(5, 5),0)
    thresh = cv2.threshold(gblur,150,255,cv2.THRESH_BINARY)[1]
    return thresh

#Function that defines the polygonal region of interest
def regionOfInterest(img, polygon):
    mask = np.zeros_like(img)
    x1, y1 = polygon[0]
    x2, y2 = polygon[1]
    x3, y3 = polygon[2]
    x4, y4 = polygon[3]
    m1 = (y2-y1)/(x2-x1)
    m2 = (y3-y2)/(x3-x2)
    m3 = (y4-y3)/(x4-x3)
    m4 = (y4-y1)/(x4-x1)
    b1 = y1 - m1*x1
    b2 = y2 - m2*x2
    b3 = y3 - m3*x3
    b4 = y4 - m4*x4
    
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if i>=m1*j+b1 and i>=m2*j+b2 and i>=m3*j+b3 and i<=m4*j+b4:
                mask[i][j] = 1

    masked_img = np.multiply(mask, img)
    return masked_img

#Function to find slope and y-intercept of a line given two co-ordinates of a point
def slopeIntercept(line):
    m = (line[1][1]-line[0][1])/(line[1][0]-line[0][0])
    b = line[1][1] - m*line[1][0]
    return m, b

#Function to remove multiple lines detected on the same lane side
def removeCloseLines(linelist, m):
    linelist_copy = copy.deepcopy(linelist)
    
    for line in linelist:
        m1, _ = slopeIntercept(line)
        if abs(m-m1)<=0.5:
            linelist_copy.remove(line)
            
    return linelist_copy

#Function that draws lines on the image
def lineDetection(img, masked_img, solid_line_previous_left, solid_line_previous_right):
    img_copy = copy.deepcopy(img)
    height, width = masked_img.shape
    #paramters = (image, rho, theta, threshold[, lines[, minLineLength[, maxLineGap]]]) 
    linesP = cv2.HoughLinesP(masked_img, 1, np.pi/180, 50, None, 30, 20)
    linelist = linesP.tolist() if linesP is not None else []
    linelist = [tuple((line[0][:2], line[0][2:])) for line in linelist]

    if not linelist:  # If no lines detected, return previous lines
        return img_copy, solid_line_previous_left, solid_line_previous_right

    line_length = [math.dist(line[0], line[1]) for line in linelist]

    # Sort the lines based on length, in descending order
    sorted_lines = [line for _, line in sorted(zip(line_length, linelist), reverse=True)]

    try:
        solid_line_left = sorted_lines[0]  # Longest line (left lane boundary)
        solid_line_right = sorted_lines[1]  # Second longest line (right lane boundary)
    except IndexError:
        # If we don't detect two lines, use the previous ones
        solid_line_left = solid_line_previous_left
        solid_line_right = solid_line_previous_right

    # Draw the left solid line
    m_left, b_left = slopeIntercept(solid_line_left)
    initial_left = (int((height*0.6-b_left)/m_left), int(height*0.6))
    final_left = (int((height-b_left)/m_left), height)
    detected_line = cv2.line(img_copy, initial_left, final_left, (0,255,0), 5)

    # Draw the right solid line
    m_right, b_right = slopeIntercept(solid_line_right)
    initial_right = (int((height*0.6-b_right)/m_right), int(height*0.6))
    final_right = (int((height-b_right)/m_right), height)
    detected_line = cv2.line(detected_line, initial_right, final_right, (0,255,0), 5)

    return detected_line, solid_line_left, solid_line_right


video = cv2.VideoCapture("data/straight_lane_detection.mp4")
out = cv2.VideoWriter('results/straight_lane_detection.avi',cv2.VideoWriter_fourcc(*'XVID'), 25, (960,540))
solid_line_previous_left = None
solid_line_previous_right = None
print("Generating video output...\n")

while True:
    isTrue, img = video.read()
    if isTrue == False:
        break
    processed_img = preprocessing(img)
    height, width = processed_img.shape
    polygon = [(int(width*0.1), height), (int(width*0.45), int(height*0.6)), (int(width*0.55), int(height*0.6)), (int(0.95*width), height)]
    masked_img = regionOfInterest(processed_img, polygon)
    detected_lines, solid_line_left, solid_line_right = lineDetection(img, masked_img, solid_line_previous_left, solid_line_previous_right)
    solid_line_previous_left = solid_line_left
    solid_line_previous_right = solid_line_right
    out.write(detected_lines)

out.release()
print("Video output generated.\n")