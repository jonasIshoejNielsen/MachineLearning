import cv2
import numpy as np
import math

import time

kernel = np.ones((5,5))

def dist(a, b):
    return np.linalg.norm(a-b)


def contour_center(contour):
    moments = cv2.moments(contour)
    if moments['m00'] == 0:
        cx, cy = 0, 0
    else:
        cx = moments['m10']/moments['m00']
        cy = moments['m01']/moments['m00']
    return np.array([cx, cy])


def find_pupil(img, debug=True, threshold = 55):
    """Detects and returns a single pupil candidate for a given image.
    You can use the debug flag for showing threshold images, print messages, etc.

    Returns: A pupil candidate in OpenCV ellipse format.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _,binaryGray = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)
    binaryGray = cv2.morphologyEx(binaryGray, cv2.MORPH_OPEN, kernel)
    
    
    (countours, hierarchy) = cv2.findContours(binaryGray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE )
    largest, maxSize = None, None
    for c in countours:
        size = cv2.contourArea(c)
        centerC = getCenterOfMass(c)
        perimeter = cv2.arcLength(c,True)
        division = (2*math.sqrt(math.pi*size))
        if division != 0:
            circularityFeature = perimeter / division
            if circularityFeature <2 and len(c)>=5 and (largest is None or size > maxSize) :
                largest = c
                maxSize = size
    if maxSize is None:
        return find_pupil(img, debug=True, threshold = threshold*1.2)
    else :  
        if debug:
            cv2.imshow("threshold", binaryGray)
            colored = cv2.cvtColor(binaryGray, cv2.COLOR_GRAY2BGR)
            cv2.drawContours(colored, [largest], 0, (0,255,0), 2)
            cv2.imshow("drawContours", colored)
        
        ((x,y), (a,b), angle) = cv2.fitEllipse(largest)
        return ((x,y), (a,b), angle)

def getCenter(contour):
    x,y,w,h = cv2.boundingRect(contour)
    return [x+w/2, y+h/2]

def getCenterOfMass(contour):
    sumx, sumy = 0,0
    for c in contour:
        sumx += c[0][0]
        sumy += c[0][1]
    return [sumx/len(contour), sumy/len(contour)]

def sortMoveOn (lst, reverse, take):
    lst = lst.copy()
    lst.sort(reverse=reverse, key = (lambda val : val[0]) )
    lst = lst[0  :min(len(lst), take)]
    return [c[1 : len(c)] for c in lst]
def euclidean_distance(p1, p2):
    return math.sqrt( ((p1[0]-p2[0])*(p1[0]-p2[0]))+((p1[1]-p2[1])*(p1[1]-p2[1])) )

def find_glints(img, center, debug=True, threshold=220):
    """Detects and returns up to four glint candidates for a given image.
    You can use the debug flag for showing threshold images, print messages, etc.

    Returns: Detected glint positions.
    """
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _,binaryGray = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)
    binaryGray = cv2.morphologyEx(binaryGray, cv2.MORPH_OPEN, kernel)
    (countours, hierarchy) = cv2.findContours(binaryGray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE )
    lst = []
    for c in countours:
        #features
        if len(c)>4:
            area = cv2.contourArea(c)
            centerC = getCenterOfMass(c)
            perimeter = cv2.arcLength(c,True)
            division = (2*math.sqrt(math.pi*area))
            if division != 0:
                circularityFeature = perimeter / division
                distance = euclidean_distance(center, centerC)
                if circularityFeature < 1.15 and distance <38 and area >8:
                    lst.append((circularityFeature, distance, area, c))

    lst = sortMoveOn(lst, False, 6)
    lst = sortMoveOn(lst, False, 4)
    #lst = sortMoveOn(lst, True, 4)
    
    lst = [c[len(c)-1] for c in lst]
    
    #debugging
    if debug:
        colorimg = cv2.cvtColor(binaryGray, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(colorimg, lst, -1, (0,255,0), 1)
        cv2.imshow("testcolorimg", colorimg)
        
    return [getCenter(c) for c in lst]
