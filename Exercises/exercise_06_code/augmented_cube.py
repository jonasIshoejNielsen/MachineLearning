import cv2
import math
import numpy as np
from enum import Enum


class Pattern(Enum):
    """Define the pattern type used to calibrate the camera."""
    CHESSBOARD = 1
    CIRCLES = 2

def draw3DAxes(image, corners, imagePoints):
    # Get the 3D coordinates of point p(0, 0) in the calibration pattern.
    corner = tuple(corners[0].ravel())

    # Draw the 3D axes.
    imagePoints = imagePoints.astype(int)
    image = cv2.line(image, corner, tuple(imagePoints[0].ravel()), (255, 0, 0), 3)
    image = cv2.line(image, corner, tuple(imagePoints[1].ravel()), (0, 255, 0), 3)
    image = cv2.line(image, corner, tuple(imagePoints[2].ravel()), (0, 0, 255), 3)

    return image

def draw3DCube(image, corners, imagePoints):
    # Convert the cube vertices to a vector.
    imagePoints = np.int32(imagePoints).reshape(-1, 2)

    # Draw the ground floor in green.
    image = cv2.drawContours(image, [imagePoints[:4]], -1, (0, 255, 0), -1)

    # Draw vertical pillars in blue.
    for i, j in zip(range(4), range(4, 8)):
        image = cv2.line(image, tuple(imagePoints[i]), tuple(imagePoints[j]), (255, 0, 0), 3)

    # Draw the top layer in red.
    image = cv2.drawContours(image, [imagePoints[4:]], -1, (0, 0, 255), 3)

    return image

def createsPatternVectors(square_size=1.0, pattern_type=Pattern.CHESSBOARD):
    """Creates a standard vectors of the calibration pattern points."""
    # Check what pattern type is used to calibrate the camera.
    pattern_size = (9, 6) if pattern_type == Pattern.CHESSBOARD else (11, 4)

    # Create the main vector.
    pattern_points = np.zeros((np.prod(pattern_size), 3), np.float32)
    pattern_points[:, :2] = np.indices(pattern_size).T.reshape(-1, 2)
    pattern_points *= square_size

    # If the circle pattern will be used, it is necessary to modify the main vector.
    if pattern_type == Pattern.CIRCLES:
        pattern_points[:, 0] = pattern_points[:, 0] * 2.0 + pattern_points[:, 1] % 2
        pattern_size = (4, 11)

    # Return the final result.
    return pattern_points, pattern_size

# Define the 3D cube vertices (size 3).
cube = np.float32([[6, 6,  0], [6, 15,  0], [15, 15,  0], [15, 6,  0],
                   [6, 6, -9], [6, 15, -9], [15, 15, -9], [15, 6, -9]])
center = np.float32([10.5, 10.5, 0])

origin = np.float32([[6, 0,  0], [0, 6,  0], [0, 0,  -6]])

#<!--------------------------------------------------------------------------->
#<!--                            YOUR CODE HERE                             -->
#<!--------------------------------------------------------------------------->

def to_homogeneous(points):
    """ Convert column vectors to row vectors"""
    if len(points.shape) == 1:
        points = points.reshape((*points.shape, 1))
    return np.vstack((points, np.ones((1, points.shape[1]))))

def to_euclidean(points, dimension=2):
    """
    function convert homogeneous vector 2 a point of dimension dimension
    """
    return points[:dimension] / points[dimension]

def transformCube (K, t, R, cube):
    p = np.dot(K,np.hstack((R, t)))
    cube2 = [np.dot(p, to_homogeneous(e)).T for e in cube]
    return cube2
    
K= np.array([[1, 0,0],[0,1,0],[0, 0,1]])
t= np.array([[0, 0,0]]).T  
#1    
R= np.array([[1, 0,0],[0,1,0],[0, 0,1]])
print(transformCube(K, t, R, cube))
#2 
R2= np.array([[0.9397, 0.3420,0],[-0.3420,0.9397,0],[0, 0,1]])
print("")
print(transformCube(K, t, R2, cube))
#4 
t2= np.array([[0, 0,2]]).T
print("")
print(transformCube(K, t2, R, cube))
# t = translation

# 5
f = 2
K= np.array([[f, 0,center[0]],[0,f,center[1]],[0, 0,1]])
print("")
print(transformCube(K, t, R, cube))
# focal lenght scales
#<!--------------------------------------------------------------------------->
#<!--                                                                       -->
#<!--------------------------------------------------------------------------->