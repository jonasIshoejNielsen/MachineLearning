import cv2
import numpy as np
from enum import Enum


class Pattern(Enum):
    """Define the pattern type used to calibrate the camera."""
    CHESSBOARD = 1
    CIRCLES = 2

def createsPatternVectors(square_size=1.0, pattern_type=Pattern.CHESSBOARD):
    """
    Creates a standard vector of calibration pattern points in the calibration
    pattern coordinate space.
    """
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

#<!--------------------------------------------------------------------------->
#<!--                            YOUR CODE HERE                             -->
#<!--------------------------------------------------------------------------->


#<!--------------------------------------------------------------------------->
#<!--                                                                       -->
#<!--------------------------------------------------------------------------->
