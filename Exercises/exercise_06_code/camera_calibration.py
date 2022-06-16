
import cv2
import numpy as np
import os
from glob import glob
from enum import Enum


def main():
    # Folder where the chessboard images are saved.
    folder = "./outputs/"

    # Pattern filename.
    pattern_files = "%s/Pattern_??.png" % folder
    filenames = glob(pattern_files)

    # Check if there are valid chessboard images.
    if len(filenames) < 5:
        print("First, grab the calibration pattern images.")
        exit(0)

    # Image resolution.
    h, w = cv2.imread(filenames[0], cv2.IMREAD_GRAYSCALE).shape[:2]

    # Create the vector of vectors of the calibration pattern points.
    imagePoints = []
    objectPoints = []

    # <Exercise 6.3 (Task 2)>
    for filename in filenames:
        res = detect_pattern_image(filename)
        if (res is None)==False:
            corners, patter_points = res
            imagePoints.append(corners)
            objectPoints.append(patter_points)
 
    # <Exercise 6.3 (Task 3)>
    # Calculate camera calibration
    rms,K, distCoeffs, rvecs, tvecs = cv2.calibrateCamera(objectPoints, imagePoints, (h,w),None, None)

    # <Exercise 6.3 (Task 4)>
    # Save the calibration files.


    # <Exercise 6.3 (Task 5)>
    # Undistort the image with the calibration.
    for filename in filenames:

        # Create the new filenames.
        filepath = filename[:-4]
        old_image = filepath + "_Chessboard.png"
        new_image = filepath + "_Undistorted.png"

        # Open the distorted image.
        image = cv2.imread(old_image)
        if image is None:
            continue

        # Image resolution.
        h, w = image.shape[:2]

        # <Exercise 6.3 (Task 5.A)>
        # Return the new camera matrix based on the free scaling parameter.


        # <Exercise 6.3 (Task 5.B)>
        # Transform an image to compensate for lens distortion.


        # <Exercise 6.3 (Task 5.C)>


    # When everything done, release the capture and record objects.
    cv2.destroyAllWindows()


def creates_pattern_vectors(square_size=1.0):
    """
    Creates a standard vector of calibration pattern points in the calibration
    pattern coordinate space.
    """
    # Check what pattern type is used to calibrate the camera.
    pattern_size = (9, 6)

    # Create the main vector.
    pattern_points = np.zeros((np.prod(pattern_size), 3), np.float32)
    pattern_points[:, :2] = np.indices(pattern_size).T.reshape(-1, 2)
    pattern_points *= square_size

    # Return the final result.
    return pattern_points, pattern_size


def detect_pattern_image(filename, square_size=1.0):
    """Try to detect the pattern corners coordinates in the input image."""
    # Open the input image as a grayscale image.
    image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None

    # Define the pattern used to calibrated the camera.
    pattern_points, pattern_size = creates_pattern_vectors(square_size)
    retval = False

    # Try to find the chessboard pattern.
    retval, corners = cv2.findChessboardCorners(image, pattern_size)
    if retval:
        term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 30, 0.1)
        cv2.cornerSubPix(image, corners, (5, 5), (-1, -1), term)

        color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        cv2.drawChessboardCorners(color, pattern_size, corners, True)

        filepath = filename[:-4]
        new_image = filepath + "_Chessboard.png"
        cv2.imwrite(new_image, color)

    # Check if there are valid data.
    if not retval:
        return None

    return (corners.reshape(-1, 2), pattern_points)


if __name__ == '__main__':
    main()
