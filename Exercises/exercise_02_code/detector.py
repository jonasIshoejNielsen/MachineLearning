# Course: Introduction to Image Analysis and Machine Learning (ITU)
# Version: 2020.1

import argparse
import cv2
import numpy as np

# Construct the argument parse and parse the arguments.
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="Path to the (optional) video file")
args = vars(ap.parse_args())

# Load a video camera or a video file.
if not args.get("video", False):
    video = cv2.VideoCapture(0)
else:
    video = cv2.VideoCapture(args["video"])

# Grab each individual frame.
while True:
    # Grabs, decodes and returns the next video frame.
    retval, frame = video.read()

    # Check if there is a valid frame.
    if not retval:
        break

    # <Exercise 2.6>

    # Remove this line after you finish the exercise.
    original = frame.copy()

    # <Exercise 2.6 (a)>
    resultHSV = cv2.cvtColor(original, cv2.COLOR_BGR2HSV)
    #result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

    # <Exercise 2.6 (b)>
    lower_blue = np.array([95,50,50])
    upper_blue = np.array([150,255,255])
    mask = cv2.inRange(resultHSV, lower_blue, upper_blue)
    
    
    # <Exercise 2.6 (c)>
    result = cv2.bitwise_and(original,original, mask= mask)

    # Show the processed images.
    cv2.imshow("original", original)
    cv2.imshow("Result", result)
    cv2.imshow("mask", mask)

    # Get the keyboard event.
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Closes video file or capturing device.
video.release()

# Destroys all of the HighGUI windows.
cv2.destroyAllWindows()
