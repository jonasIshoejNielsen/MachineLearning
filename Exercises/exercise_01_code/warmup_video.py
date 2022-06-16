# Course: Introduction to Image Analysis and Machine Learning (ITU)
# Version: 2020.1

import cv2
import time

# Defines the filepath.
filepath = "./inputs/PPAP.mp4"

# Create a capture video object.
capture = cv2.VideoCapture(filepath)

# Get the video frame rate.
fps = int(round(capture.get(cv2.CAP_PROP_FPS)))

# Check if the fps variable has a correct value.
fps = fps if fps > 0 else 30

# Create an OpenCV window.
cv2.namedWindow("Video", cv2.WINDOW_AUTOSIZE)

# This repetion will run while there is a new frame in the video file or
# while the user do not press the "q" (quit) keyboard button.
while True:
    # Capture frame-by-frame.
    retval, frame = capture.read()

    # Check if there is a valid frame.
    if not retval:
        break

    # Resize the frame.
    frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)

    # Display the resulting frame.
    cv2.imshow("Video", frame)
    if cv2.waitKey(fps) & 0xFF == ord("q"):
        break

# When everything done, release the capture and record objects.
capture.release()
cv2.destroyAllWindows()
