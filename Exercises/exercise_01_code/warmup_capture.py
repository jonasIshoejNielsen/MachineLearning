# Course: Introduction to Image Analysis and Machine Learning (ITU)
# Version: 2020.1

import cv2
from imutils.video import FPS

# Create a capture video object.
capture = cv2.VideoCapture(0)

# Define the video resolution.
capture.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Create an OpenCV window.
cv2.namedWindow("Image", cv2.WINDOW_AUTOSIZE)

# Calculate the number of frames per second.
fps = FPS().start()

# This repetion will run while there is a new frame in the video file or
# while the user do not press the "q" (quit) keyboard button.
while True:
    # Capture frame-by-frame.
    retval, frames = capture.read()

    # Update the FPS counter.
    fps.update()

    # Check if there is a valid frame.
    if not retval:
        break

    # Display the resulting frame.
    cv2.imshow("Image", frames)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Stop the timer and display FPS information.
fps.stop()
print("Elasped time: {:.2f}".format(fps.elapsed()))
print("Camera framerate: {:.2f}".format(fps.fps()))

# When everything done, release the capture object.
capture.release()
cv2.destroyAllWindows()
