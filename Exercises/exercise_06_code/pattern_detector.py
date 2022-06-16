import cv2
import os
from enum import Enum
from imutils.video import FPS


def main():
    # Create a capture video object.
    capture = cv2.VideoCapture(0)

    # Create an OpenCV window.
    cv2.namedWindow("Pattern", cv2.WINDOW_AUTOSIZE)

    # Calculate the number of frames per second.
    fps = FPS().start()

    # This repetition will run while there is a new frame in the video file or
    # while the user do not press the "q" (quit) keyboard button.
    while True:
        # Capture frame-by-frame.
        retval, image = capture.read()

        # Update the FPS counter.
        fps.update()

        # Check if there is a valid frame.
        if not retval:
            break

        # Create a copy of the input image.
        pattern = image.copy()

        # Try to detect a pattern in the current image.
        detected = pattern_detector(pattern)

        # Write a text in the image.
        if detected:
            draw_message(
                pattern, "Pattern: Detected!\nPress (s) to save the pattern image.")
        else:
            draw_message(pattern, "Pattern: No detected.")

        # Display the resulting frame.
        cv2.imshow("Pattern", pattern)
        key = cv2.waitKey(1)

        if key == ord("q"):
            break
        elif key == ord("s") and detected:
            save_pattern_image(image)

    # Stop the timer and display FPS information.
    fps.stop()
    print("Elasped time: {:.2f}".format(fps.elapsed()))
    print("Camera framerate: {:.2f}".format(fps.fps()))

    # When everything done, release the capture object.
    capture.release()
    cv2.destroyAllWindows()


def draw_message(image, text):
    y0, dy = 20, 15
    for i, line in enumerate(text.split("\n")):
        y = y0 + i * dy
        cv2.putText(image, line, (10, y),
                    cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255))


def save_pattern_image(image):
    """
    This function will save the detect pattern (chessboard or circles) in a
    PNG file in the outputs folder.
    """
    # Folder where the image will be saved.
    folder = "./outputs/"

    # Current image ID.
    index = 0
    while True:
        # Create the image filename.
        filename = "%sPattern_%02d.png" % (folder, index + 1)

        # Check if it is available in the folder and avoid to overwrite it.
        if not os.path.isfile(filename):

            # Save the image as grayscale.
            grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            cv2.imwrite(filename, grayscale)

            print("Saved: %s" % filename)
            break

        # Try to use a new image ID.
        index += 1


def pattern_detector(image):
    """
    This function try to detect the calibration chessboard in the input image.
    """
    # Convert the input image to grayscale color space.
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Finds the positions of internal corners of the chessboard.
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_FAST_CHECK
    retval, corners = cv2.findChessboardCorners(grayscale, (9, 6), flags=flags)

    if retval:
        # Refines the corner locations.
        criteria = (cv2.TERM_CRITERIA_EPS +
                    cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)
        corners = cv2.cornerSubPix(
            grayscale, corners, (11, 11), (-1, -1), criteria)
        cv2.drawChessboardCorners(image, (9, 6), corners, True)

    return retval


if __name__ == '__main__':
    main()
