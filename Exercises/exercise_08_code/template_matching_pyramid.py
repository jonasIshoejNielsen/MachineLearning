import numpy as np
import matplotlib.pyplot as plt
import cv2
import math
import matplotlib
matplotlib.use("tkagg")

# Selected points on image.
left_point = None
right_point = None
selected = np.zeros((1, 1, 1), np.uint8)


def show_images(**images):
    """Show multiple images using matplotlib."""
    # When a double-starred parameter is declared such as $**images$, then all
    # the keyword arguments from that point till the end are collected as a
    # dictionary called $'images'$.

    # Create a new matplotlib window.
    plt.figure()

    # Set the default colormap to gray and apply to current image if any.
    plt.gray()

    # Enumarate the ID, window name and images passed as parameter.
    for (pos, (name, image)) in enumerate(images.items()):
        # Show the image in a new subplot.
        plt.subplot(1, len(images), pos+1)
        plt.title(name)
        plt.imshow(image)

    # Show the images.
    plt.show()


def update_image():
    """Update the input image to show the selected region."""
    if (left_point is None) | (right_point is None):
        cv2.imshow("Select an area", selected)
        return

    # Draw a rectangle in the selected area.
    processed = selected.copy()
    cv2.rectangle(processed, left_point, right_point, (0, 0, 255), 2)
    cv2.imshow("Select an area", processed)


def on_mouse(event, x, y, flags, param):
    """Get the mouse events over an OpenCV windows."""
    global left_point
    global right_point

    if flags & cv2.EVENT_FLAG_LBUTTON:
        left_point = x, y

    if flags & cv2.EVENT_FLAG_RBUTTON:
        right_point = x, y

    update_image()


def select_roi(image):
    """
    This function returns the corners of the selected area as:
    [(UpLeftCorner), (DownRightCorner)]

    Use the right and left buttons of mouse and click on the image to set the
    region of interest.

    When you finish, you have to press the following key in your keyboard:
    Enter - OK
    ESC   - Exit (Cancel)
    """
    global selected
    global left_point
    global right_point

    left_point = None
    right_point = None

    # Create an OpenCV window.
    cv2.namedWindow("Select an area", cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback("Select an area", on_mouse)

    # Show the input image.
    selected = image.copy()
    update_image()

    # Handle with the mouse events.
    while True:
        # Get one keyboard event.
        ch = cv2.waitKey()

        # Cancel if the user press ESC.
        if ch == 27:
            return

        # Stop the while when press ENTER.
        if (ch == 13 and left_point is not None and
                right_point is not None):
            cv2.destroyWindow("Select an area")
            break

    # Create the selected points structure.
    points = []

    up_left = (min(left_point[0], right_point[0]),
               min(left_point[1], right_point[1]))
    down_right = (max(left_point[0], right_point[0]),
                  max(left_point[1], right_point[1]))

    points.append(up_left)
    points.append(down_right)

    # Return the final result.
    return points


def match_all(image, template, threshold, numOfScales=0):
    """
    This function does the cross correlation in an input image (image) and a
    template (template) and shows a rectangle around each possible match in the
    original image (i.e. larger than threshold).
    """
    # <Exercise 8.2 (Task 2.2)>


# Create the Matplotlib window.
# There is a bug on macOS that it is not possible to open an OpenCV windows
# before openning a Matplotlib windows.
fig = plt.figure()
plt.close()

# Load the input images.
image = cv2.imread("./inputs/pattern.jpg", cv2.IMREAD_COLOR)

# Select the template area.
points = select_roi(image)
if points is None:
    exit()

# Slicing the input image.
template = image[points[0][1]:points[1][1], points[0][0]:points[1][0]].copy()
h, w = template.shape[:2]

# Show both images in a Matplotlib window.
image = cv2.cvtColor(image,    cv2.COLOR_BGR2RGB)
template = cv2.cvtColor(template, cv2.COLOR_BGR2RGB)
show_images(Pattern=image, Template=template)

# <Exercise 8.2 (Task 2.1)>

# <Exercise 8.2 (Task 2.3)>