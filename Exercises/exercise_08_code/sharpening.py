import cv2
import matplotlib.pyplot as plt
import numpy as np


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
        plt.subplot(2, len(images) / 2, pos+1)
        plt.title(name)
        plt.imshow(image)

    # Show the images.
    plt.show()


def shift_to_left(image, n):
    """Shift all pixel of the input image n column to the left."""
    result = image.copy()

    n = 2 * n + 1

    H = np.zeros((n, n))
    H[n//2, n-1] = 1.

    result = cv2.filter2D(result, -1, H, borderType=cv2.BORDER_CONSTANT)

    return result


# Read image
image = cv2.imread('inputs/lena.jpg', cv2.IMREAD_GRAYSCALE)


# <Exercise 8.3.2>

filterS1 = np.array([[-1, -1, -1],
                     [-1, 17, -1],
                     [-1, -1, -1]]) / 9.

filterS2 = np.array([[-1,  -1,  -1],
                     [-1, 9,  -1],
                     [-1,  -1,  -1]]) / 9

filterS3 = np.array([[-1, -1, -1, -1, -1],
                     [-1,  2,  2,  2, -1],
                     [-1,  2,  8,  2, -1],
                     [-1,  2,  2,  2, -1],
                     [-1, -1, -1, -1, -1]]) / 8.

# <Exercise 8.3.3>
filteredS1 = cv2.filter2D(image, -1, filterS1)
filteredS2 = cv2.filter2D(image, -1, filterS2)
filteredS3 = cv2.filter2D(image, -1, filterS3)

show_images(image=image, filteredS1=filteredS1,
            filteredS2=filteredS2, filteredS3=filteredS3)

filteredS4 = shift_to_left(image, 200)

show_images(image=image, filteredS1=filteredS4)















def filterImage(image, H):
    result = image.copy()
    result = cv2.filter2D(result, -1, H, borderType=cv2.BORDER_CONSTANT)
    result = result.astype(int)
    return result


def shift_to_left2(image, n):
    """Shift all pixel of the input image n column to the left."""
    # <Exercise 8.4.2>
    # Change this Python code.
    # H = np.array([[0, 0, 0],
    #               [0, 0, 1],
    #               [0, 0, 0]])

    # for i in range(n):
    #     result = cv2.filter2D(result, -1, H, borderType = cv2.BORDER_CONSTANT)

    n = 2 * n + 1
    H = np.zeros((n, n))
    H[n//2, n-1] = 1.
    return filterImage(image, H)


# <Exercise 8.4.3>

def shift_to_right(image, n):
    n = 2 * n + 1
    H = np.zeros((n, n))
    H[n//2, 0] = 1.
    return filterImage(image, H)


def shift_to_up(image, n):
    n = 2 * n + 1
    H = np.zeros((n, n))
    H[n-1,n//2] = 1.
    return filterImage(image, H)


def shift_to_down(image, n):
    n = 2 * n + 1
    H = np.zeros((n, n))
    H[0,n//2] = 1.
    return filterImage(image, H)


# Read image
image = cv2.imread('inputs/lena.jpg', cv2.IMREAD_GRAYSCALE)

# Apply filters
n = 100
filteredLeft    = shift_to_left2(image, n)
filteredRight   = shift_to_right(image, n)
filteredup      = shift_to_up(image, n)
filteredDown    = shift_to_down(image, n)
# Show examples
show_images(filteredLeft=filteredLeft, filteredRight=filteredRight, filteredup=filteredup, filteredDown=filteredDown)
