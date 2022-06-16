# Course: Introduction to Image Analysis and Machine Learning (ITU)
# Version: 2020.1

import cv2
import matplotlib.pyplot as plt
import numpy as np

# Get the input filename
filename = "./inputs/zico.jpg"

# Loads a gray-scale image from a file passed as argument.
image = cv2.imread(filename, cv2.IMREAD_COLOR)

# Create the Matplotlib figure.
fig = plt.figure("Images")

# This function creates a Matplotlib window and shows four images.
def show_image(image, pos, title="Image", isGray=False):
    sub = fig.add_subplot(1, 3, pos)
    sub.set_title(title)
    sub.imshow(image)
    plt.axis("off")
    if isGray:
        sub.imshow(image, cmap="gray")
    else:
        sub.imshow(image)

# <Exercise 2.5>

# <Exercise 2.5 (a)>

# Construct a mask to display the interested regions.

# <Exercise 2.5 (b)>

# <Exercise 2.5 (c)>

# <Exercise 2.5 (d)>

# Show the Matplotlib windows.
plt.show()
