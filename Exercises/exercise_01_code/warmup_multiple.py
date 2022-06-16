# Course: Introduction to Image Analysis and Machine Learning (ITU)
# Version: 2020.1

import matplotlib
matplotlib.use("tkagg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Vector to keep the input images
images = []

# Open the images as color images.
images.append(mpimg.imread("./inputs/tinky_winky.jpg"))
images.append(mpimg.imread("./inputs/dipsy.jpg"))
images.append(mpimg.imread("./inputs/laa_laa.jpg"))
images.append(mpimg.imread("./inputs/po.jpg"))

# Create the Matplotlib window.
fig = plt.figure("Images")

# Show the input images in a Matplotlib window.
for i, image in zip(range(4), images):
    fig.add_subplot(1, 4, i + 1)
    plt.axis("off")
    plt.imshow(image)

plt.show()

# Create the Matplotlib window.
fig = plt.figure("Matrix")

# Show the input images in a Matplotlib window.
for i, image in zip(range(4), images):
    fig.add_subplot(2, 2, i + 1)
    plt.axis("off")
    plt.imshow(image)

plt.show()