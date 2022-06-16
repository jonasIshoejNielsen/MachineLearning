# Course: Introduction to Image Analysis and Machine Learning (ITU)
# Version: 2020.1

import cv2
import matplotlib.pyplot as plt
import numpy as np

# Get the input filename
filename = "./inputs/lena.jpg"

# Loads a gray-scale image from a file passed as argument.
image = cv2.imread(filename, cv2.IMREAD_COLOR)
grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Create the Matplotlib figures.
fig_imgs = plt.figure("Images")
fig_hist = plt.figure("Histograms")


# This function creates a Matplotlib window and shows four images.
def show_image(image, pos, title="Image", isGray=False):
    sub = fig_imgs.add_subplot(2, 2, pos)
    sub.set_title(title)
    if isGray:
        sub.imshow(image, cmap="gray")
    else:
        sub.imshow(image)
    sub.axis("off")


# This function creates a Matplotlib window and shows four histograms.
def show_histogram(histogram, pos, title="Histogram"):
    sub = fig_hist.add_subplot(2, 2, pos)
    sub.set_title(title)
    plt.xlabel("Bins")
    plt.ylabel("Number of Pixels")
    plt.xlim([0, 256])
    plt.ylim([0, 10000])
    plt.plot(histogram)

# <Exercise 2.4 (a)>
# Gray-scale image.
#grayHist = cv2.calcHist(grayscale, [0], None, [256], [0,255])
#show_histogram(grayHist, (2,10))

# <Exercise 2.4 (b)>
# Shuffled image.
#shuffledImg = np.random.shuffle(image)
shuffledImg = np.random.shuffle(image)
shuffledHist = cv2.calcHist([shuffledImg],[0],None,[256],[0,256])
originalHist = cv2.calcHist([image],[0],None,[256],[0,256])

show_histogram(shuffledHist, (-10,20))
show_histogram(originalHist, (-10,20))
print("originalHist",originalHist)
# <Exercise 2.4 (c)>
# Histogram distance

# <Exercise 2.4 (d)>
# Calculate the distance between regular and shuffled image

# <Exercise 2.4 (e)>
# RGB image.

# <Exercise 2.4 (f)>
# HSV image.

# Show the Matplotlib windows.
plt.show()
