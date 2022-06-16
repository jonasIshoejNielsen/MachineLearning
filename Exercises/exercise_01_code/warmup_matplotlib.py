# Course: Introduction to Image Analysis and Machine Learning (ITU)
# Version 2020.1

import cv2
import matplotlib
matplotlib.use("tkagg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Full path where is located the input image.
filepath = "./inputs/black.jpg"

# Open the image as a grayscale image.
image1 = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
print("Width: %d pixels." % (image1.shape[1]))
print("Height: %d pixels." % (image1.shape[0]))

# Open the image using Matplotlib.
image2 = mpimg.imread(filepath)
image2 = cv2.cvtColor(image2, cv2.COLOR_RGB2GRAY)

# Create the Matplotlib window.
plt.figure("Matplotlib")

# Create the OpenCV window.
cv2.namedWindow("OpenCV", cv2.WINDOW_AUTOSIZE)

# Show the input image in a OpenCV window.
cv2.imshow("OpenCV", image1)

# Show the input image in a Matplotlib window.
imgplot = plt.imshow(image2)
plt.show()

# When everything done, release the OpenCV window.
cv2.destroyAllWindows()
