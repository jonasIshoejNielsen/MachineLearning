# Course: Introduction to Image Analysis and Machine Learning (ITU)
# Version: 2020.1

import cv2

# Full path where is located the input image.
filepath = "./inputs/lena.jpg"

# Open the image as a grayscale image.
image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)

# Show the input image in a OpenCV window.
cv2.imshow("Image", image)
cv2.waitKey(0)

# Save the converted image.
cv2.imwrite("./outputs/lena_grayscale.jpg", image)

# When everything done, release the OpenCV window.
cv2.destroyAllWindows()
