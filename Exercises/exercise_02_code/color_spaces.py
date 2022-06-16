# Course: Introduction to Image Analysis and Machine Learning (ITU)
# Version: 2020.1

import numpy as np
import cv2

filename = './inputs/lena.jpg'

# <Exercise 2.2>
file = cv2.imread(filename)
newFile = cv2.cvtColor(file, cv2.COLOR_BGR2RGB)

r,g,b = cv2.split(newFile)
#cv2.imshow("newFiler", r)
#cv2.imshow("newFileg", g)
#cv2.imshow("newFileb", b)


zeroSameShape = np.zeros(r.shape[:2], dtype=np.uint8)
print("shaper", r.shape)
print("r=", r)
print("shapezero", zeroSameShape.shape)
print("zero=", zeroSameShape)

#pureRed = cv2.merge([zeroSameShape, g, zeroSameShape])

cv2.imshow("original", file)
cv2.imshow("combined", cv2.merge([b, g, r]))
cv2.imshow("Channel R", cv2.merge([zeroSameShape, zeroSameShape, r]))
cv2.imshow("Channel G", cv2.merge([zeroSameShape, g, zeroSameShape]))
cv2.imshow("Channel B", cv2.merge([b, zeroSameShape, zeroSameShape]))



cv2.waitKey(0)
