import cv2
import math
import numpy as np


def main():
    image = cv2.imread('inputs/ITULogo.png')
    window = TransformationWindow(image)
    window.update()

    if cv2.waitKey(0) == ord('q'):
            cv2.destroyAllWindows()


def perspective_transform(image, p=(0, 0, 0), theta=0, t=(0, 0), s=1.):
    """
    This function return a 3x3 transformation matrix that combines rotation,
    translation, scaling, and perspective warping.

    :param image: Input image.
    :param p: Perspective parameters (a, b, c).
    :param theta: Rotation angle.
    :param t: Translation (x, y).
    :param s: Scaling.
    """

    M = np.array([[s * math.cos(theta), s * (-math.sin(theta)), t[0]],
                  [s * math.sin(theta),    s * math.cos(theta), t[1]],
                  [p[0], p[1], p[2]]],
                 dtype=np.float64)

    rows, cols, _ = image.shape

    result = cv2.warpPerspective(image, M, (cols, rows))

    return result


class TransformationWindow:
    """Handles the transformation GUI."""

    def __init__(self, image):
        self.image = image

        self.angle = 0
        self.scale = 1
        self.translation = [0, 0]
        self.perspective = [0, 0, 1]

        cv2.namedWindow("Transformation")
        cv2.createTrackbar("Rotation", "Transformation",
                           180, 360, self.on_change_rotation)
        cv2.createTrackbar("Scale", "Transformation",
                           9, 19, self.on_change_scale)
        cv2.createTrackbar("Translation (X)", "Transformation",
                           50, 100, self.on_change_translation_x)
        cv2.createTrackbar("Translation (Y)", "Transformation",
                           50, 100, self.on_change_translation_y)
        cv2.createTrackbar("Perspective (A)", "Transformation",
                           50, 100, self.on_change_perspective_a)
        cv2.createTrackbar("Perspective (B)", "Transformation",
                           50, 100, self.on_change_perspective_b)
        cv2.createTrackbar("Perspective (C)", "Transformation",
                           50, 100, self.on_change_perspective_c)

    def on_change_rotation(self, value):
        self.angle = value-180
        self.update()

    def on_change_scale(self, value):
        self.scale = (value + 1) / 10.
        self.update()

    def on_change_translation_x(self, value):
        self.translation[0] = (value-50)*5
        self.update()

    def on_change_translation_y(self, value):
        self.translation[1] = -(value-50)*5
        self.update()

    def on_change_perspective_a(self, value):
        self.perspective[0] = (value/10000)-0.005
        self.update()

    def on_change_perspective_b(self, value):
        self.perspective[1] = (value/10000)-0.005
        self.update()

    def on_change_perspective_c(self, value):
        self.perspective[2] = (value/100)-0.5 + 1
        self.update()

    def update(self):
        #self.image = cv2.resize(self.image, (0, 0), fx=0.4, fy=0.4)
        theta = math.radians(self.angle)
        rotated = perspective_transform(
            self.image, self.perspective, theta, self.translation, self.scale)

        cv2.imshow("Transformation", np.hstack([self.image, rotated]))


if __name__ == '__main__':
    main()