# Course: Introduction to Image Analysis and Machine Learning (ITU)
# Version: 2020.1

import cv2
import matplotlib.pyplot as plt
import numpy as np

class PointProcessing:
    # This vector contains a list of all valid gray-scale level.
    bins = np.array(range(256))

    def __init__(self, filename):
        """Initialises the plot and image windows."""
        # Create a Matplotlib window.
        self.fig = plt.figure()

        self.contrast = 1.  # contrast parameter.
        self.brightness = 0.  # brightness transformation
        self.inverted = 0.  # inversion (0/1)

        self.image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)

        self.setup_trackbar()

        self.contrast_plot = self.create_plot(1, "Contrast", "r-")
        self.brightness_plot = self.create_plot(2, "Brightness", "g-")
        self.negative_plot = self.create_plot(3, "negative", "b-")
        self.pp_plot = self.create_plot(4, "Point Processing", "k-")

        plt.plot()
        self.update()

    def create_plot(self, pos, name, color):
        """Helper function for creating one of the subplots."""
        sub = self.fig.add_subplot(2, 2, pos)
        sub.set_title(name)
        plt.axis([0, 255, 0, 255])
        plt.grid(None, 'major', 'both')
        return sub.plot(PointProcessing.bins, PointProcessing.bins,
                        color, linewidth=5)[0]

    def setup_trackbar(self):
        """Creates an OpenCV window with three trackbars."""
        # Creates an OpenCV window with three trackbars.
        cv2.namedWindow("Point Processing")

        cv2.createTrackbar("Contrast", "Point Processing",
                           10, 20, self.change_contrast)
        cv2.createTrackbar("Brightness", "Point Processing",
                           0, 256, self.change_brightness)
        cv2.createTrackbar("Negative", "Point Processing",
                           0, 1, self.change_inverted)

    def update(self):
        """Updates the plots and image whenever one of the trackbars is moved"""
        plt.pause(0.001)
        # Call here the transformation function.
        g = self.point_processing(self.image)

        # Display the resulting image.
        cv2.imshow("Point Processing", g)

        # Calculate plot data
        contrast_data = self.apply_contrast(self.bins)
        brightness_data = self.apply_brightness(self.bins)
        negative_data = self.apply_inverted(self.bins)
        pp_data = self.point_processing(self.bins)

        # Draw plots
        self.contrast_plot.set_ydata(contrast_data)
        self.brightness_plot.set_ydata(brightness_data)
        self.negative_plot.set_ydata(negative_data)
        self.pp_plot.set_ydata(pp_data)

        # Show the updated window.
        plt.show()
        plt.pause(0.001)

    def change_contrast(self, value):
        """Update change in contrast slider."""
        # <Exercise 2.3 (b)>

        print("a:%f" % value)

        self.contrast = value

        # Update the transformed image.
        self.update()

    def change_brightness(self, value):
        """Update change in brightness slider."""
        # <Exercise 2.3 (b)>

        print("b:%f" % value)
        self.brightness = value

        # Update the transformed image.
        self.update()

    def change_inverted(self, value):
        """Update change in inverted slider."""
        # <Exercise 2.3 (b)>

        print("c: %f" % value)
        self.inverted = value

        # Update the transformed image.
        self.update()

    def apply_contrast(self, f):
        """Apply contrast change to array. Note that this operation is the same for 1d and 2d (image) arrays."""
        # <Exercise 2.3 (c)>

        # Apply processing here
        g = self.contrast * f.astype('float')
        
        g = np.uint8(g)
        g[g > 255] = 255
        g[g < 0] = 0
        return g

    def apply_brightness(self, f):
        """Apply brightness change to array. Note that this operation is the same for 1d and 2d (image) arrays."""
        # <Exercise 2.3 (d)>

        g = f.astype('float')

        # Apply processing here
        b = np.zeros(g.shape)
        b[b != self.brightness] = self.brightness
        g = b + g
        
        g = g.astype('uint8')
        g[g > 255] = 255
        g[g < 0] = 0

        return g

    def apply_inverted(self, f):
        """Apply inversion change to array. Note that this operation is the same for 1d and 2d (image) arrays."""
        # <Exercise 2.3 (e)>
        g = f.astype('float')

        # Apply processing here
        if self.inverted == 1 :
            g = g.astype('uint8')
            return g
        b = np.zeros(g.shape)
        b[b != 255] = 255
        g = b - g
        
        g = g.astype('uint8')
        g[g > 255] = 255
        g[g < 0] = 0
        
        g = g.astype('uint8')
        return g

    def point_processing(self, f):
        """Combine all point processing functions."""
        # <Exercise 2.3 (f)>
        
        g = self.apply_contrast( f)
        g = self.apply_brightness( g)
        g = self.apply_inverted(g)

        return g

# Get the input filename
filename = "./inputs/lena.jpg"

pointProcessor = PointProcessing(filename)
