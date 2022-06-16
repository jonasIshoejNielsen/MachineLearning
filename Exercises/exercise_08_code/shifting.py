import argparse
import cv2
import matplotlib.pyplot as plt
import numpy as np

# Construct the argument parser and parse the arguments.
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Input Image")
args = vars(ap.parse_args())


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



#shifting moved to sharpening.py