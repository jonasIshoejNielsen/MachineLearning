import cv2


def load_dataset(dataset):
    """
    This function load all images from a dataset and return a list of Numpy images.
    """
    # List of images.
    images = []

    # Read the dataset file.
    file = open(dataset)
    filename = file.readline()

    # Read all filenames from the dataset.
    while filename != "":
        # Get the current filename.
        filename = (dataset.rsplit("/", 1)[0] + "/" +
                    filename.split("/", 1)[1].strip("\n"))

        # Read the input image.
        image = cv2.imread(filename)

        # Read the next image filename.
        filename = file.readline()

        # Add the current image on the list.
        if image is not None:
            images.append(image)

    # Return the images list.
    return images