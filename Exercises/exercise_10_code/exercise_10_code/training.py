import cv2
import numpy as np
import random
import math

from util import load_dataset


def get_svm_detector(svm):
    """
    This function calculates and returns the feature descriptor.
    """
    # Retrieves all the support vectors.
    sv = svm.getSupportVectors()

    # Retrieves the decision function.
    rho, _, _ = svm.getDecisionFunction(0)

    # Transpose the support vectors matrix.
    sv = np.transpose(sv)

    # Returns the feature descriptor.
    return np.append(sv, [[-rho]], 0)


def sample_negative_images(images, negative_sample, size=(64, 128), N=10):
    """
    INRIA Dataset has several images of different resolution without pedestrians,
    i.e. called here as negative images. This function select "N" 64x128 negavite
    sub-images randomly from each original negative image.
    """
    # Initialize internal state of the random number generator.
    random.seed(1)

    # Final image resolution.
    w, h = size[0], size[1]

    # Read all images from the negative list.
    for image in images:
        # Generate N image samples from the current negative image.
        for j in range(N):
            y = int(random.random() * (len(image) - h))
            x = int(random.random() * (len(image[0]) - w))
            sample = image[y:y + h, x:x + w].copy()
            negative_sample.append(sample)


def random_range(min, max):
    return random.random() * (max - min) + min


def sample_positive_images(images, positive_sample):
    """
    Creates positive samples using data augmentation.

    :param images: List of positive images.
    :param positive_sample: Output list of samples
    """

    for image in images:
        positive_sample.append(image)


def compute_hog(images, hog_list, size=(64, 128)):
    """
    This function computes the Histogram of Oriented Gradients (HOG) of each
    image from the dataset.
    """
    # Creates the HOG descriptor and detector with default parameters.
    hog = cv2.HOGDescriptor()
    sizeX,sizeY=size
    # Read all images from the image list.
    for image in images:

        # <Exercise 10.1 (Task 1.1>
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        (y,x) = gray.shape
        gray3 = gray[int((y-sizeY)/2):int((y+sizeY)/2), int((x-sizeX)/2):int((x+sizeX)/2)]
        gray2 = gray[0:sizeY, 0:sizeX]
        hog = cv2.HOGDescriptor().compute(gray3)
        hog_list.append(hog)
        #todo add size Use the centered 64  128 pixels window in the positive images



# Dataset filenames.
positiveFile = "./inputs/train/pos.lst"
negativeFile = "./inputs/train/neg.lst"

# Vectors used to train the dataset.
hogList = []
positive_list = []
negative_list = []
positive_sample = []
negative_sample = []
hard_negative_list = []
labels = []

# Load the INRIA dataset.
positive_list = load_dataset(positiveFile)
print('[INFO] loaded positive dataset')
negative_list = load_dataset(negativeFile)
print('[INFO] loaded negative dataset')

# Get a sample of negative images.
sample_negative_images(negative_list, negative_sample)
print('[INFO] created negative samples')
sample_positive_images(positive_list, positive_sample)
print('[INFO] created positive samples')

# Compute the Histogram of Oriented Gradients (HOG).
compute_hog(positive_sample, hogList)
print('[INFO] computed positive HOG features')
compute_hog(negative_sample, hogList)
print('[INFO] computed negative HOG features')
if len(hogList) == 0:
    print('[ERROR] no images found')
    exit(0)

# Create the class labels, i.e. (+1) positive and (-1) negative.
[labels.append(+1) for _ in range(len(positive_sample))]
[labels.append(-1) for _ in range(len(negative_sample))]

# Create an empty SVM model.
svm = cv2.ml.SVM_create()

# Define the SVM parameters.
# By default, Dalal and Triggs (2005) use a soft (C=0.01) linear SVM trained
# with SVMLight.
svm.setDegree(3)
criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 1000, 1e-3)
svm.setTermCriteria(criteria)
svm.setGamma(0.1)
svm.setKernel(cv2.ml.SVM_LINEAR)
svm.setNu(0.5)
svm.setP(0.1)
svm.setC(0.01)
svm.setType(cv2.ml.SVM_EPS_SVR)

print('[INFO] started first training')

# <Exercise 10.1 (Task 1.2>
print(cv2.ml.ROW_SAMPLE)
print(np.array(labels).shape)
print(np.array(hogList).shape)
svm.train(np.array(hogList),cv2.ml.ROW_SAMPLE, np.array(labels))
print("[INFO] Model trained")


print('[INFO] completed first training')

# Create the HOG descriptor and detector with default params.
hog = cv2.HOGDescriptor()
print("test1")
hog.setSVMDetector(get_svm_detector(svm))
print("test2")

feature = get_svm_detector(svm)
print("test3")
np.save("./outputs/feature_first.npy", feature)

print('[INFO] started hard negative mining')


# <Exercise 10.1 (Task 2.1)>

# <Exercise 10.1 (Task 2.2)>

print('[INFO] completed final training')

# Save the HOG feature.
feature = get_svm_detector(svm)
np.save("./outputs/feature.npy", feature)
