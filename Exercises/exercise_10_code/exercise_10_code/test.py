import cv2
import numpy as np

from util import load_dataset


def load_video(path):
    """Load Video from a subdirectory in path using OpenCV.
    """
    #print(os.path.isfile(path))
    #return np.load(path, allow_pickle=True)
    
    cap = cv2.VideoCapture(path)
    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps          = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    while(True):
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
    cap.release()
    
    return frames, fps

# Create the HOG descriptor and import the HOG feature.
feature = np.load("./outputs/feature.npy")
hog = cv2.HOGDescriptor()
hog.setSVMDetector(feature)

# Dataset filename.
positive_file = "./inputs/Test/pos.lst"

# Load the INRIA dataset.
positive_list = load_dataset(positive_file)

# <Exercise 10.2 (all tasks)>
for img in positive_list:
    (rects, weights) = hog.detectMultiScale(img)
    for (x, y, w, h) in rects:
	    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
    cv2.imshow("Detections", img)
    cv2.waitKey(0)