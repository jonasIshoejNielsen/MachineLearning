import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
import utils
import os   # read folder
import detector
import Evaluation
import PolynomialGaze
import NormalizedGaze
import json



def load_json(directory, filename):
    """Load json file from subdirectory in "inputs/videos" with the given filename
    - without .json extension!

    Returns: The json data as a dictionary or array (depending on the file).
    """
    with open(os.path.join('inputs/videos', directory, f'{filename}.json')) as file:
        return json.load(file)


def load_images(directory):
    """Load images from a subdirectory in "inputs/videos" using OpenCV.

    Returns: The list of loaded images in order.
    """
    with open(os.path.join('inputs/videos', directory, 'positions.json')) as file:
        screen_points = json.load(file)

    images = [cv2.imread(os.path.join(
        'inputs/videos', directory, f'{i}.jpg')) for i in range(len(screen_points))]

    return images


def load_video(path):
    """Load Video from a subdirectory in "inputs/videos" using OpenCV.
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

def alterImage(eyePath, screenPath, output, kx, ky, order):
    framesEye, _ = load_video(eyePath)
    framesScreen, fps = load_video(screenPath)
    
    height, width, layers = framesScreen[0].shape
    video = cv2.VideoWriter(output, 0, fps, (width,height))
    for i in range(min(len(framesEye), len(framesScreen)) ):
        gazeX, gazeY = PolynomialGaze.estimateWithKxKy(framesEye[i], kx,ky, order)          # for PolynomialGaze
        #gazeX, gazeY = NormalizedGaze.estimateWithKxKy(framesEye[i], kx,ky)                 # for NormalizedGaze
        currFrameScreen = framesScreen[i]
        fitx, fity = min(width, max(0, int(gazeX))), min(height, max(0, int(gazeY)))
        cv2.circle(currFrameScreen, (int(fitx), int(fity)),10, (0,0,255), 2)
        
        video.write(currFrameScreen)

def getModelKxKy(order, folder):
    images    = load_images(folder)
    positions = load_json(folder, "positions")
    kx,ky = PolynomialGaze.PolynomialGaze(images, positions, order).calibrate()         # for PolynomialGaze
    #kx,ky = NormalizedGaze.NormalizedGaze(images, positions).calibrate()               # for NormalizedGaze
    return kx,ky

def main(): 
    order = 4
    kx,ky = getModelKxKy(order, "video0")
    print("start video0")
    alterImage('inputs/videos/video0/eyes.avi','inputs/videos/video0/screen.avi', "outputs/video0-screen.avi", kx,ky,order )
    print("start video1")
    kx,ky = getModelKxKy(order, "video1")
    alterImage('inputs/videos/video1/eyes.avi','inputs/videos/video1/screen.avi', "outputs/video1-screen.avi", kx,ky,order )
    print("done both")
    
if __name__ == '__main__':
    main()
