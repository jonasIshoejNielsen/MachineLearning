import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
import utils
import os   # read folder
import detector

def dist(p1,p2):
    return detector.euclidean_distance(p1,p2)

def calDistancesToGlints(img, c, realGlints, distThisImg, differenceNumberOfGlints, threshold=0, useThreshold=False):
    glints = []
    if useThreshold:
        glints = detector.find_glints(img, c, debug=False, threshold=threshold)
    else:
        glints = detector.find_glints(img, c, debug=False)
    differenceNumberOfGlints.append(len(realGlints) - len(glints))
    for g in glints:
        first,gDistMin = True, None
        for realG in realGlints:
            gDist = dist(realG, g)
            if first or gDistMin > gDist:
                gDistMin = gDist
                first=False
        distThisImg.append(gDist)
        

def mean(lst):
    sum = 0.0
    for i in lst:
        sum +=i
    return sum/len(lst)
def median(lst):
    lstC = lst.copy()
    lstC.sort()
    return lstC[int(len(lst)/2)-1]

def histShowAxis(value, axis):
    (distance, label) = value
    print(label+ " Means, Median")
    print(mean(distance), median(distance))
    axis.hist(distance, alpha=0.5, label=label, fill=False)
    axis.set_title(label)

def histShow(values):
    """param: values = [(distance1, label1), (distance2, label2)] """
    
    fig, axes = plt.subplots(nrows=1, ncols=3)
    ax0, ax1, ax2 = axes.flatten()
    print("Pupils:")
    histShowAxis(values[0],ax0)
    print("Glints:")
    histShowAxis(values[1],ax1)
    histShowAxis(values[2],ax2)
    fig.tight_layout()
    plt.show()
    """
    for (distance, label) in values:
        plt.hist(distance, density=True, bins=30, label=label)
    plt.show()"""

def getDistances():
    dirs = ["pattern0", "pattern1", "pattern2"]
    dirs = [(utils.load_images(d), utils.load_json(d, "glints"), utils.load_json(d, "pupils"), d) for d in dirs]
    
    distancesPupils = []
    distancesGlints = []
    differenceNumberOfGlints = []
    for (imgs, glints, pupils, d) in dirs:
        for i in range(len(imgs)):
            c, ax, angle = detector.find_pupil(imgs[i], debug=False)
            distancesPupils.append(dist(c, [pupils[i]['cx'], pupils[i]['cy'] ]))
            calDistancesToGlints(imgs[i], c, glints[i], distancesGlints, differenceNumberOfGlints, useThreshold=False)
    return distancesPupils, distancesGlints, differenceNumberOfGlints

def getBestDistancesPupils(threshold):
    dirs = ["pattern0", "pattern1", "pattern2"]
    dirs = [(utils.load_images(d), utils.load_json(d, "glints"), utils.load_json(d, "pupils"), d) for d in dirs]
    distancesPupils = []
    for (imgs, glints, pupils, d) in dirs:
        for i in range(len(imgs)):
            c, ax, angle = detector.find_pupil(imgs[i], debug=False, threshold=threshold)
            distancesPupils.append(dist(c, [pupils[i]['cx'], pupils[i]['cy'] ]))
    return distancesPupils

def computeBestThreshold():
    bestTPupils, bestDistancesPupils = 0, []
    first, bestMedianPupils, bestMedianGlints = True, 0,0
    for t in range(40, 80):
        print(t)
        distPupiils = getBestDistancesPupils(t)
        medPupils = median(distPupiils) + mean(distPupiils)
        if first:
            first=False
            bestTPupils = t
            bestDistancesPupils = distPupiils
            bestMedianPupils = medPupils
        if medPupils<bestMedianPupils:
            bestMedianPupils = medPupils
            bestTPupils = t
            bestDistancesPupils = distPupiils
    
    return (bestTPupils, bestDistancesPupils)

def main():   
    """
    t1, distancesPupils = computeBestThreshold()
    print(t1)
    """ 
    distancesPupils, distancesGlints, differenceNumberOfGlints = getDistances()
    histShow([[distancesPupils, "distancespupils"], [distancesGlints, "distancesGlints"], [differenceNumberOfGlints, "differenceNumberOfGlints"]])
    
        

if __name__ == '__main__':
    main()
