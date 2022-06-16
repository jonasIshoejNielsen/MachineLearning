import numpy as np
import matplotlib.pyplot as plt
import detector
import utils
import Evaluation
from sklearn.preprocessing import PolynomialFeatures
import gaze
import transformations as trans


class NormalizedGaze:
    """Linear regression model for gaze estimation.
    """

    def __init__(self, calibration_images, calibration_positions):
        """Uses calibration_images and calibratoin_positions to
        create regression mode.
        """
        self.images = calibration_images
        self.positions = calibration_positions
        self.calibrate()

    def calibrate(self):
        """Create the regression model here.
        """
        xy = [detector.find_pupil(img, debug=False)[0] for img in self.images]
        xyNomr = self.normalize(xy)
        modelD = [[1,c[0],c[1]] for c in xyNomr]      
        positionX,positionY = [],[]
        for pos in self.positions:    #y,x positions      not x,y positions
            positionX.append(pos[1])
            positionY.append(pos[0])
            
        kx = np.linalg.lstsq(modelD,positionX, None)[0]
        ky = np.linalg.lstsq(modelD,positionY, None)[0]
        return kx, ky
    
    def estimate(self, image):
        """Given an input image, return the estimated gaze coordinates.
        """
        kx, ky = self.calibrate()
        (x,y)=detector.find_pupil(image, debug=False)[0]
        imageNomr = self.normalize([(x,y)])[0]
        return estimateWithPointKxKy(imageNomr, kx, ky)
    
    def normalize(self, pupils):
        res = []
        for i in range(len(pupils)):
            pupilX,pupiY = pupils[i]
            glints = detector.find_glints(self.images[i], (pupilX,pupiY))
            if len(glints) >0:
                maxX, maxY, minX, minY = maxMinList(glints)
                difFmaxminX, difFmaxminY = maxX-minX, maxY-minY
                glintsOut = [(normalize(x, minX, difFmaxminX),normalize(y, minY, difFmaxminY)) for (x,y) in glints]
                resEq =trans.learn_affine(trans.pointsToCorrecSyntax(glints), trans.pointsToCorrecSyntax(glintsOut)) 
                transFormed = resEq.transformationMultiplePointsApply_3x3([[pupilX], [pupiY]])
                res.append((transFormed[0][0], transFormed[1][0]))
            else:
                res.append((1,1))
        return res
        
def estimateWithKxKy(image, kx, ky):
    (x,y)=detector.find_pupil(image, debug=False)[0]
    return estimateWithPointKxKy((x,y), kx, ky)
    
def estimateWithPointKxKy(point, kx, ky):
    """Given an input point and model, return the estimated gaze coordinates.
    """
    (x,y)=point
    modelD = np.array([1,x,y])
    return np.dot(modelD, kx), np.dot(modelD, ky)

def normalize(f, fmin, difFmaxmin):
    if difFmaxmin == 0:
        return 1
    return (f-fmin)/difFmaxmin

def maxMinList (lst):
    maxX, maxY, minX, minY, first = None,None,None,None, True
    for (x,y) in lst:
        if first:
            first = False
            maxX, maxY, minX, minY = x,y, x,y
        if maxX < x:
            maxX = x
        if maxY < y:
            maxY = y
        if minX > x:
            minX = x
        if minY > y:
            minY = y
    return (maxX, maxY, minX, minY)   


def estimatedModelDistances(gazeModels, models ):
    """
    for gazeModels [g1,g2,...,gn] and models [m1,m2,...,mn]
    return: [res1, res2,..., resn]
        where resi = [disti_1, disti_2,..., disti_n, ] not including disti_i
    
    """
    distRes = []
    for d in range(len(gazeModels)):
        kx,ky = models[d]
        distLst = []
        for g in range(len(gazeModels)):
            if d != g:
                distLstD = []
                for i in range(len(gazeModels[g].images)):
                    (estX, estY) = estimateWithPointKxKy(detector.find_pupil(gazeModels[g].images[i], debug=False)[0], kx,ky)
                    distLstD.append(detector.euclidean_distance((estY, estX) ,gazeModels[g].positions[i]))
                distLst.append(distLstD)
        distRes.append(distLst)
    return distRes


def gazeModelMeansMedians(dirs, debugRes=False):
    gazeModels = [NormalizedGaze(utils.load_images(d), utils.load_json(d, "positions")) for d in dirs]
    models     = [g.calibrate() for g in gazeModels]
    dists      = estimatedModelDistances(gazeModels,  models)
    meanMedian = [[(Evaluation.mean(dist), Evaluation.median(dist)) for dist in distLst] for distLst in dists]
    if debugRes:
        print()
        print(dirs)
        for i in range(len(dists)):
            print(dirs[i]+ " (Means, Median) list")
            print(np.array(meanMedian[i]).astype(int))
    return meanMedian

def histFunc (ax, val, label):
    ax.hist(val, alpha=0.5, label=label, fill=False, cumulative=True, density=True)
    ax.set_title(label)

def histShow(meanMedianPattern, meanMedianMovement):
    
    fig, axes = plt.subplots(nrows=2, ncols=2)
    ax0, ax1, ax2, ax3 = axes.flatten()
    for lst in meanMedianPattern:
        for mean,median in lst:
            histFunc(ax0, mean,"mean-Pattern*")
            histFunc(ax1, median,"median-Pattern*")
    for lst in meanMedianMovement:
        for mean,median in lst:
            histFunc(ax2, mean,"mean-Moving*")
            histFunc(ax3, median,"median-Moving*")
    fig.tight_layout()
    
    plt.show()


def main():
    
    dirsPattern  = ["pattern0", "pattern1", "pattern2", "pattern3"]
    dirsMovement = ["moving_medium", "moving_hard"]
    meanMedianPattern = gazeModelMeansMedians(dirsPattern,   debugRes=True)
    meanMedianMovement = gazeModelMeansMedians(dirsMovement, debugRes=True)
    histShow(meanMedianPattern, meanMedianMovement)
        
        

if __name__ == '__main__':
    main()