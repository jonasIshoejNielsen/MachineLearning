import numpy as np
import matplotlib.pyplot as plt
import detector
import utils
import Evaluation
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
import gaze

class PolynomialGaze:
    """Linear regression model for gaze estimation.
    """

    def __init__(self, calibration_images, calibration_positions, order):
        """Uses calibration_images and calibratoin_positions to
        create regression mode.
        """
        self.images = calibration_images
        self.positions = calibration_positions
        self.order = order
        self.calibrate()

    def calibrate(self):
        """Create the regression model here.
        """
        xy = [detector.find_pupil(img, debug=False)[0] for img in self.images]
        features  = [[c[0],c[1]] for c in xy]
        poly = PolynomialFeatures(self.order)
        modelD = poly.fit_transform(features)
        
        positionX,positionY = [],[]
        for pos in self.positions:    #y,x positions      not x,y positions
            positionX.append(pos[1])
            positionY.append(pos[0])
        
        kx = linear_model.LinearRegression().fit(modelD, positionX)
        ky = linear_model.LinearRegression().fit(modelD, positionY)
        return kx, ky
    
    def estimate(self, image):
        """Given an input image, return the estimated gaze coordinates.
        """
        kx, ky = self.calibrate()
        (x,y)=detector.find_pupil(image, debug=False)[0]
        return estimateWithKxKy(image, kx, ky, self.order)

def estimateWithKxKy(image, kx, ky, order):
    """Given an input image, return the estimated gaze coordinates.
    """
    (x,y)=detector.find_pupil(image, debug=False)[0]
    poly = PolynomialFeatures(order)
    modelD = poly.fit_transform([[x,y]])   
    
    return kx.predict(modelD), ky.predict(modelD)




def estimatedModelDistances(gazeModels, models ):
    """
    for dirs [d1,d2,d3...,dn] and gazeModels [g1,g2,...,gn] and models [m1,m2,...,mn]
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
                    (estX, estY) = estimateWithKxKy(gazeModels[g].images[i], kx,ky, gazeModels[g].order)
                    distLstD.append(detector.euclidean_distance((estY, estX) ,gazeModels[g].positions[i]))
                distLst.append(distLstD)
        distRes.append(distLst)
    return distRes



def gazeModelMeansMedians(dirs, order, debugRes=False):
    gazeModels = [PolynomialGaze(utils.load_images(d), utils.load_json(d, "positions"), order) for d in dirs]
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
    order = 4
    dirsPattern  = ["pattern0", "pattern1", "pattern2", "pattern3"]
    dirsMovement = ["moving_medium", "moving_hard"]
    meanMedianPattern = gazeModelMeansMedians(dirsPattern,   order, debugRes=True)
    meanMedianMovement = gazeModelMeansMedians(dirsMovement, order, debugRes=True)
    histShow(meanMedianPattern, meanMedianMovement)
        
        

if __name__ == '__main__':
    main()
