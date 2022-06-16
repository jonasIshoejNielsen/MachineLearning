import os
from glob import glob

import cv2
import numpy as np
from skimage import io
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import math
import dset
import PCA

def RMSE (x_true, x_new):
    flatten1, flatten2 = np.array(x_true).flatten(), np.array(x_new).flatten()
    merged_list = [(flatten1[i] - flatten2[i])**2 for i in range(0, len(flatten1))]
    return math.sqrt(sum(merged_list))
    
    
    
    
def reconstruction_error(X, w, u):  #dataset X, principle components W, means/mu u
    t1 = PCA.trans_to_principal_component_space(X, w, u)
    t2 = PCA.trans_to_feature_space(t1, w, u)
    return RMSE(X, t2)
  
def plotErrors(x_range, errors):
    plt.plot(x_range, errors)
    plt.show()  

def getIndexList(eigenValues):
    eigenValuesx = np.array(eigenValues).shape[0]
    return list(range(1, eigenValuesx+1))
    

def plotErrors():
    pc                  = PCA.principalComponent()
    indexedList         = getIndexList(pc.eigenValues)
    errors  	        = [reconstruction_error(pc.eigenValues, pc.get_matrix_w(k), pc.mu) for k in indexedList]
    
    plt.plot(indexedList, errors)
    plt.xlabel("pca_dimension_sizes")
    plt.ylabel("Reconstruction error")
    plt.show()

def plotVariance():
    pc                  = PCA.principalComponent()
    indexedList         = getIndexList(pc.eigenValues)
    pv                  = pc.Proportional_variance()
    
    eigenValues         = [x[0] for x in pc.eigen_pairs]
    cpv                 = [pc.Cumulative_proportional_variance(eigenValues, k) for k in indexedList]

    plt.plot(indexedList, pv, label="Proportional variance")
    plt.plot(indexedList, cpv, label="Cumulative proportional variance")
    plt.xlabel("pca_dimension_sizes")
    plt.ylabel("Variance")
    plt.legend(loc="middle right")
    plt.show()
    
#plotErrors()
plotVariance()


