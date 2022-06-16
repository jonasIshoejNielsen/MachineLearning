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
import util



def createSliderPlot(n):
    pc                  = PCA.principalComponent()
    matrix_w            = np.array(pc.get_matrix_w(n))
    std                 = pc.Standatd_deviation()
    sp                  = util.SliderPlot(matrix_w, std, pc.mu, PCA.trans_to_feature_space)
    plt.show()
    if cv2.waitKey(1) == 'q':
        return
    
createSliderPlot(15)
createSliderPlot(10)
createSliderPlot(5)
createSliderPlot(2)