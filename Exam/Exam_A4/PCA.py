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

class principalComponent:
    def __init__(self):
        path = "IMM-Frontal Face DB SMALL"
        shapes, _ = dset.face_shape_data(path)
        self.shapes = np.array(shapes).T
        self.means = np.array([np.mean(img) for img in np.array(self.shapes).T])
        self.mu = np.array([np.mean(line) for line in self.shapes])
        cov_matrix = np.cov(self.shapes - self.means)
        eigen_val, eigen_vec = np.linalg.eig(cov_matrix)
        self.eigen_pairs = [(np.abs(eigen_val[i]), eigen_vec[:,i]) for i in range(len(eigen_val))]
        self.eigen_pairs.sort(key=lambda x: x[0], reverse=True)
        self.eigenValues = [x[0] for x in self.eigen_pairs]
            
    def get_matrix_w(self, pca_dimension_size):
        return np.array([x[1] for x in self.eigen_pairs[:pca_dimension_size]])      #self.pca_dimension_size × d -dimensional eigenvector matrix
    
    def Proportional_variance(self):
        eigenValues = [x[0] for x in self.eigen_pairs]
        sumValues   = sum(eigenValues)
        return [x / sumValues for x in eigenValues]
   
    def Cumulative_proportional_variance(self, eigenValues, pca_dimension_size):
        sumValues   = sum(eigenValues)
        return sum(eigenValues[:pca_dimension_size]) / sumValues
      
    def Standatd_deviation(self):
        return [math.sqrt(x[0]) for x in self.eigen_pairs] 
    
   
            
def trans_to_principal_component_space(vectors,  matrix_w, mu):   #self.vector, self.principal_components, self.mu
    return matrix_w.dot(vectors - mu)         # 10x120 * 120*1 = 10x1 

def trans_to_feature_space(vectors,  matrix_w, mu):               #self.vector, self.principal_components, self.mu
    return matrix_w.T.dot(vectors) + mu       # 120x10 * 10*1 = 120
    
def testOfUse():
    pc              = principalComponent()
    matrix_w        = pc.get_matrix_w(10)   
    vector          = np.zeros((matrix_w.shape[0], ))
    print(np.array([x[0] for x in pc.eigen_pairs]).shape)
    t0              = trans_to_feature_space(vector, matrix_w, pc.mu)  
    print('t0:', np.array(t0).shape)
    
    t1              = trans_to_principal_component_space(pc.eigenValues, matrix_w, pc.mu)
    print('t1:', np.array(t1).shape)
    t2              = trans_to_feature_space(t1, matrix_w, pc.mu)  
    print('t2:', np.array(t2).shape)
    
#testOfUse()