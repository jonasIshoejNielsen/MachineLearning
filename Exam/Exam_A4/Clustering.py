import os
from glob import glob

import cv2
import numpy as np
from skimage import io
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import math
import pickle
import dset
import util
from sklearn.cluster   import *
from sklearn.neighbors import *
import operator

def getWindows(scale, windowsize, stride):
    path    = "IMM-Frontal Face DB SMALL"
    images  = np.array(dset.read_image_files(path, scale=scale))
    windows = [util.image_windows(img, size=windowsize, stride=stride) for img in images]
    return windows

def save(folder, title, item):
    with open(os.path.join(folder, title + '.pkl'), 'wb') as file:
        pickle.dump(item, file)

def load(folder, title):
    with open(os.path.join(folder, title + '.pkl'), 'rb') as file:
        return pickle.load(file)


class Cluster_img:
    def __init__(self, scale, windowsize, stride, model, title):
        self.windowsize     = windowsize
        self.stride         = stride
        windows             = getWindows(scale, windowsize, stride)
        mergedWindows       = np.vstack(windows)
        print("Start trainning cluster model:", title)
        self.model          = model.fit(mergedWindows)
        self.title          = title
    def plot(self):
        util.plot_image_windows(self.model.cluster_centers_, self.title,size=self.windowsize)

def loadAndPlotClusterCenters(title):
    cluster     = load("models", title)
    cluster.plot()   
    return cluster


def train(scale, windowsize, stride, model, title):
    print("Start", title)
    cluster     = Cluster_img(scale,windowsize, stride, model, title=title)
    save("models", title, cluster)

def histogramsForCluster(title, scale, isNotAgglomerative, saveHist = False, printHistograms=False):
    cluster             = load("models",title)
    mergedWindows       = getWindows(scale, cluster.windowsize, cluster.stride)
    if isNotAgglomerative:
        numberOfBins,_      = np.array(cluster.model.cluster_centers_).shape
        clusterPrediction   = np.array([cluster.model.predict(window) for window in mergedWindows])
    else:
        numberOfBins        = cluster.model.n_clusters_
        clusterPrediction   = np.array([cluster.model.labels_[i] for i,_ in enumerate(mergedWindows)])
    histograms          = []
    for i,clusters in enumerate(clusterPrediction):
        plt.close()
        n, bins, patches = plt.hist(clusters, bins=list (range(0,numberOfBins)), rwidth=0.5, align='left', density=True, facecolor='green', alpha=0.75)
        if saveHist and i<3:
            plt.title(title+"-image:"+str(i))
            plt.xlabel('ClusterId')
            plt.ylabel('Probability')
            plt.savefig("outputImages/Histograms_cluster/hist_cluster_"+title+"_img"+str(i)+".png")
            if printHistograms:
                plt.show()
        histograms.append(n)
    return np.array(histograms)

def TrainNearestNeighbors(title, histograms):
    print("Start: CreateNearestNeighbor:", title)
    model               = NearestNeighbors()
    model.fit(histograms)
    save("NearestNeighbors", title, model)


def kneighbors(histograms, title, x, k ):
    print("Start: kneighbors:", title)
    model               = load("NearestNeighbors", title)
    selectedHistograms  = np.array([histograms[i] for i in x])
    #neigh_dist          = model.kneighbors(selectedHistograms, k, return_distance=True)
    neigh_ind           = model.kneighbors(selectedHistograms, k, return_distance=False)
    return neigh_ind

def modelTitle(modelType, scale, windowsize, stride):
    return modelType+"_scale_"+str(scale)+"_windowsize_"+str(windowsize)+"_stride_"+str(stride)

clustersToFind =16
listOfModels = [
        (0.1,   (10,10), (10,10), KMeans(n_clusters=clustersToFind,n_jobs=20),        modelTitle("KMeans", 0.1,                     (10,10), (10,10)), True),
        (0.25,  (10,10), (10,10), KMeans(n_clusters=clustersToFind,n_jobs=20),        modelTitle("KMeans", 0.25,                    (10,10), (10,10)), True),
        (0.1,   (20,20), (10,10), KMeans(n_clusters=clustersToFind,n_jobs=20),        modelTitle("KMeans", 0.1,                     (20,20), (10,10)), True),
        (0.25,  (20,20), (10,10), KMeans(n_clusters=clustersToFind,n_jobs=20),        modelTitle("KMeans", 0.25,                    (20,20), (10,10)), True),
        (0.1,   (10,10), (10,10), MeanShift(n_jobs=20),                               modelTitle("MeanShift", 0.1,                  (10,10), (10,10)), True),
        (0.1,   (20,20), (10,10), MeanShift(n_jobs=20),                               modelTitle("MeanShift", 0.1,                  (20,20), (10,10)), True),
        (0.1,   (10,10), (20,20), KMeans(n_clusters=clustersToFind,n_jobs=20),        modelTitle("KMeans", 0.1,                     (10,10), (20,20)), True),
        (0.25,  (10,10), (20,20), KMeans(n_clusters=clustersToFind,n_jobs=20),        modelTitle("KMeans", 0.25,                    (10,10), (20,20)), True),
        (0.1,   (20,20), (20,20), KMeans(n_clusters=clustersToFind,n_jobs=20),        modelTitle("KMeans", 0.1,                     (20,20), (20,20)), True),
        (0.25,  (20,20), (20,20), KMeans(n_clusters=clustersToFind,n_jobs=20),        modelTitle("KMeans", 0.25,                    (20,20), (20,20)), True),
        (0.1,   (10,10), (20,20), MeanShift(n_jobs=20),                               modelTitle("MeanShift", 0.1,                  (10,10), (20,20)), True),
        (0.1,   (20,20), (20,20), MeanShift(n_jobs=20),                               modelTitle("MeanShift", 0.1,                  (20,20), (20,20)), True),
        (0.1,   (10,10), (10,10), AgglomerativeClustering(n_clusters=clustersToFind), modelTitle("AgglomerativeClustering", 0.1,    (10,10), (10,10)), False),
        (0.1,   (20,20), (10,10), AgglomerativeClustering(n_clusters=clustersToFind), modelTitle("AgglomerativeClustering", 0.1,    (20,20), (10,10)), False),
        (0.1,   (10,10), (20,20), AgglomerativeClustering(n_clusters=clustersToFind), modelTitle("AgglomerativeClustering", 0.1,    (10,10), (20,20)), False),
        (0.1,   (20,20), (20,20), AgglomerativeClustering(n_clusters=clustersToFind), modelTitle("AgglomerativeClustering", 0.1,    (20,20), (20,20)), False),
        (0.1,   (10,10), (10,10), AgglomerativeClustering(n_clusters=clustersToFind), modelTitle("AgglomerativeClustering", 0.25,   (10,10), (10,10)), False),
        (0.1,   (20,20), (10,10), AgglomerativeClustering(n_clusters=clustersToFind), modelTitle("AgglomerativeClustering", 0.25,   (20,20), (10,10)), False),
        (0.1,   (10,10), (20,20), AgglomerativeClustering(n_clusters=clustersToFind), modelTitle("AgglomerativeClustering", 0.25,   (10,10), (20,20)), False),
        (0.1,   (20,20), (20,20), AgglomerativeClustering(n_clusters=clustersToFind), modelTitle("AgglomerativeClustering", 0.25,   (20,20), (20,20)), False)
    ]

def trainAllModels():
    for (scale, windowsize,stride, model, title, isNotAgglomerative) in listOfModels:
        train(scale, windowsize, stride, model, title)
        histograms = histogramsForCluster(title, scale, isNotAgglomerative, saveHist=False, printHistograms=False)
        TrainNearestNeighbors(title, histograms)

def allAssingments(plotClusterCenters=False, printHistograms=False, plotkneighbors=False):
    numberOfImages = len(dset.read_image_files("IMM-Frontal Face DB SMALL", scale=1))
    if plotClusterCenters:
        for (scale, windowsize,stride, model, title, isNotAgglomerative) in listOfModels:
            if isNotAgglomerative:
                loadAndPlotClusterCenters(title)
        plt.show()
    for (scale, windowsize,stride, model, title, isNotAgglomerative) in listOfModels:
        print("Start:", title)         
        #task4.3.2
        histograms      = histogramsForCluster(title, scale,isNotAgglomerative, saveHist=True, printHistograms=printHistograms)
        all_kneighbors  = kneighbors(histograms, title, range(0,5), 3)
        print("allkneighbors=",all_kneighbors.shape)
        print(all_kneighbors)
        print()
        print()
        plt.close()
        plt.hist(np.array(all_kneighbors).T, bins=numberOfImages, label=["image_"+str(i) for i,_ in enumerate(all_kneighbors)])
        plt.legend(loc="upper right")
        plt.title(title)
        plt.savefig("outputImages/Histograms_images/hist_images_"+title+".png")
        if plotkneighbors:
            plt.show()
trainAllModels()
allAssingments(plotClusterCenters=True, printHistograms=False, plotkneighbors=False)