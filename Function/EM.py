import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt

class Cluster():
    def __init__(self, data, colors = ['red', 'blue'], k = 2, iter = 10):
        self.k = k
        self.data = data
        self.N = len(data)
        self.center = np.array([data[i] for i in range(k)])
        self.R = np.zeros((self.N, k))
        self.distance = np.zeros((self.N, k))
        self.J = 1e10
        self.J_ = []
        self.colors = colors
        self.iter = iter

    def DistributeData(self):
        for i in range(self.N):
            self.distance[i] = np.array([(self.data[i][0]-self.center[j][0])**2 
                                         + (self.data[i][1]-self.center[j][1])**2 for j in range(self.k)])
            self.R[i] = np.zeros(self.k)
            self.R[i][np.argmin(self.distance[i])] = 1
    
    def DistortionMeasure(self):
        count = 0
        for i in range(self.N):
            j = np.argmax(self.R[i])
            count += self.distance[i][j]
        self.J = count
    
    def UpdateCenter(self):
        for i in range(self.k):
            r_nk = self.R.T[i]
            self.center[i] = (r_nk@self.data)/np.sum(r_nk)
    
    def Plot_Data(self, axs):
        for i in range(self.N):
            j = np.argmax(self.R[i])
            axs.scatter(self.data[i][0], self.data[i][1], color = self.colors[j], marker = 'o', s = 40)
        for j in range(self.k):
            axs.scatter(self.center[j][0], self.center[j][1], color = self.colors[j], marker = 'x', s = 100)
        axs.spines['top'].set_visible(False)
        axs.spines['right'].set_visible(False)
        return axs

    def Initialise(self):
        self.DistributeData()
        self.DistortionMeasure()
        self.J_.append(self.J)

    def Iterate(self):
        self.UpdateCenter()
        self.DistributeData()
        self.DistortionMeasure()
        self.J_.append(self.J)
    
    def Run(self):
        self.Initialise()
        for i in range(self.iter):
            self.Iterate()

