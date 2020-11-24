# Author: Abouzar Ghavami
# Email: ghavamip@gmail.com
# This code is protected by copyright laws in US.
# Please do not reuse in any format without permission of Abouzar Ghavami.

import numpy as np
import copy
import matplotlib.pyplot as plt
import random
import math

class Kmeans():

    ###########################
    ## Max distance point #####
    ###########################
    def MaxDistancePoint(self, x, data):
        K = len(data[0])
        y = data[0]
        distmax = 0
        for x1 in data:
            dist = sum([(x[j] - x1[j])**2 for j in range(K)])
            if (dist > distmax):
                distmax = dist
                y = x
        return y
    
    ###########################
    ## Max distance seeding ###
    ###########################
    def MaxDistanceSeeding(self, data, Clusternumber):

        x0 = data[0]
        K = len(x0)
        N = len(data)
        m = [sum([x[i] for x in data])/N for i in range(K)]

        ClusterSeeds = list()
        distmax = 0
        y = x0
        for x1 in data:
            dist = sum([(m[j] - x1[j])**2 for j in range(K)])
            if (dist > distmax):
                distmax = dist
                y = x1
        ClusterSeeds.append(y)

        for i in range(Clusternumber):
            distmax = 0
            c = data[0]
            for x in data:
                dist = 0
                found = 0
                for y in ClusterSeeds:
                    if ((found == 0) \
                        & (sum([y[i] - x[i] for i in range(K)]) != 0)):
                        dist = dist + sum([(x[j] - y[j])**2 for j in range(K)])
                    else:
                        found = 1

                if ((found == 0) & (dist > distmax)):
                    distmax = dist
                    c = x

            ClusterSeeds.append(c)

        return ClusterSeeds
    
    ###########################
    ## function k++seeding ####
    ###########################
    def kplusplusseeding(self, data, Clusternumber):
        x0 = [ 0 for i in range(len(data[0])) ]
        K = len(x0)
        N = len(data)
        cmean = [x0 for i in range(Clusternumber)]
        m = [0 for j in range(K)]
        for j in range(K):
            m[j] = sum([x[j] for x in data])/N
        #print(m)
        x1 = data[0]
        cmean[0] = self.MaxDistancePoint(m, data)            
        cmean[1] = self.MaxDistancePoint(cmean[0], data)
        #print(cmean[0])
        data2 = list()
        for x in data:
            data2.append(x)
    
        for i in range(2, Clusternumber):
            data1 = list()
            for x in data2:
                if (sum([x[k] - cmean[i-1][k] for k in range(K)]) != 0):
                    data1.append(x)
            dist2 = [0 for j in range(len(data1))]
            for j in range(len(dist2)):
                dist2[j] = math.sqrt(sum([(cmean[i - 1][k] - data1[j][k])**2 \
                                     for k in range(K)] ))
                #print(dist[j])
            sumdist2 = sum([dist2[j]**2 for j in range(len(dist2))])
            p = [(dist2[j]**2)/sumdist2 for j in range(len(dist2))]
            P = [0 for j in range(len(p) + 1)]
            #P[0] = p[0]
            for j in range(1, len(P)):
                P[j] = P[j - 1] + p[j - 1]
            print(P[len(P) - 1])
            r = np.random.uniform()
            for j in range(1, len(P)):
                if ((r > P[j - 1]) & (r < P[j])):
                    cmean[i] = [data1[j - 1][k] for k in range(K)]
            data2 = list()
            for x in data1:
                data2.append(x)

        #print('seed = ', cmean)       
        return cmean
    #######################
    ## function kmeans ####
    #######################
    def Kmeans(self, data, Clusternumber, **kwargs):

        mode = 'maxdistance'
        K = len(data[0])
        x0 = [0 for i in range(K)]

        Clustered = [-1 for i in range(len(data))]
        Clusterlength = [1 for i in range(Clusternumber)]
        Clustermeans = [x0 for i in range(Clusternumber)]
    
        if mode == 'kmeans++':
            Clustermeans = self.kplusplusseeding(data, Clusternumber)

        if ('mode' in kwargs.keys()):
            mode = kwargs.get('mode')

        if mode == 'random':
            print('random mode')
            samples = random.sample(range(len(data)), Clusternumber)
            for i in range(Clusternumber):
                Clustermeans[i] = [data[samples[i]][j] for j in range(K)]

        if mode == 'maxdistance':
            Clustermeans = self.MaxDistanceSeeding(data, Clusternumber)
        
        swap = 1
        n = 0
        while (n < 100) & (swap == 1):
            n = n + 1
            swap = 0
            
            for i in range(len(data)):
                cmean = [x0 for i in range(Clusternumber)]
                for j in range(Clusternumber):
                    if (Clustered[i] == j):
                        cmean[j] = [Clustermeans[j][k] \
                                    for k in range(len(cmean[j]))]
                    else:
                        cmean[j] = [( Clustermeans[j][k] * Clusterlength[j] \
                                        + data[i][k]) / (Clusterlength[j] + 1)\
                                    for k in range( K ) ]
                    
                ci = 0
                distc0 = math.sqrt( sum([(data[i][j] - cmean[0][j])**2 \
                            for j in range(K)]) )
                for c in range(1, Clusternumber):
                    distcj = math.sqrt( sum( [ (data[i][j] - cmean[c][j])**2 \
                            for j in range( K ) ] ) )
                    if distcj < distc0:
                        ci = c
                        distc0 = distcj

                if ci != Clustered[i]:
                    swap = 1
                    
                if (Clustered[i] == -1):
                    Clustered[i] = ci
                    Clusterlength[ci] = Clusterlength[ci] + 1
                    Clustermeans[ci] = [cmean[ci][j] for j in range(K)]
                else:
                    if (ci != Clustered[i]):
                    
                        Clustermeans[Clustered[i]] \
                            = [(Clustermeans[Clustered[i]][j] \
                            * Clusterlength[Clustered[i]] \
                          - data[i][j]) / (Clusterlength[Clustered[i]] - 1) \
                           for j in range(len(data[i])) ]

                        Clustermeans[ci] = [cmean[ci][j] for j in range(len(cmean[ci]))]
                        Clusterlength[Clustered[i]] = Clusterlength[Clustered[i]] - 1
                        Clusterlength[ci] = Clusterlength[ci] + 1
                        Clustered[i] = ci

        print('n = ', n)
        return [cmean, Clustered]
