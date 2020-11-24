# Author: Abouzar Ghavami
# Email: ghavamip@gmail.com
# This code is protected by copyright laws in US.
# Please do not reuse in any format without permission of Abouzar Ghavami.

import numpy as np
import random
import time
import copy

class NN:
    def f(self, x):
        a = 1
        return 1/(1 + np.exp(-a * x))

    def df(self, x):
        a = 1
        return a * np.exp(-a * x)/ (1 + np.exp(-a * x))**2

    #####################
    ### initialize w ####
    #####################
    def initialw(self, L):
        w = list()
        for l in range(len(L) - 1):
            wl = [[np.random.rand() for outputcols in range(L[l + 1])] \
                  for inputrows in range(L[l])]
            for i in range(L[l]):
                wl[i][0] = 0
            w.append(wl)

        return w

    ##########################
    ## feedforward ###########
    ##########################
    def feedforward(self, x, w):
    
        x1 = [1] + list(x)
        yhat = list()
        yhatprime = list()
        #print(yhat)
        for l in range(len(w)):
            yhat1 = [0 for x in range(len(w[l]))]
            yhatprime1 = [0 for x in range(len(w[l]))]
            yhat.append(yhat1)
            yhatprime.append(yhatprime1)
        yhat1 = [0 for x in range(len(w[len(w) - 1][0]))] # add last output
        yhat.append(yhat1)
        yhatprime1 = [0 for x in range(len(w[len(w) - 1][0]))] # add last output
        yhatprime.append(yhatprime1)
    
        for i in range(len(yhat[0])):
            yhat[0][i] = x1[i]
            yhatprime[0][i] = 1
        
        for l in range(len(yhat) - 1):
            yhat[l][0] = 1
            yhatprime[l][0] = 0
            for output in range( 1, len(yhat[l + 1]) ):
                s = 0
                for inputs in range(len(yhat[l])):
                    s = s + yhat[l][inputs] * w[l][inputs][output]
                yhat[l + 1][output] = self.f(s)
                yhatprime[l + 1][output] = self.df(s)
          
        return yhat, yhatprime

    #######################################
    ### Back propagation ##################
    #######################################
    def backpropagate(self, x, y, yhat, yhatprime, w, **kwargs):
        
        cost = 'MSE'
        if ('cost' in kwargs.keys()):
            cost = kwargs.get('cost')
    
        bp = copy.deepcopy(yhatprime)
        
        dJ = [yhat[len(yhat) - 1][i] - y[i - 1] \
              for i in range(1, len(y) + 1)]
        if (cost == 'ln'):
            dJ = [-y[i - 1]/yhat[len(yhat) - 1][i] \
                  + (1 - y[i - 1])/(1 - yhat[len(yhat) - 1][i]) \
                  for i in range(1, len(y) + 1)]

        for o in range(1, len(bp[len(bp) - 1])):
            bp[len(bp) - 1][o] = bp[len(bp) - 1][o] * dJ[o - 1]
        for l in range(len(bp) - 2, -1, -1):
            for i in range(len(bp[l])):
                s = 0
                for o in range(len(bp[l + 1])):
                    s = s + bp[l + 1][o] * w[l][i][o]
                bp[l][i] = yhatprime[l][i] * s
    
        return bp

    ########################
    ## delta omega #########
    ########################
    def deltaw(self, x, y, w, **kwargs):
        
        cost1 = 'MSE'
        if ('cost' in kwargs.keys()):
            cost1 = kwargs.get('cost')
        
        yhat, yhatprime = self.feedforward(x, w)
        dw = copy.deepcopy(w)
        
        bp = self.backpropagate(x, y, yhat, yhatprime, w, cost = cost1)

     
        for l in range(len(w)):
            for i in range(len(w[l])):
                for o in range(len(w[l][i])):
                    dw[l][i][o] = yhat[l][i] * bp[l + 1][o]
    
        return dw

    ########################
    ## Train function ######
    ########################
    def train(self, X, y, Layers, alpha, Nsim, error, **kwargs):
        starttime = time.time()

        costfunction = 'MSE'
        method = 'gradientdescent'
        if ('cost' in kwargs.keys()):
            costfunction = kwargs.get('cost')
        if ('method' in kwargs.keys()):
            method = kwargs.get('method')
            
        print('method = ', method)
        x0 = X[0]
                
        L = [len(x0)] + Layers
        
        for l in range(len(L)):
            L[l] = L[l] + 1
    
        w = self.initialw(L)

        m = copy.deepcopy(w)
        v = copy.deepcopy(w)
        alphat = alpha
        beta1 = 0.9
        beta2 = 0.999
        epsilon = 0.00000001
        
        J = [0 for i in range(Nsim)]
        Jmin = 10000
        wmin = list()
        batchfactor = 0.05
        batchsize = int(len(X) * batchfactor)
        alpha = alpha / batchfactor
        
        for t in range(1, Nsim):
            wk = copy.deepcopy(w)
            R = random.sample(range(len(X)), batchsize)
            for k in R: #range(len(X)):
                deltawx = self.deltaw(X[k], y[k], w, cost = 'ln')
                if (method == 'adam'):
                    vhat2 = 0
                    alphat = alpha * np.sqrt(1 - beta2 ** t)/(1 - beta1 ** t)
                    for l in range(len(w)):
                        for i in range(len(w[l])):
                            for o in range(len(w[l][i])):
                                m[l][i][o] = beta1 * m[l][i][o] \
                                             + (1 - beta1) * deltawx[l][i][o]
                                v[l][i][o] = beta2 * v[l][i][o] \
                                      + (1 - beta2) * deltawx[l][i][o] ** 2
                                vhat2 = vhat2 + v[l][i][o] #/ (1 - beta2 ** t)
                    for l in range(len(w)):
                        for i in range(len(w[l])):
                            for o in range(len(w[l][i])):
                                wk[l][i][o] = wk[l][i][o] \
                                    - alphat * m[l][i][o] \
                                    /(np.sqrt(vhat2) \
                                      + np.sqrt(epsilon * (1 - beta2)) ** t)
                                
                else:
                    for l in range(len(w)):
                        for i in range(len(w[l])):
                            for o in range(len(w[l][i])):
                                wk[l][i][o] = wk[l][i][o] \
                                   - alpha * deltawx[l][i][o]

            w = copy.deepcopy(wk)
            s = 0
            e = 0
            for k in range(len(X)):
                x = X[k]
                yhat, yhatprime = self.feedforward(x, w)
                #print(yhat)
                e = [(yhat[len(yhat) - 1][i + 1] - y[k][i]) ** 2 \
                 for i in range(len(y[k]))]
                if (costfunction == 'ln'):
                    e = [-y[k][i] * np.log( yhat[len(yhat) - 1][i + 1] ) \
                         - (1 - y[k][i]) * np.log( 1 - yhat[len(yhat) - 1][i + 1] ) \
                           for i in range(len(y[k]))] 
            
                s = s + sum(e)
        
            J[t] = s/len(X)
            if t % 10 == 0:
                print('error ', t, ' = ', J[t])
                
            if (J[t] < Jmin):
                Jmin = J[t]
                wmin = copy.deepcopy(w)
                if Jmin < error:
                    break
        
        #w = copy.deepcopy(wmin)

        print('Jmin = ', Jmin)

        print('training elapsed = ', time.time() - starttime)

        return wmin
