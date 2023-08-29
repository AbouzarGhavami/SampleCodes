# Author: Abouzar Ghavami
# Email: ghavamip@gmail.com


import numpy as np
import matplotlib.pyplot as plt

class ML:
    def __init__(self, Omega1, Omega2):
        self.Omega1 = Omega1
        self.Omega2 = us
        self.accs = accs
        
    def dot(self, v1, v2):
        s = 0
        if (len(v1) != len(v2)):
            print('Can not perform dot product!')
            return 0
        for i in range(len(v1)):
            s = s + float(v1[i]) * float(v2[i])
        return s

    def Perceptron(self, omega1, omega2, **kwargs):
        N = 50 * (len(omega1) + len(omega2))
        ro = 0.2
        if ('Iter' in kwargs.keys()):
            N = kwargs.get('Iter')
            print('N = ', N)
        if 'rho' in kwargs.keys():
            ro = kwargs.get('rho') 

        #print(type(Omega1))

        Omega1 = list()
        Omega2 = list()
        for s in omega1:
            s1 = [0 for x in range(len(s) + 1)]
            s1[len(s)] = 1
            s1[0 : len(s)] = s
            Omega1.append(s1)
            #print(s1)

        for s in omega2:
            s1 = [0 for x in range(len(s) + 1)]
            s1[len(s)] = 1
            s1[0 : len(s)] = s
            Omega2.append(s1)

        s = list(Omega1).pop()
        w = [0 for x in range(len(s))]
        wmin = w
        mismatchmin = len(Omega1) + len(Omega2)

        iter = 0
        
        while(iter < N):
            mismatchnumber = 0
            for s in Omega1:
                s1 = [float(i) * ro for i in s]
                #print(self.dot(w, s1))
                sum = np.dot(s, w)
                for i in range(len(s)):
                    sum = sum + float(s[i]) * float(w[i])
                if sum <= 0:
                    mismatchnumber = mismatchnumber + 1 
                    w = [w[i] + ro * float(s[i]) for i in range(len(w))]
            for s in Omega2:
                #s1 = [ro * float(i) for i in s]
                sum = np.dot(s, w)
                if sum > 0:
                    mismatchnumber = mismatchnumber + 1 
                    w = [w[i] - ro * float(s[i]) for i in range(len(w))]
            if iter > N /2:
                if mismatchnumber < mismatchmin:
                    mismatchmin = mismatchnumber
                    wmin = w
            iter = iter + 1
        #print(mismatchmin)
        return wmin

    def ClassSeparation(self, Omega1, Omega2, w):

        ViolationClass1 = list()
        SatisfiedClass1 = list()
        ViolationClass2 = list()
        SatisfiedClass2 = list()

        for s in Omega1:
            d = np.dot(np.append(s, 1), w)
            if (d < 0):
                ViolationClass1.append(s)
            else:
                SatisfiedClass1.append(s)
    
        for s in Omega2:
            d = np.dot(np.append(s, 1), w)
            if (d > 0):
                ViolationClass2.append(s)
            else:
                SatisfiedClass2.append(s)

        return ViolationClass1, SatisfiedClass1, \
               ViolationClass2, SatisfiedClass2

    def PerceptronClassify(self, Omega1, Omega2, sample):
        samp = list(sample)
        print('sample = ', samp)
        w = self.Perceptron(Omega1, Omega2)
        print('w = ', w)
        if (len(samp) == len(w) - 1):
            s = [0 for x in range(len(samp) + 1)]
            s[0 : len(samp)] = samp
            s[len(s) - 1] = 1
        if self.dot(w, s) > 0:
            return 1
        else:
            return 2

    def Sigmoid(self, w, x):
        f = 1/(1 + np.exp(-1 * self.dot(w, x)))
        return f

    def ClassPlot(self, Omega1, Omega2):
        Omega1x = list()
        Omega1y = list()
        for s in Omega1:
            Omega1x.append(s[0])
            Omega1y.append(s[1])
            
        Omega2x = list()
        Omega2y = list()
        for s in Omega2:
            Omega2x.append(s[0])
            Omega2y.append(s[1])

        plt.plot(Omega1x, Omega1y, 'ro')
        plt.plot(Omega2x, Omega2y, 'bx')
        plt.text(Omega1x[0], Omega1y[1], 'Class1')
        plt.text(Omega2x[0], Omega2y[1], 'Class2')

    def HyperPlanePlot(self, w, xleft = None, xright = None, label = None):        
        xl = 0
        xr = 100
        labeltext = 'w'
        if xleft is not None:
            xl = int( np.floor(xleft * 100) )
        if xright is not None:
            xr = int(np.floor(xright * 100))
        if label is not None:
            labeltext = label
        x = [i/100 for i in range(xl, xr)]
        y = [-w[1] * i / w[2] - w[0]/w[2] for i in x]

        plt.plot(x, y, 'b')
        plt.text(x[0], y[0], labeltext)
