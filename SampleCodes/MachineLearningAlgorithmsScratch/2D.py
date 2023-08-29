# Author: Abouzar Ghavami
# Email: ghavamip@gmail.com


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
from sklearn.decomposition import PCA

def reduceEIG(data, M):
    # Each data point is array of length L
    # Data is a list of arrays
    L = len(data[0]) # Number of data
    if (L >= M):
        
        N = len(data)    # dimension of data

    mymat = [[data[j][i] for i in range(L)] for j in range(N)]
    print(mymat)
        D = np.matmul(np.transpose(data), data )

        eig, vec = np.linalg.eig(D)

        ind = np.argsort(-eig)

        v = list()
        l = list()

        for i in range(M):
            l.append(eig[ind[i]])
            v.append(vec[:, ind[i]])

        print(eig, vec)
        d1 = list()
        for i in range(N):
            x = data[i]
            xp = [np.dot(x, v[j]) for j in range(M)]
            d1.append(xp)
        
        print(eig, ", ", ind)

        return eig, vec, np.array(d1)

    else:
        print('Reduced dimension is larger than primary dimension!')
        return 0

def reduceSVD(data, M):
    L = len(data[0]) # Number of data
    if (L >= M):
        
        N = len(data)    # dimension of data
        u, s, vec = np.linalg.svd(data)
        ind = np.argsort(-s)

        v = list()
        l = list()

        for i in range(M):
            l.append(s[ind[i]])
            v.append(vec[ind[i]])

        print(u, s ** 2, v)
        d2 = list()
        for i in range(N):
            x = data[i]
            xp = [np.dot(x, v[j]) for j in range(M)]
            d2.append(xp)

        return d2

    else:
        print('Reduced dimension is larger than primary dimension!')
        return 0

def reduceSVDmat(data, M):
    u, s, v = np.linalg.svd(data)
    ind = np.argsort(-s)
    eig, vec = np.linalg.eig( np.matmul(np.transpose(data), data) )

    N = len(data)
    L = len(data[0])

    d3 = s[ind[0]] * np.array(u[:, ind[0]])

    for i in range(1, M):
        a = s[ind[i]] * np.array(u[:, ind[i]])
        d3 = np.column_stack(d3, )

    return d3
    
data = [[1, 3, 10],
         [2, -2, 12],
         [3, 6, 17], [4, -5, -2], [5, 4, 10]]

data = np.random.rand(10000, 10)
mu = [np.mean(data[i]) for i in range(len(data))]
data_centered = [data[i] - mu[i] for i in range(len(data))]
starttime = time.time()
d2 = reduceEIG(data_centered, 2)
d22 = d2[2]
print('Fastest One 2D, EIG time = ', time.time() - starttime)

starttime = time.time()
d3 = reduceEIG(data_centered, 3)
d32 = d3[2]
print('Fastest One 3D, EIG time = ', time.time() - starttime)

starttime = time.time()
pca = PCA(n_components = 2)
X2D = pca.fit_transform(data_centered)
print('Sklearn time = ', time.time() - starttime)
starttime = time.time()
d2 = reduceSVD(data, 2)
print('SVD time = ', time.time() - starttime)

starttime = time.time()
d3 = reduceSVDmat(data, 2)
print('SVDmat time = ', time.time() - starttime)

x = [x[0] for x in d22]
y = [x[1] for x in d22]

plt.plot(x, y, 'r+')

x1 = [x[0] for x in data]
y1 = [x[1] for y in data]
z1 = [x[2] for z in data]

fig = plt.figure()
Axes3D = fig.gca(projection='3d')
Axes3D.scatter(x1, y1, z1, zdir='z', s=20, c=None, depthshade=True)

plt.show()
