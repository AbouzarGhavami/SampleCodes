# Author: Abouzar Ghavami
# Email: ghavamip@gmail.com


import numpy as np
import matplotlib.pyplot as plt

def contourplot(X, y, clf):
    ind0 = []
    ind1 = []
    for i in range(len(y)):
        if y[i] == 1:
            ind1.append(i)
        else:
            ind0.append(i)

    plt.scatter(X[ind0][:, 0], X[ind0][:, 1], color = 'red', marker = '+')
    plt.scatter(X[ind1][:, 0], X[ind1][:, 1], color = 'blue', marker = 'o')
    # create a mesh to plot in
    h = 0.02  #mesh step size
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)

