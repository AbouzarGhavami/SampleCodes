# Author: Abouzar Ghavami
# Email: ghavamip@gmail.com


import cv2
import os
import matplotlib.pyplot as plt
import pickle
import numpy as np

d = dict()
dim = (28, 28)
train_images = []
train_labels = []
for i in range(10):
    path = ''.join(['./', str(i)])
    a = os.listdir(path)
    d[i] = []
    for x in a:
        image = cv2.imread(''.join([path, '/', x]), cv2.IMREAD_GRAYSCALE)
        resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
        d[i].append(resized)
        train_images.append(resized)
        train_labels.append(i)
d_test = dict()
dim = (28, 28)
test_images = []
test_labels = []
for i in range(10):
    path = ''.join(['./test/', str(i)])
    a = os.listdir(path)
    d_test[i] = []
    for x in a:
        image = cv2.imread(''.join([path, '/', x]), cv2.IMREAD_GRAYSCALE)
        resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
        d_test[i].append(resized)
        test_images.append(resized)
        test_labels.append(i)
with open('train_images', 'wb') as handle:
    pickle.dump(np.asarray(train_images), handle, protocol = pickle.HIGHEST_PROTOCOL)
with open('train_labels', 'wb') as handle:
    pickle.dump(np.asarray(train_labels), handle, protocol = pickle.HIGHEST_PROTOCOL)
with open('test_images', 'wb') as handle:
    pickle.dump(np.asarray(test_images), handle, protocol = pickle.HIGHEST_PROTOCOL)
with open('test_labels', 'wb') as handle:
    pickle.dump(np.asarray(test_labels), handle, protocol = pickle.HIGHEST_PROTOCOL)
