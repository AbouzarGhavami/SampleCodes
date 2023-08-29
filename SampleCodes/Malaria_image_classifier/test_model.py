# Author: Abouzar Ghavami
# Email: ghavamip@gmail.com


import os
import pandas as pd
import cv2
import numpy as np
import tensorflow.keras as keras
from keras import backend as K
from matplotlib import pyplot as plt
import sys
import time
import pickle as pkl
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import plot_precision_recall_curve
from sklearn.metrics import roc_curve, auc

time1 = time.time()
print('Start time = ', time.ctime())

f = open('Malaria_grey.pkl', 'rb')
im_dict = pkl.load(f)
Ps = im_dict['Parasites']
Us = im_dict['UnInfected']
Total_images = Ps + Us
Plabels = [1 for i in range(len(Ps))]
Ulabels = [0 for i in range(len(Us))]
Total_labels = Plabels + Ulabels

train_images = []
train_labels = []
test_images = []
test_labels = []
ratio = 0.9
for i in range(len(Total_images)):
    x = np.random.uniform()
    if x < ratio:
        train_images.append(Total_images[i])
        train_labels.append(Total_labels[i])
    else:
        test_images.append(Total_images[i])
        test_labels.append(Total_labels[i])
        
train_images = np.asarray(train_images)/255
test_images = np.asarray(test_images)/255
train_labels = np.asarray(train_labels)
test_labels = np.asarray(test_labels)

model = keras.models.load_model('./malaria_models/simple_grey/model.27-0.86.h5')

y_pred = model.predict(test_images)
y_class = [np.argmax(y_pred[i]) for i in range(len(y_pred))]
cm = confusion_matrix(y_class, test_labels)

print(cm)

accuracy = 0.5 * (cm[0][0]/(cm[0][0] + cm[0][1]) + cm[1][1]/(cm[1][1] + cm[1][0]))
print('accuracy = ', accuracy)
time2 = time.time()
print('Elapsed time = ', time2 - time1)
