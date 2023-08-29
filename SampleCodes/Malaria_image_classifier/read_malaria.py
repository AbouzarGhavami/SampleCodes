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

def Black_White(im):
    for i in range(len(im)):
        for j in range(len(im[0])):
            if im[i][j] == 0:
                im[i][j] = 255
            if im[i][j] >= 120:
                im[i][j] = 255
            if im[i][j] < 120:
                im[i][j] = 0
    return im

start_time = time.time()
print('Start time = ', time.ctime())
P_dir = 'Parasitized'
U_dir = 'Uninfected'
Ps = os.listdir(''.join(['./', P_dir]))
Us = os.listdir(''.join(['./', U_dir]))

fn_p = 'patientid_cellmapping_parasitized.csv'
fn_u = 'patientid_cellmapping_uninfected.csv'
df_p = pd.read_csv(fn_p, header = None, dtype = object)
df_u = pd.read_csv(fn_u, header = None, dtype = object)

Pns = df_p[[0]]
Uns = df_u[[0]]

Pims = []
Uims = []
Pims_grey = []
Uims_grey = []

im_size_x = 50
im_size_y = 50

x = np.random.random()
for i in range(len(Ps)):
    if i % 1000 == 0:
        print(i, ' / ', len(Ps))
    if 'png' in Ps[i]:
        im = cv2.imread(''.join(['./', P_dir, '/', Ps[i]]))
        im1 = cv2.resize(im, (im_size_x, im_size_y))
        grey1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
        Pims.append(im1)

time1 = time.time()
print('Read Parasite files in ', time1 - start_time)

for i in range(len(Us)):
    if i % 1000 == 0:
        print(i, ' / ', len(Us))
    if 'png' in Us[i]:
        im = cv2.imread(''.join(['./', U_dir, '/', Us[i]]))
        im1 = cv2.resize(im, (im_size_x, im_size_y))
        grey1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
        Uims.append(im1)

time2 = time.time()
print('Read Parasite files in ', time2 - time1)

f = open('Malaria.pkl', 'wb')
d = dict()
d['Parasites'] = Pims
d['UnInfected'] = Uims
pkl.dump(d, f)
f.close()


