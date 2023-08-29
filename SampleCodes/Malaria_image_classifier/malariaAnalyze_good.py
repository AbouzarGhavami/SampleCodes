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

start_time = time.time()
print('Start time = ', time.ctime())

f = open('Malaria.pkl', 'rb')
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

batch_size = 32
num_classes = 2
epochs = 100

# input image dimensions
img_rows, img_cols = 28, 28

#if K.image_data_format() == 'channels_first':
#    x_train = train_images.reshape(train_images.shape[0], 3, img_rows, img_cols)
#    x_test = test_images.reshape(test_images.shape[0], 3, img_rows, img_cols)
#    input_shape = (1, img_rows, img_cols)
#else:
#    x_train = train_images.reshape(train_images.shape[0], img_rows, img_cols, 3)
#    x_test = test_images.reshape(test_images.shape[0], img_rows, img_cols, 3)
#    input_shape = (img_rows, img_cols, 1)

model = keras.models.Sequential()

model.add(keras.layers.Conv2D(64, 3, activation = 'relu', padding = 'SAME', input_shape = (img_rows, img_cols, 3)))

model.add(keras.layers.MaxPooling2D(pool_size = 2))

#model.add(keras.layers.Conv2D(256, 3, activation = 'relu', padding = 'same'))

#model.add(keras.layers.MaxPooling2D(pool_size = 2))

model.add(keras.layers.Conv2D(128, 3, activation = 'relu', padding = 'same'))

model.add(keras.layers.MaxPooling2D(pool_size = 2))

model.add(keras.layers.Flatten())

model.add(keras.layers.Dense(200, activation = 'relu'))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Dropout(0.5))

model.add(keras.layers.Dense(100, activation = 'relu'))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Dropout(0.5))

model.add(keras.layers.Dense(num_classes, activation = 'softmax'))

model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
csv_logger = keras.callbacks.CSVLogger('malaria_models/training.log')
my_callbacks = [keras.callbacks.ModelCheckpoint(filepath='malaria_models/alex_net/model.{epoch:02d}-{val_accuracy:.2f}.h5'),\
               csv_logger]
history = model.fit(train_images, train_labels, \
                    batch_size = batch_size, validation_split = 0.2, \
                    epochs = epochs, callbacks=my_callbacks)

model.save('my_model.h5')

y_pred = model.predict(test_images)
y_class = [np.argmax(y_pred[i]) for i in range(len(y_pred))]
cm = confusion_matrix(y_class, test_labels)
#grey = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
#bw = copycorrect(grey)
#cv2.imshow('image', im1)
#cv2.imshow('Converted Image', bw)
#cv2.waitKey()
