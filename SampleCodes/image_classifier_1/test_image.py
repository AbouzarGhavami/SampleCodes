# Author: Abouzar Ghavami
# Email: ghavamip@gmail.com

import cv2
import os
import matplotlib.pyplot as plt
import pickle
import numpy as np
import tensorflow.keras as keras
import keras.backend as K

dim = (50, 50)
fn = './test/test.jpg'
image = cv2.imread(fn, cv2.IMREAD_GRAYSCALE)
#resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
for k in range(len(image)):
    for l in range(len(image[0])):
        if image[k][l] < 128:
            image[k][l] = 0
        else:
            image[k][l] = 1

test_image = np.asarray([image])
img_rows, img_cols = len(test_image[0]), len(test_image[0][0])

if K.image_data_format() == 'channels_first':
    test_image = test_image.reshape(1, 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    test_image = test_image.reshape(1, img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)
    
model = keras.models.load_model("my_keras_model_conv2D.h5")

y_proba = model.predict(test_image)
y_proba.round(2)
print(y_proba)

plt.imshow(resized)
plt.show()
