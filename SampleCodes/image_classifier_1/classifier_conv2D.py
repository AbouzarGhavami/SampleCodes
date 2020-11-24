# Author: Abouzar Ghavami
# Email: ghavamip@gmail.com
# This code is protected by copyright laws in US.
# Please do not reuse in any format without permission of Abouzar Ghavami.

import tensorflow.keras as keras
from keras import backend as K
from matplotlib import pyplot as plt
import time

print(time.ctime())

with open('train_images', 'rb') as handle:
    train_images = pickle.load(handle)
with open('train_labels', 'rb') as handle:
    train_labels = pickle.load(handle)
with open('test_images', 'rb') as handle:
    test_images = pickle.load(handle)
with open('test_labels', 'rb') as handle:
    test_labels = pickle.load(handle)

train_images = train_images/255
test_images = test_images/255

#batch_size = 32
num_classes = 10
epochs = 500

# input image dimensions
img_rows, img_cols = 28, 28

if K.image_data_format() == 'channels_first':
    x_train = train_images.reshape(train_images.shape[0], 1, img_rows, img_cols)
    x_test = test_images.reshape(test_images.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = train_images.reshape(train_images.shape[0], img_rows, img_cols, 1)
    x_test = test_images.reshape(test_images.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

model = keras.models.Sequential()

model.add(keras.layers.Conv2D(30, 7, \
                              activation = 'relu',  #tf.keras.layers.LeakyReLU(alpha=0.1), 
                              padding = 'same', input_shape = (img_rows, img_cols, 1)))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.AveragePooling2D(pool_size = 2))

model.add(keras.layers.Conv2D(60, 7, activation = 'relu', padding = 'same'))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.AveragePooling2D(pool_size = 2))

model.add(keras.layers.Conv2D(256, 5, activation = tf.keras.layers.LeakyReLU(alpha=0.1), padding = 'valid'))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.MaxPooling2D(pool_size = 2))

model.add(keras.layers.Flatten())

model.add(keras.layers.Dense(50, activation = 'tanh'))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Dropout(0.5))

model.add(keras.layers.Dense(20, activation = 'tanh'))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Dropout(0.5))

model.add(keras.layers.Dense(num_classes, activation = 'softmax'))

model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])

checkpoint_cb = keras.callbacks.ModelCheckpoint("my_keras_model.h5", save_best_only=True)

csv_logger = keras.callbacks.CSVLogger('training.log')
my_callbacks = [#keras.callbacks.ModelCheckpoint(filepath='cfar10_models/model.{epoch:02d}-{val_loss:.2f}.h5'),\
               csv_logger, checkpoint_cb]
history = model.fit(x_train, train_labels, \
                    epochs = epochs, callbacks=my_callbacks, validation_split = 0.2)
model = keras.models.load_model("my_keras_model.h5")
y_proba = model.predict(x_test)
y_proba.round(2)
print(y_proba)