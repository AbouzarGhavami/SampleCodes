# Author: Abouzar Ghavami
# Email: ghavamip@gmail.com
# This code is protected by copyright laws in US.
# Please do not reuse in any format without permission of Abouzar Ghavami.

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras as keras


shift = 1
stride = 1 # distance between elements
batch_size = 32
train_window_length = 15
test_window_length = 5

total_window_length = train_window_length + test_window_length

fn1 = './timeseries1.csv'
fn2 = './timeseries2.csv'

df1 = pd.read_csv(fn1, dtype = object)
df2 = pd.read_csv(fn2, dtype = object)
x1 = df1.where(df1["Date"] >= '2019').dropna().data.astype('float64')
x2 = df2.where(df2["Date"] >= '2019').dropna().data.astype('float64')

x1_vals = x1.values
x2_vals = x2.values

X_train = [ np.concatenate(
                                [
                                    x1_vals[i : i + train_window_length],
                                    x2_vals[i : i + train_window_length]
                                ]
                               )                 
               for i in range(len(x1_vals - total_window_length)) ]
y_train = [x1_vals[i +  train_window_length : i + train_window_length + test_window_length]
           for i in range(len(x1_vals - total_window_length))]

Dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
train_dataset = tf.data.Dataset.tf.convert_to_tensor((X_train, y_train))

X1 = tf.data.Dataset.from_tensor_slices(X_train)
X2 = tf.data.Dataset.from_tensor_slices(x2)

X1_windowed = X1.window(total_window_length, shift = 1, stride = 1, drop_remainder=True)
X2_windowed = X2.window(total_window_length, shift = 1, stride = 1, drop_remainder=True)
for i in range(len(X1_windowed):
combined_dataset = tf.concat([X1_windowed, X2_windowed], 1)

X1_batch = X1_windowed.flat_map(lambda window: window.batch(total_window_length))
X2_batch = X2_windowed.flat_map(lambda window: window.batch(total_window_length))

X1_shuffle = X1_batch.shuffle(10000).batch(batch_size)
X2_shuffle = X2_batch.shuffle(10000).batch(batch_size)

X1_split = X1_shuffle.map(lambda windows: (windows[:, :train_window_length], windows[:, train_window_length:]) )
X2_split = X2_shuffle.map(lambda windows: (windows[:, :], []) )
