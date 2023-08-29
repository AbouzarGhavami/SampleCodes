# Author: Abouzar Ghavami
# Email: ghavamip@gmail.com


import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras as keras

def pipeline(X, train_percentage = 0.8, val_percentage=0.1, test_percentage = 0.1, batch_size = 32,
             train_window_length = 50, test_window_length = 10):
#Procedure:
#dataset = tf.data.Dataset.range(7).window(3, 1, 1, True)
#for window in dataset:
#    print(list(window.as_numpy_iterator()))
#[0, 1, 2]
#[1, 2, 3]
#[2, 3, 4]
#[3, 4, 5]
#[4, 5, 6]

#dataset = tf.data.Dataset.range(10).repeat(3)
#print(list(dataset)) # repeats of [1, ..., 10] three times each element is tensor

#    fn = './timeseries.csv'

#    df = pd.read_csv(fn, dtype = object)

#   X = df.data.astype('float64')

    train_length = int(len(X) * train_percentage)
    val_length = int(len(X) * val_percentage)
    test_length = int(len(X) * test_percentage)

    total_window_length = train_window_length + test_window_length
    
    train_dataset = tf.data.Dataset.from_tensor_slices(X[:train_length])
    val_dataset = tf.data.Dataset.from_tensor_slices(X[train_length : train_length + val_length])
    test_dataset = tf.data.Dataset.from_tensor_slices(X[train_length + val_length : ])

    train_dataset_windowed = train_dataset.window(total_window_length, shift = 1, stride = 1, drop_remainder=True)
    val_dataset_windowed = val_dataset.window(total_window_length, shift = 1, stride = 1, drop_remainder=True)
    test_dataset_windowed = test_dataset.window(total_window_length, shift = 1, stride = 1, drop_remainder=True)

    # for window in dataset:
    #     print(list(window.as_numpy_iterator()))

    train_dataset_batch_inside = train_dataset_windowed.flat_map(lambda window: window.batch(total_window_length))
    val_dataset_batch_inside = val_dataset_windowed.flat_map(lambda window: window.batch(total_window_length))
    test_dataset_batch_inside = test_dataset_windowed.flat_map(lambda window: window.batch(total_window_length))

    # print(list(train_dataset_batch_inside))

    train_dataset_batch_shuffled = train_dataset_batch_inside.shuffle(10000).batch(batch_size)
    val_dataset_batch_shuffled = val_dataset_batch_inside.shuffle(10000).batch(batch_size)
    test_dataset_batch_shuffled = test_dataset_batch_inside.shuffle(10000).batch(batch_size)

    # print(list(train_dataset_shuffled))

    train_dataset_split = train_dataset_batch_shuffled.map(lambda windows: 
                                                       (windows[:, :train_window_length], windows[:, train_window_length:]) )
    val_dataset_split = val_dataset_batch_shuffled.map(lambda windows: 
                                                       (windows[:, :train_window_length], windows[:, train_window_length:]) )
    test_dataset_split = test_dataset_batch_shuffled.map(lambda windows: 
                                                       (windows[:, :train_window_length], windows[:, train_window_length:]) )

    train_dataset_prefetched = train_dataset_split.prefetch(1)
    val_dataset_prefetched = val_dataset_split.prefetch(1)
    test_dataset_prefetched = test_dataset_split.prefetch(1)

    return [train_dataset_prefetched, val_dataset_prefetched, test_dataset_prefetched]
