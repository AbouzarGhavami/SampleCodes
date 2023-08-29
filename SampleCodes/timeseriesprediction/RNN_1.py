# Author: Abouzar Ghavami
# Email: ghavamip@gmail.com


import numpy as np
import tensorflow.keras as keras
import matplotlib.pyplot as plt
import pandas as pd
import pipeline1_good as pipeline

epochs = 5000
batch_size = 32

fn = './timeseries.csv'

df = pd.read_csv(fn, dtype = object)

X = df.CLOSE.astype('float64')

train_percentage = 0.8
validate_percentage = 0.19
test_percentage = 0.01

train_window_length = 100
test_window_length = 10

[train, val, test] = pipeline.pipeline(X, train_percentage = train_percentage, 
                                       val_percentage = validate_percentage, test_percentage = test_percentage, 
                                       batch_size = batch_size,
                                       train_window_length = train_window_length, 
                                       test_window_length = test_window_length)
x_train = X_train.reshape(-1, 1)

deep = keras.models.Sequential([
    keras.layers.Flatten(input_shape = [len(X_train[0]), 1]),
    keras.layers.BatchNormalization(),
    keras.layers.Conv1D(filters = 20, kernel_size = 4, strides = 2, padding = 'valid'), #input_shape = [None, 1]), 
    keras.layers.LSTM(40, return_sequences = True, input_shape = [None, 1]),
    keras.layers.Dropout(rate = 0.2),
    keras.layers.LayerNormalization(),
    keras.layers.LSTM(40, return_sequences = True), 
    keras.layers.Dropout(rate = 0.2),
    keras.layers.LayerNormalization(),
    keras.layers.LSTM(test_window_length, activation = 'linear')])
deep.compile(optimizer = 'adam', loss = 'mse')
min_val_save = keras.callbacks.ModelCheckpoint(''.join(['Ex11_', str(epochs), '.hdf5']), save_best_only=True, monitor='val_loss', mode='min')
history = deep.fit(train, validation_data = val, epochs = epochs, batch_size = 32, 
                  callbacks=[min_val_save])
y_pred = deep.predict(test)
mse_deep = np.mean(keras.losses.mean_squared_error(y_test, y_pred))
print(mse_deep)
series = generate_time_series(1, n_steps + 10)
X_new, Y_new = series[:, :n_steps], series[:, n_steps:]
Y_pred = deep.predict(X_new)

plt.plot([i for i in range(n_steps + 10)], series[0])
plt.plot([i for i in range(n_steps,n_steps + 10)], Y_pred[0])
plt.show()
