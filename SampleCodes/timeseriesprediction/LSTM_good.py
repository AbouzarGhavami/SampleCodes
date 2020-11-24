#Author Abouzar Ghavami
#Email: ghavamip@gmail.com
# This code is protected by copyright laws in US.
# Please do not reuse in any format without permission of Abouzar Ghavami.

import numpy as np
import tensorflow.keras as keras
import matplotlib.pyplot as plt
import pandas as pd
from keras.layers.advanced_activations import LeakyReLU

def generate_sequence(v, pred_steps, remain_steps, n_steps):
    series = []
    for i in range(0, len(v) - remain_steps):
        serie = np.array(v[i : i + n_steps + pred_steps])
        series.append(serie)
    return np.array(series)

fn = './timeseries.csv'
df = pd.read_csv(fn, dtype = object)
v = df.values
data = np.array([np.array([float(v[i][4])]) for i in range(len(v))])


epochs = 500
n_steps = 50
remain_steps = 51
pred_steps = 1
series = generate_sequence(data, pred_steps, remain_steps, n_steps)
X_train = series[:int(0.8 * len(series)), :n_steps]
y_train = series[:int(0.8 * len(series)), -pred_steps :, 0]
X_valid = series[int(0.8 * len(series)): int(0.9 * len(series)), :n_steps]
y_valid = series[int(0.8 * len(series)) : int(0.9 * len(series)), -pred_steps :, 0]

X_test, y_test = series[int(0.9 * len(series)):, : n_steps, 0], series[int(0.9 * len(series)):, -pred_steps :, 0]

deep = keras.models.Sequential([
    keras.layers.BatchNormalization(),
    keras.layers.Conv1D(filters = 20, kernel_size = 4, strides = 2, padding = 'valid', input_shape = [None, 1]), 
    keras.layers.LSTM(64, return_sequences = True), 
    keras.layers.LeakyReLU(alpha=0.1),
    keras.layers.Dropout(rate = 0.2),
    keras.layers.LayerNormalization(),
    keras.layers.LSTM(128, return_sequences = True),
    keras.layers.LeakyReLU(alpha=0.1),
    keras.layers.LayerNormalization(),
    keras.layers.LSTM(256, return_sequences = True), 
    keras.layers.LeakyReLU(alpha=0.1),
    keras.layers.LayerNormalization(),
    keras.layers.LSTM(128, return_sequences = True), 
    keras.layers.LeakyReLU(alpha=0.1),    
    keras.layers.LayerNormalization(),    
    keras.layers.LSTM(64, return_sequences = True), 
    keras.layers.LeakyReLU(alpha=0.1),
    keras.layers.LayerNormalization(),
    keras.layers.Dropout(rate = 0.2),
    keras.layers.LayerNormalization(),
    keras.layers.LSTM(pred_steps, activation = 'relu')])
deep.compile(optimizer = 'adam', loss = 'mse')
model_fn = ''.join(['LSTM_', str(epochs), '.hdf5'])
min_val_save = keras.callbacks.ModelCheckpoint(model_fn, save_best_only=True, monitor='val_loss', mode='min')
history = deep.fit(X_train, y_train, validation_data = (X_valid, y_valid), epochs = epochs, batch_size = 32, 
                  callbacks=[min_val_save])
deep = keras.models.load_model(model_fn)
y_pred = deep.predict(X_test)
mse_nn = np.mean(keras.losses.mean_squared_error(y_test, y_pred))
print(mse_nn)
plt.plot([i for i in range(n_steps)], X_test[0])
plt.plot([i for i in range(n_steps,n_steps + 10)], y_test[0])
plt.plot([i for i in range(n_steps,n_steps + 10)], y_pred[0])
plt.show()