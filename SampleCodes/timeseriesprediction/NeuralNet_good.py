# Author: Abouzar Ghavami
# Email: ghavamip@gmail.com
# This code is protected by copyright laws in US.
# Please do not reuse in any format without permission of Abouzar Ghavami.

print('Importing packages ...')
import numpy as np
import tensorflow.keras as keras
import matplotlib.pyplot as plt
import pandas as pd

def generate_sequence(v, pred_steps, remain_steps, n_steps):
    series = []
    for i in range(0, len(v) - remain_steps):#, n_steps):
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
pred_steps = 10
series = generate_sequence(data, pred_steps, remain_steps, n_steps)
X_train = series[:int(0.8 * len(series)), :n_steps]
y_train = series[:int(0.8 * len(series)), -pred_steps :, 0]
X_valid = series[int(0.8 * len(series)): int(0.9 * len(series)), :n_steps]
y_valid = series[int(0.8 * len(series)) : int(0.9 * len(series)), -pred_steps :, 0]

X_test, y_test = series[int(0.9 * len(series)):, : n_steps, 0], series[int(0.9 * len(series)):, -pred_steps :, 0]

model = keras.models.Sequential([
keras.layers.Flatten(input_shape = [n_steps, 1]),
    keras.layers.Dense(150, activation = 'relu'),
    keras.layers.LayerNormalization(),
    keras.layers.Dropout(rate = 0.2),
    keras.layers.Dense(100, activation = 'relu'),
    keras.layers.LayerNormalization(),
    keras.layers.Dropout(rate = 0.2),
    keras.layers.Dense(50, activation = 'relu'),
    keras.layers.LayerNormalization(),
    keras.layers.Dropout(rate = 0.2),
    keras.layers.Dense(pred_steps, activation = 'linear')
])
model.compile(optimizer = 'adam', loss = 'mse')
model_fn = ''.join(['NeuralNet_', str(epochs), '.hdf5'])
min_val_save = keras.callbacks.ModelCheckpoint(model_fn, save_best_only=True, monitor='val_loss', mode='min')
history = model.fit(X_train, y_train, validation_data = (X_valid, y_valid), epochs = epochs, batch_size = 32, 
                   callbacks=[min_val_save])

model1 = keras.models.load_model(model_fn)
y_pred = model1.predict(X_test)
mse_nn = np.mean(keras.losses.mean_squared_error(y_test, y_pred))
print(mse_nn)
plt.plot([i for i in range(n_steps)], X_test[0])
plt.plot([i for i in range(n_steps,n_steps + remain_steps)], y_test[-remain_steps:])
plt.plot([i for i in range(n_steps,n_steps + remain_steps)], y_pred[-remain_steps:])
plt.show()