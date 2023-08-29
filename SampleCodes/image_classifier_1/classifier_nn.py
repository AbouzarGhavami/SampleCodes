# Author: Abouzar Ghavami
# Email: ghavamip@gmail.com


import tensorflow as tf
from tensorflow import keras
from matplotlib import pyplot as plt
import pandas as pd
import pickle

with open('farsi_numeric_train', 'rb') as handle:
    X_train_dict = pickle.load(handle)
with open('farsi_numeric_test', 'rb') as handle:
    X_test_dict = pickle.load(handle)

X_train_full = np.array()

(X_train_full, y_train_full), (X_test, y_test) = fashion_mnist.load_data()

X_valid = X_train_full[:5000] / 255.0
X_train = X_train_full[5000:] / 255.0

y_valid = y_train_full[:5000]
y_train = y_train_full[5000:]

class_names = ['T-shirt/Top', 'Trouser', 'Pullover', 'Dress', 'Coat', \
              'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

#print(class_names[y_train[1]])
#plt.imshow(X_train_full[1], cmap = 'gray', vmin = 0, vmax = 255)
method = 'relu'
print(method)

model = keras.models.Sequential()
model.add(keras.layers.Flatten(input_shape = [28, 28]))
model.add(keras.layers.Dense(200, activation = method))
model.add(keras.layers.Dense(100, activation = method))
model.add(keras.layers.Dense(10, activation = 'softmax'))

model.compile(loss = 'sparse_categorical_crossentropy', optimizer = 'adam', \
             metrics = ['accuracy'])

fitted = model.fit(X_train, y_train, epochs = 20, validation_split = 0.2)

model.evaluate(X_test, y_test)

df = pd.DataFrame(fitted.history)
df.plot(figsize = (8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1)

X_new = X_test[:3]
y_proba = model.predict(X_new)
y_proba.round(2)

keras.utils.plot_model(model, "my_first_model_with_shape_info.png", show_shapes=True)

plt.show()
