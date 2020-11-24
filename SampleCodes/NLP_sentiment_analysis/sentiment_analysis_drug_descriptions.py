# Author: Abouzar Ghavami
# Email: ghavamip@gmail.com
# This code is protected by copyright laws in US.
# Please do not reuse in any format without permission of Abouzar Ghavami.

import time
print('Started at: ', time.ctime())
import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding

fn = 'train_F3WbcTw.csv'
fn_test = 'test_tOlRoBf.csv'

start_time = time.time()
df = pd.read_csv(fn, dtype = object)
df_test = pd.read_csv(fn_test, dtype = object)

review_drug = df['text'] + ' ' + df['drug']
reviews = review_drug.str.lower().values

sentiments = df.sentiment.astype(int).values

review_drug_test = df_test['text'] + ' ' + df_test['drug'] + ' ' + df_test['drug']
reviews_test = review_drug_test.str.lower().values

num_classes = len(set(sentiments))

vocab_size = 20000
word_dim = 10
oov_token = "oov_token"
num_oov_buckets = 1

t = Tokenizer(num_words = vocab_size, oov_token = "oov")

t.fit_on_texts(reviews)

seq = t.texts_to_sequences(reviews)
seqs = np.asarray([np.asarray(seq[i][:400]) for i in range(len(seq))])
seqs = pad_sequences(seqs)

inputs = keras.layers.Input(shape=[None])
mask = keras.layers.Lambda(lambda inputs: K.not_equal(inputs, 0))(inputs)

embed = Embedding(num_words, word_dim, input_length=400, mask_zero = True)

K = keras.backend
inputs = keras.layers.Input(shape=[None])
mask = keras.layers.Lambda(lambda inputs: K.not_equal(inputs, 0))(inputs)
z = keras.layers.Embedding(vocab_size + num_oov_buckets, word_dim)(inputs)
z = keras.layers.GRU(128, return_sequences=True)(z, mask=mask)
z = keras.layers.GRU(128)(z, mask=mask)
outputs = keras.layers.Dense(num_classes, activation="sigmoid")(z)
model = keras.Model(inputs=[inputs], outputs=[outputs])

loss_fn = keras.losses.SparseCategoricalCrossentropy()
model.compile(loss = loss_fn, optimizer = 'adam', metrics = ['accuracy'])

fitted = model.fit(seqs, sentiments, epochs = 50, validation_split=0.2)

tests = [['b' for i in range(200)], 'The drug was fantastic!', 'The drug was awful. Terribly bad.', 'Expected better drug! Could improve substances. Totally nice medicine.']
testseq = t.texts_to_sequences(tests)
testseqs = pad_sequences(testseq)
preds = model.predict(testseqs)

model.compile('rmsprop', 'mse')
output_array = model.predict([seqs[0]])


end_time = time.time()

print('Elapsed time = ', end_time - start_time)