# Author: Abouzar Ghavami
# Email: ghavamip@gmail.com

import time
print('Started at: ', time.ctime())
import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.layers import Embedding

fn = 'drug.csv'

start_time = time.time()
df = pd.read_csv(fn, dtype = object)

reviews = df.review.str.lower().values

num_words = 10000
word_dim = 10
oov_token = 10100

t = Tokenizer(num_words = 10000, oov_token = oov_token)

t.fit_on_texts(reviews)

seq = t.texts_to_sequences(reviews)
seqs = np.asarray([np.asarray(seq[i][:200]) for i in range(len(seq))])
embed = Embedding(num_words, word_dim, input_length=200)


model = keras.models.Sequential()
model.add(embed)
model.compile('rmsprop', 'mse')
output_array = model.predict([seqs[0]])


end_time = time.time()

print('Elapsed time = ', end_time - start_time)
