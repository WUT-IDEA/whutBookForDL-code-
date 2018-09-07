# coding: utf-8

from __future__ import print_function
import codecs
from keras.layers import Input, Dense, Embedding, LSTM, TimeDistributed
from keras.layers import Bidirectional
from keras.models import Model
import numpy as np



def read_file(path):
    seqs = []
    with codecs.open(path, 'r') as f:
        for line in f:
            seqs.append([int(n) for n in line.strip().split(" ")])
    return seqs

num_words = 5178
max_length = 18
latent_dim = 100
inputs = Input(shape=(max_length, ))
embeddings = Embedding(input_dim=num_words+1, output_dim=200, mask_zero=True)(inputs)
bilstm = Bidirectional(LSTM(units=latent_dim, return_sequences=True))(embeddings)
inter_dense = TimeDistributed(Dense(units=100, activation='relu'))(bilstm)
outputs = TimeDistributed(Dense(units=20, activation='softmax'))(inter_dense)
model = Model(inputs=inputs, outputs=outputs)
print(model.summary())

# training model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
train_file_content = read_file('../data/train_words_seq.txt')
train_tags_content = read_file("../data/train_tags_seq.txt")
words_seq = np.array(train_file_content)
num_tag = 20
tag_seq = np.zeros(shape=(len(words_seq), max_length, num_tag))
for i in range(len(words_seq)):
    for j in range(max_length):
        tag_seq[i][j][train_tags_content[i][j]] = 1

vld_file_content = read_file('../data/validation_words_seq.txt')
vld_tags_content = read_file('../data/validation_tags_seq.txt')
vld_words_seq = np.array(vld_file_content)
vld_tag_seq = np.zeros(shape=(len(vld_words_seq), max_length, num_tag))
for i in range(len(vld_words_seq)):
    for j in range(max_length):
        vld_tag_seq[i][j][vld_tags_content[i][j]] = 1

batch_size = 36
epochs = 10
model.fit(x=words_seq, y=tag_seq,
          batch_size=batch_size,
          validation_data=[vld_words_seq, vld_tag_seq],
          epochs=epochs)
model.save('ner.h5')

