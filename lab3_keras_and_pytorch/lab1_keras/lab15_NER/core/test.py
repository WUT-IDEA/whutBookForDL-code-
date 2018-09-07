# coding: utf-8

from keras.models import load_model
import numpy as np
import codecs

def load_dict(path):
    n_dict = dict()
    with codecs.open(path, 'r', encoding = 'utf-8') as reader:
        for line in reader:
            parts = line.strip().split(" ")
            n_dict[int(parts[1])] = parts[0]
    return n_dict


model = load_model('ner.h5')
word_dict = load_dict("../data/all_words.txt")
tag_dict = load_dict("../data/all_tags.txt")
seq_str = "1052 4811 604 1745 1696 407 4461 2177 1695 179 2676 3887 3034 2100 398 1 0 0"
seq = np.array([int(n_str) for n_str in seq_str.split(" ")]).reshape(1,18)
result = np.argmax(model.predict(seq), axis=2)
for i in range(18):
    print(word_dict[seq[0][i]] + "  ->  " + tag_dict[result[0][i]])


