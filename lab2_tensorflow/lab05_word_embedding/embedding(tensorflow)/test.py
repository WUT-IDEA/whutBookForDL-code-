# -*- coding: utf-8 -*-

from __future__ import print_function
from six.moves import xrange

import pickle
import numpy as np


def read(filename):
    with open(filename, 'rb') as file_stream:
        return pickle.load(file_stream)


def save(filename, obj):
    with open(filename, 'wb') as file_stream:
        pickle.dump(obj, file_stream)


class Embedding:
    def __init__(self, dictname, embedname):
        self.dictionary = read(dictname)
        self.reverse_dictionary = dict(zip(self.dictionary.values(), self.dictionary.keys()))
        self.embedding = np.load(embedname)
        # self.embedding = self.embedding / np.sqrt(np.sum(np.square(self.embedding), axis=1, keepdims=True))

    def __similarity_2words(self, index1, index2):
        return np.dot(self.embedding[index1], self.embedding[index2])

    def __similarity_embedding(self, index1):
        return np.dot(self.embedding[index1], np.transpose(self.embedding))

    def similarity(self, keyword1, keyword2):
        index1 = self.dictionary[keyword1]
        index2 = self.dictionary[keyword2]
        return self.__similarity_2words(index1, index2)

    def topk(self, keyword, top_k=10):
        index = self.dictionary[keyword]
        embedding_distance = -self.__similarity_embedding(index)
        assert embedding_distance.shape[0] == self.embedding.shape[0]
        nearest_indices = embedding_distance.argsort()[:top_k + 1]
        del embedding_distance
        assert nearest_indices[0] == index
        return [self.reverse_dictionary[ind] for ind in nearest_indices[1:]]


# # Skip-gram model
# embedding = Embedding(dictname='dictionary(Skip-gram).pkl', \
#                       embedname='embedding(Skip-gram).npy')
#
# # Nearest to or: and, than, nor, without, though, ore, forbid, scholastic,
# print(embedding.topk(keyword='or', top_k=8))

for model in ('Skip-gram', 'CBOW'):
    print('%s model' % (model))
    embedding = Embedding(dictname='dictionary(%s).pkl' % (model), \
                          embedname='embedding(%s).npy' % (model))
    print('Nearest to time: %s' % embedding.topk(keyword='time', top_k=8))

# Skip-gram model
# Nearest to time: ['week', 'exceeds', 'aerodrome', 'period', 'carbine', 'year', 'degree', 'decade']
# CBOW model
# Nearest to time: ['week', 'latency', 'ropes', 'configuration', 'ruse', 'comforts', 'periods', 'period']