import codecs
from collections import defaultdict
import jieba
import re
from gensim.models.doc2vec import Doc2Vec
import os
import numpy as np
from keras.preprocessing import image
from keras.applications.inception_v3 import preprocess_input


def load_dict():
    word_freq = defaultdict(int)
    with codecs.open('dict.txt', "r", encoding='utf-8') as dict:
        dict_str = dict.read()
        for item in dict_str.split(' '):
            word, freq = item.split(':')
            word_freq[word] =freq
    print('len(dict): ' + str(len(word_freq)))
    return word_freq


def load_stopwords():
    with codecs.open('stopwords.txt', "r", encoding='utf-8') as stop_f:
        stopwords = list()
        for line in stop_f.readlines():
            line = line.strip()
            if not len(line):
                continue
            stopwords.append(line)
    print('len(stopwords):' + str(len(stopwords)))
    return stopwords


pic_path = 'data/News_pic_info_train'
texts = []
pics = []
stopwords = load_stopwords()
word_freq = load_dict()
count = 0

# load the pre-trained doc2vec model
doc2vec_model = Doc2Vec.load('models/doc2vec.model')
for pic_name in os.listdir(pic_path):
    file_path = os.path.join(pic_path, pic_name)
    img = image.load_img(file_path, target_size=(299, 299))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    pics.append(img)

    name = pic_name.split('.')[0]
    txt_name = 'data/News_info_train/' + name + '.txt'
    with codecs.open(txt_name, "r", encoding='utf-8') as f:
        txt = f.read()
        text = txt[txt.index('\t', txt.index('\t') + 1) + 1:].strip()
        text = re.sub(ur'[\u0000-\u4dff\u9FA6-\uffff]', '', text)
        word_list = jieba.cut(text)
        result = []
        for word in word_list:
            if word_freq.has_key(word) and word not in stopwords and len(word.strip()) > 0:
                result.append(word)
        text_vector = doc2vec_model.infer_vector(result)
        texts.append(text_vector)

        count += 1
        if count % 25 == 0:
            index = count / 25
            np.save('data/texts/' + str(index) + '.npy', texts)
            np.save('data/pics/' + str(index) + '.npy', pics)
            print 'save ' + str(index) + ' npy'
            texts = []
            pics = []
