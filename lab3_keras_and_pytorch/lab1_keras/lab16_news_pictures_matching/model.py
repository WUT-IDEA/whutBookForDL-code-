# encoding: utf-8
from keras.applications.inception_v3 import InceptionV3
from keras.models import Model, load_model
from keras.layers import GlobalAveragePooling2D, Dense
import numpy as np
import os
from gensim.models.doc2vec import Doc2Vec
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing import image

# Inception v3模型，加载预训练权重，不保留顶层的三个全连接层
base_model = InceptionV3(weights='imagenet', include_top=False)

# 增加一个空域全局平均池化层
x = base_model.output
x = GlobalAveragePooling2D()(x)

# 增加个全连接层
features = Dense(1024, activation='relu')(x)

# 合并层，构建一个待fine-tune的新模型
model = Model(inputs=base_model.input, outputs=features)

# 冻结网络的前两个Inception blocks（即前249层），训练剩余层
for layer in model.layers[:249]:
    layer.trainable = False
for layer in model.layers[249:]:
    layer.trainable = True

# 打印模型概况
model.summary()

# 冻结层后，编译模型
model.compile(optimizer='rmsprop', loss='cosine_proximity')


# 提取图像的原始特征
def gen_img_raw_feature(pic_path):
    pics = []
    for pic_name in os.listdir(pic_path):
        file_path = os.path.join(pic_path, pic_name)
        img = image.load_img(file_path, target_size=(299, 299))
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = preprocess_input(img)
        pics.append(img)
    return pics


# 加载训练集，包括配图的原始像素特征向量和由Doc2vec生成的对应新闻文本特征向量
train_pics = gen_img_raw_feature('data/train_pics')
train_news_features = np.load('data/train_news_features.npy')
# 加载验证集
vld_pics = gen_img_raw_feature('data/vld_pics')
vld_news_features = np.load('data/vld_news_features.npy')

batch_size = 36
epochs = 10
model.fit(x=train_pics, y=train_news_features,
          batch_size=batch_size,
          validation_data=[vld_pics, vld_news_features],
          epochs=epochs)
# 保存模型
model.save('models/model.h5')


# 测试模型效果
# 自定义一个距离函数，距离越小，相似度越高
def cos_dis(A, B):
    num = np.dot(A, B)
    denom = np.linalg.norm(A) * np.linalg.norm(B)
    return - num / denom

# 加载训练好的模型
model = load_model('models/model.h5')
# 加载待检索的候选配图
test_pics = gen_img_raw_feature('/data/test_pics')
# 依据训练好的网络模型，计算每个候选配图的图像特征向量
predictions = model.predict(test_pics)
test_news_features = np.load('/data/test_news_features.npy')

similarity = dict()
# 将每个候选配图特征向量与测试新闻文本的文本向量计算距离
for pre_i in range(0, len(predictions)):
    similarity[pre_i] = cos_dis(test_news_features, predictions[pre_i])
# 根据计算得到的距离，从小到大排序
similarity = sorted(similarity.items(), key=lambda d: d[1])
# 输出与测试新闻文本最匹配的配图序号以及对应的距离
print similarity[0:10]