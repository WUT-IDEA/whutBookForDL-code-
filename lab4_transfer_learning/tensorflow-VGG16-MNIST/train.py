# -*- coding: utf-8 -*-

from __future__ import print_function
from six.moves import xrange

import tensorflow as tf
from PIL import Image

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('vgg_layer_num', '5',
                            """layer num of vgg output""")

tf.app.flags.DEFINE_integer('batch_size', '1000',
                            """batch size of training""")
tf.app.flags.DEFINE_float('learning_rate', '1e-4',
                          """batch size of training""")
tf.app.flags.DEFINE_integer('training_episode', '100',
                            """training episode""")
import numpy as np
import os
import matplotlib.pyplot as plt

# hyper-parameters
image_size = 28
image_chl = 3
classes_num = 10
hidden_num = 50

from tensorflow.examples.tutorials.mnist import input_data


########################################################################################
# Davi Frossard, 2016                                                                  #
# VGG16 implementation in TensorFlow                                                   #
# Details:                                                                             #
# http://www.cs.toronto.edu/~frossard/post/vgg16/                                      #
#                                                                                      #
# Model from https://gist.github.com/ksimonyan/211839e770f7b538e2d8#file-readme-md     #
# Weights from Caffe converted using https://github.com/ethereon/caffe-tensorflow      #
########################################################################################


class vgg16:
    def __init__(self, weights=None, sess=None):
        self.imgs = tf.placeholder(tf.float32, [None, image_size, image_size, image_chl])
        self.weight_file = weights
        self.sess = sess
        self.convlayers()
        if self.weight_file is not None and self.sess is not None:
            self.load_weights()

    def convlayers(self):
        self.parameters = []

        # zero-mean input
        with tf.name_scope('preprocess') as scope:
            mean = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32, shape=[1, 1, 1, 3], name='img_mean')
            self.images = self.imgs - mean

        # conv1_1
        with tf.name_scope('conv1_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 3, 64], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.images, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv1_1 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv1_2
        with tf.name_scope('conv1_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 64, 64], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.conv1_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv1_2 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # pool1
        self.pool1 = tf.nn.max_pool(self.conv1_2,
                                    ksize=[1, 2, 2, 1],
                                    strides=[1, 2, 2, 1],
                                    padding='SAME',
                                    name='pool1')

        # conv2_1
        with tf.name_scope('conv2_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 64, 128], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.pool1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv2_1 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv2_2
        with tf.name_scope('conv2_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 128, 128], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.conv2_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv2_2 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # pool2
        self.pool2 = tf.nn.max_pool(self.conv2_2,
                                    ksize=[1, 2, 2, 1],
                                    strides=[1, 2, 2, 1],
                                    padding='SAME',
                                    name='pool2')

        # conv3_1
        with tf.name_scope('conv3_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 128, 256], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.pool2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv3_1 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv3_2
        with tf.name_scope('conv3_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 256], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.conv3_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv3_2 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv3_3
        with tf.name_scope('conv3_3') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 256], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.conv3_2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv3_3 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # pool3
        self.pool3 = tf.nn.max_pool(self.conv3_3,
                                    ksize=[1, 2, 2, 1],
                                    strides=[1, 2, 2, 1],
                                    padding='SAME',
                                    name='pool3')

        # conv4_1
        with tf.name_scope('conv4_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 512], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.pool3, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv4_1 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv4_2
        with tf.name_scope('conv4_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.conv4_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv4_2 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv4_3
        with tf.name_scope('conv4_3') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.conv4_2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv4_3 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # pool4
        self.pool4 = tf.nn.max_pool(self.conv4_3,
                                    ksize=[1, 2, 2, 1],
                                    strides=[1, 2, 2, 1],
                                    padding='SAME',
                                    name='pool4')

        # conv5_1
        with tf.name_scope('conv5_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.pool4, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv5_1 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv5_2
        with tf.name_scope('conv5_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.conv5_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv5_2 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv5_3
        with tf.name_scope('conv5_3') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.conv5_2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv5_3 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # pool5
        self.pool5 = tf.nn.max_pool(self.conv5_3,
                                    ksize=[1, 2, 2, 1],
                                    strides=[1, 2, 2, 1],
                                    padding='SAME',
                                    name='pool5')

    def load_weights(self):
        weights = np.load(self.weight_file)
        keys = sorted(weights.keys())[:26]
        for i, k in enumerate(keys):
            print(i, k, np.shape(weights[k]))
            self.sess.run(self.parameters[i].assign(weights[k]))

    def from_1chl_to_3chl(self, array):
        array = np.reshape(array, [-1, image_size, image_size])
        new_array = np.zeros(shape=(list(array.shape) + [image_chl]), dtype=np.float32)
        new_array[..., 0] = array
        new_array[..., 1] = array
        new_array[..., 2] = array
        del array
        return new_array

    def pipeline(self):
        features = []

        # orignal image
        mnist = input_data.read_data_sets('./MNIST_data', one_hot=True)
        img, _ = mnist.train.next_batch(batch_size=1)
        del mnist
        img = self.from_1chl_to_3chl(img) * 255
        features += [img[0, :, :, :1]]

        self.sess.run(tf.global_variables_initializer())
        pool = self.sess.run([self.pool1, self.pool2, self.pool3, self.pool4, self.pool5], \
                             feed_dict={self.images: img})
        pool = [pl[0, ...] for pl in pool]
        features += pool

        return features

    def save_imgs(self):
        images_collection = self.pipeline()
        col_dir = 'img/'
        self.mkdir(col_dir)

        for index in xrange(len(images_collection)):
            image_dir = os.path.join(col_dir, str(index))
            self.mkdir(image_dir)
            images = images_collection[index]
            for ind in xrange(images.shape[-1]):
                Image.fromarray(images[..., ind], 'I').save(os.path.join(image_dir, '%s.png' % (ind)))

    def mkdir(self, file_dir):
        if os.path.exists(file_dir):
            pass
        else:
            os.mkdir(file_dir)

    def save_features(self, layer):
        # training
        mnist = input_data.read_data_sets('./MNIST_data', one_hot=True)
        # cus file is too big, use various files to save data
        file_dir = 'mnist_%s/' % (layer)
        self.mkdir(file_dir)
        raw_dataset = {'train': mnist.train, 'test': mnist.test}
        # 获取VGG网络中的某层网络输出作为特征
        image_pipeline = self.__getattribute__('pool%s' % layer)
        for train_type in raw_dataset.keys():
            for i in xrange(1, 1 + raw_dataset[train_type].num_examples // FLAGS.batch_size):
                if i % 10:
                    print('pool%s feature, %s %d' % (layer, train_type, (i + 1) * FLAGS.batch_size))
                batch_images, batch_label = raw_dataset[train_type].next_batch(FLAGS.batch_size)
                batch_images = self.from_1chl_to_3chl(batch_images) * 255
                batch_images = self.sess.run(image_pipeline, feed_dict={self.imgs: batch_images})
                np.savez(os.path.join(file_dir, '%s_%s.npz' % (train_type, i)), a=batch_images, b=batch_label)
        del mnist, raw_dataset


class LinearNet:
    def __init__(self, sess):
        # hyper parameters
        self.sess = sess

    def dataset(self, layer):
        self.mnist = {'train': 'mnist_%s/' % (layer), 'test': 'mnist_%s/' % (layer)}
        for train_type in self.mnist.keys():
            files = os.listdir(self.mnist[train_type])
            self.mnist[train_type] = [os.path.join(self.mnist[train_type], fl) for fl in files if train_type in fl]
            del files

        self.input_size = self.acc_mul(np.load(self.mnist[train_type][0])['a'].shape[1:])

    def acc_mul(self, array):
        result = 1
        for arr in array:
            result *= arr
        return result

    def flatten(self, images):
        images = np.reshape(images, newshape=(images.shape[0], -1))
        images = np.asarray([(img - img.min()) / (img.max() - img.min()) for img in images], dtype=np.float32)
        return images

    def build(self, layer):
        self.dataset(layer)

        self.x = tf.placeholder(dtype=tf.float32, shape=[None, self.input_size], name='x')
        self.y = tf.placeholder(dtype=tf.float32, shape=[None, classes_num], name='y')

        # hidden layer
        linear1 = tf.layers.dense(inputs=self.x, units=hidden_num, activation=tf.nn.relu, name='linear1')
        linear2 = tf.layers.dense(inputs=linear1, units=classes_num, activation=tf.nn.softmax, name='linear2')
        self.y_ = linear2

        with tf.name_scope('loss_op'):
            self.loss_op = tf.reduce_mean(-tf.reduce_sum(self.y * tf.log(self.y_), axis=-1))

        with tf.name_scope('train_op'):
            self.train_op = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate).minimize(self.loss_op)

        with tf.name_scope('accuracy_op'):
            self.acc_op = tf.reduce_mean(
                tf.cast(tf.equal(tf.argmax(self.y, 1), tf.argmax(self.y_, 1)), dtype=tf.float32))

    def train(self, layer):
        self.build(layer)
        self.sess.run(tf.global_variables_initializer())
        accuracy = []

        print('training...')
        # training
        for i in range(FLAGS.training_episode):
            for batch_file in self.mnist['train']:
                npz_file = np.load(batch_file)
                batch_xs, batch_ys = npz_file['a'], npz_file['b']
                batch_xs = self.flatten(batch_xs)
                _ = self.sess.run([self.train_op], feed_dict={self.x: batch_xs, self.y: batch_ys})
                del npz_file

            # evaluate
            acc = 0.0
            for batch_file in self.mnist['test']:
                npz_file = np.load(batch_file)
                batch_xs, batch_ys = npz_file['a'], npz_file['b']
                batch_xs = self.flatten(batch_xs)
                batch_acc = self.sess.run(self.acc_op, feed_dict={self.x: batch_xs, self.y: batch_ys})
                acc += batch_acc
                del npz_file
            acc = acc * 100.0 / len(self.mnist['test'])
            accuracy.append(acc)
            print('%.3f' % (acc))
        np.save('layer%s' % (layer), accuracy)


if __name__ == '__main__':
    # config = tf.ConfigProto()
    # 制定显存大小
    # config.gpu_options.per_process_gpu_memory_fraction = 0.5
    # 根据需要自动申请
    # config.gpu_options.allow_growth = True
    # sess = tf.Session(config=config)
    # vgg = vgg16('../vgg16_weights.npz', sess)
    # vgg.save_imgs()
    # for i in xrange(1, 1 + FLAGS.vgg_layer_num):
    #     vgg.save_features(layer=i)
    # del vgg

    # net = LinearNet(sess)
    # net.train(layer=5)

    accuracy = [np.load('layer%s.npy' % (i)) for i in xrange(1, 1 + FLAGS.vgg_layer_num)]
    for ind, acc in enumerate(accuracy):
        plt.plot(xrange(1, 101), acc, label='layer%s(VGG16)' % (ind + 1))

    accuracy = np.load('scratch_accuracy.npy')
    plt.plot(xrange(1, 101), accuracy, label='fc')

    plt.legend(loc='best')
    plt.show()
