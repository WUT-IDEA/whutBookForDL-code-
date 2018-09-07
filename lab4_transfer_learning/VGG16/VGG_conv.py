# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from six.moves import xrange

'''
adapted on http://www.cs.toronto.edu/~frossard/post/vgg16/
download VVG16.npz in the same path(https://www.cs.toronto.edu/~frossard/vgg16/vgg16_weights.npz)
'''

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np


class vgg16_conv:
    def __init__(self,
                 learning_rate=1e-3,
                 batch_size=100,
                 num_classes=10,
                 optimizer=tf.train.AdamOptimizer,
                 weights=None):
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.optimizer = optimizer
        self.weights = weights
        self.gstep = tf.Variable(0, dtype=tf.float32,
                                 trainable=False, name='global_step')

        # with tf.device('/cpu:0'):
        with tf.device('/gpu:0'):
            self.build()

    def convlayers(self, trainable=False):
        self.parameters = []

        # zero-mean input
        with tf.name_scope('preprocess') as scope:
            mean = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32, shape=[1, 1, 1, 3], name='img_mean')
            images = self.imgs - mean

        # conv1_1
        with tf.name_scope('conv1_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 3, 64], dtype=tf.float32,
                                                     stddev=1e-1), name='weights', trainable=trainable)
            conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32),
                                 trainable=trainable, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv1_1 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv1_2
        with tf.name_scope('conv1_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 64, 64], dtype=tf.float32,
                                                     stddev=1e-1), name='weights', trainable=trainable)
            conv = tf.nn.conv2d(self.conv1_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32),
                                 trainable=trainable, name='biases')
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
                                                     stddev=1e-1), name='weights', trainable=trainable)
            conv = tf.nn.conv2d(self.pool1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32),
                                 trainable=trainable, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv2_1 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv2_2
        with tf.name_scope('conv2_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 128, 128], dtype=tf.float32,
                                                     stddev=1e-1), name='weights', trainable=trainable)
            conv = tf.nn.conv2d(self.conv2_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32),
                                 trainable=trainable, name='biases')
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
                                                     stddev=1e-1), name='weights', trainable=trainable)
            conv = tf.nn.conv2d(self.pool2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                                 trainable=trainable, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv3_1 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv3_2
        with tf.name_scope('conv3_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 256], dtype=tf.float32,
                                                     stddev=1e-1), name='weights', trainable=trainable)
            conv = tf.nn.conv2d(self.conv3_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                                 trainable=trainable, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv3_2 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv3_3
        with tf.name_scope('conv3_3') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 256], dtype=tf.float32,
                                                     stddev=1e-1), name='weights', trainable=trainable)
            conv = tf.nn.conv2d(self.conv3_2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                                 trainable=trainable, name='biases')
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
                                                     stddev=1e-1), name='weights', trainable=trainable)
            conv = tf.nn.conv2d(self.pool3, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=trainable, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv4_1 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv4_2
        with tf.name_scope('conv4_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                     stddev=1e-1), name='weights', trainable=trainable)
            conv = tf.nn.conv2d(self.conv4_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=trainable, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv4_2 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv4_3
        with tf.name_scope('conv4_3') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                     stddev=1e-1), name='weights', trainable=trainable)
            conv = tf.nn.conv2d(self.conv4_2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=trainable, name='biases')
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
                                                     stddev=1e-1), name='weights', trainable=trainable)
            conv = tf.nn.conv2d(self.pool4, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=trainable, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv5_1 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv5_2
        with tf.name_scope('conv5_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                     stddev=1e-1), name='weights', trainable=trainable)
            conv = tf.nn.conv2d(self.conv5_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=trainable, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv5_2 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv5_3
        with tf.name_scope('conv5_3') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                     stddev=1e-1), name='weights', trainable=trainable)
            conv = tf.nn.conv2d(self.conv5_2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=trainable, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv5_3 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # pool5
        self.pool5 = tf.nn.max_pool(self.conv5_3,
                                    ksize=[1, 2, 2, 1],
                                    strides=[1, 2, 2, 1],
                                    padding='SAME',
                                    name='pool4')

    def fc_layers(self):
        # fc1
        with tf.name_scope('fc1') as scope:
            shape = int(np.prod(self.pool5.get_shape()[1:]))
            fc1w = tf.Variable(tf.truncated_normal([shape, 128],
                                                   dtype=tf.float32,
                                                   stddev=1e-1), name='weights')
            fc1b = tf.Variable(tf.constant(1.0, shape=[128], dtype=tf.float32),
                               trainable=True, name='biases')
            pool5_flat = tf.reshape(self.pool5, [-1, shape])
            fc1l = tf.nn.bias_add(tf.matmul(pool5_flat, fc1w), fc1b)
            self.fc1 = tf.nn.relu(fc1l)
            self.parameters += [fc1w, fc1b]

        # fc2
        with tf.name_scope('fc2') as scope:
            fc2w = tf.Variable(tf.truncated_normal([128, 10],
                                                   dtype=tf.float32,
                                                   stddev=1e-1), name='weights')
            fc2b = tf.Variable(tf.constant(1.0, shape=[10], dtype=tf.float32),
                               trainable=True, name='biases')
            fc2l = tf.nn.bias_add(tf.matmul(self.fc1, fc2w), fc2b)
            self.fc2 = tf.nn.relu(fc2l)
            self.parameters += [fc2w, fc2b]

    def load_weights(self, weight_file, sess):
        weights = np.load(weight_file)
        keys = sorted(weights.keys())
        print('\nModel structure:')
        for i, k in enumerate(keys[:26]):
            print(i, k, np.shape(weights[k]))
            sess.run(self.parameters[i].assign(weights[k]))

    def get_data(self):
        with tf.name_scope('data'):
            mnist = input_data.read_data_sets('./MNIST_data', one_hot=True)

            def from1D_to_3D_3ch(array):
                array = np.reshape(array, [-1, 28, 28])
                new_array = np.zeros(shape=(list(array.shape) + [3]), dtype=np.float32)
                new_array[..., 0] = array
                new_array[..., 1] = array
                new_array[..., 2] = array
                del array
                return new_array

            mnist.train._images = from1D_to_3D_3ch(mnist.train.images)
            mnist.test._images = from1D_to_3D_3ch(mnist.test.images)

            # Create datasets and iterator
            train_data = tf.data.Dataset.from_tensor_slices((
                mnist.train._images, mnist.train._labels
            ))
            train_data = train_data.shuffle(10000)  # if you want to shuffle your data
            train_data = train_data.batch(self.batch_size)

            test_data = tf.data.Dataset.from_tensor_slices((
                mnist.test._images, mnist.test._labels
            ))
            test_data = test_data.batch(self.batch_size)

            iterator = tf.data.Iterator.from_structure(train_data.output_types,
                                                       train_data.output_shapes)
            self.imgs, self.label = iterator.get_next()

            self.train_init = iterator.make_initializer(train_data)  # initializer for train_data
            self.test_init = iterator.make_initializer(test_data)  # initializer for train_data

            del mnist

    def loss(self):
        with tf.name_scope('loss_op'):
            entropy = tf.nn.softmax_cross_entropy_with_logits(labels=self.label, logits=self.fc2)
            self.loss_op = tf.reduce_mean(entropy, name='loss')

    def optimize(self):
        with tf.name_scope('optimizer_op'):
            self.optimizer_op = self.optimizer(self.learning_rate).minimize(self.loss_op,
                                                                            global_step=self.gstep)  # global_step isvery important

    def accuracy(self):
        with tf.name_scope('acc_op'):
            self.acc_op = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.label, 1), tf.argmax(self.fc2, 1)), \
                                                 tf.float32))

    def build(self):
        self.get_data()

        self.convlayers()
        self.fc_layers()

        self.loss()
        self.optimize()
        self.accuracy()

    def train_one_epoch(self, sess):
        sess.run(self.train_init)
        cost = 0.0
        step = self.gstep.eval()

        try:
            while True:
                _, batch_cost, acc_op = sess.run([self.optimizer_op, self.loss_op, self.acc_op])
                cost += batch_cost
                step += 1
                print('Epoch %s, loss %s, acc %s' % (step, batch_cost, acc_op))

        except tf.errors.OutOfRangeError:
            pass

        return cost * 1.0 / step

    def eval_one_epoch(self, sess):
        sess.run(self.test_init)
        accuracy = 0.0
        step = 0

        try:
            while True:
                _, batch_acc = sess.run([self.optimizer_op, self.acc_op])
                accuracy += batch_acc
                step += 1
        except tf.errors.OutOfRangeError:
            pass

        return accuracy * 100.0 / step

    def train(self, n_epoches=100):
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            # load model
            if self.weights is not None:
                self.load_weights(self.weights, sess)

            print('\nTrainable parameters:')
            for variable in tf.trainable_variables():
                print(variable)

            for epoch in xrange(1, 1 + n_epoches):
                cost = self.train_one_epoch(sess)
                accuracy = self.eval_one_epoch(sess)
                print('Epoch %s/%s, loss value is %s, accuracy is %s.' % (epoch, n_epoches, cost, accuracy))


vgg = vgg16_conv(weights='vgg16_weights.npz')
vgg.train(n_epoches=10)
