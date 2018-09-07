# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from six.moves import xrange

# dataset
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("./MNIST_data/")

import numpy as np

# placeholder
graph = tf.Graph()
with graph.as_default():
    with tf.name_scope('placehold'):
        images = tf.placeholder(dtype=tf.float32, shape=[None, 784], name='x')
        labels = tf.placeholder(dtype=tf.int64, shape=[None, ], name='y')

    # images_reshape = tf.reshape(images, shape=[-1, 28, 28, 1])
    #
    # with tf.name_scope('conv'):
    #     conv1 = tf.layers.conv2d(images_reshape, filters=16, kernel_size=(3, 3))
    #     conv1_mp = tf.layers.max_pooling2d(conv1, pool_size=1, strides=1)
    #
    # features = tf.layers.flatten(conv1_mp)

    with tf.name_scope('fc'):
        fc1 = tf.layers.dense(inputs=images, units=128, activation=tf.nn.relu)
        fc2 = tf.layers.dense(inputs=fc1, units=10)

    with tf.name_scope('loss'):
        loss_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=fc2))
    with tf.name_scope('optimizer'):
        train_op = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(loss_op)
    with tf.name_scope('accuracy'):
        acc_op = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(fc2, axis=-1), labels), dtype=tf.float32))

# initialize
with tf.Session(graph=graph) as sess:
    sess.run(tf.global_variables_initializer())
    accuracy = []

    batch_size = 1000
    for ep in xrange(100):
        for i in xrange(mnist.train.num_examples // batch_size):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            _, = sess.run([train_op], feed_dict={images: batch_x, labels: batch_y})

        acc = acc_op.eval({images: mnist.test.images, labels: mnist.test.labels}) * 100
        accuracy.append(acc)
        print('%.3f' % (acc))
    np.save('scratch', accuracy)
