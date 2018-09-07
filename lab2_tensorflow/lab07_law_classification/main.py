# -*- coding: utf-8 -*-

from __future__ import print_function
from six.moves import xrange

import tensorflow as tf
import pickle
import os
import numpy as np
from datetime import datetime
import math

np.random.seed(666)


def load_pkl(filename):
    with open(filename, 'rb') as stream:
        return pickle.load(stream)


embedding_matrix = None  # load your trained word embedding vector
print('load word embedding, its shape is', embedding_matrix.shape)
sequences = None  # load your text matrix
print('load input sequences, its shape is', sequences.shape)
train_df = None  # load your labels
print('load input labels, its shape is', train_df.shape)
# select label type
model_type = 'max_law'  # or 'max_accu'
num_classes = {'max_law': 83,
               'max_accu': 202}[model_type]
labels = np.asarray(train_df[model_type], dtype=np.int32)
print('load input labels of %s, its shape is' % (model_type), labels.shape)
del train_df

input_num = 500
BATCH_SIZE = 30

graph = tf.Graph()
with graph.as_default():
    with tf.name_scope('placeholder') as scope:
        input_x = tf.placeholder(dtype=tf.int32, shape=[None, input_num], name='placeholder_x')
        y = tf.placeholder(dtype=tf.int32, shape=[None, ], name='placeholder_y')
        seq_len = tf.placeholder(dtype=tf.int32, shape=[None, ], name='placeholder_x_length')
        global_step = tf.Variable(0, trainable=False, name='global_step')
    input_y = tf.one_hot(y, num_classes)


    def EMBEDDING(embedding_matrix, trainable):
        # embedding, vector for each word in the vocabulary
        embeddings = tf.Variable(embedding_matrix, trainable=trainable,
                                 name='embedding', dtype=tf.float32)
        print('layer embedding, shape: %s' % ([None] + [int(dim) for dim in embeddings.shape[1:]]))
        return embeddings


    # Embedding layer
    with tf.name_scope('embedding'):
        # with tf.name_scope('embedding'), tf.device('/cpu:0'):
        # embedding
        embedding_layer = EMBEDDING(embedding_matrix, False)
        # input_x shape: (batch_size, sequence_length)
        embedding_inputs = tf.nn.embedding_lookup(embedding_layer, input_x)
    print('layer embed_vec, shape: %s' % ([None] + [int(dim) for dim in embedding_inputs.shape[1:]]))


    # Define Basic RNN Cell
    def basic_rnn_cell(rnn_size):
        return tf.contrib.rnn.GRUCell(rnn_size)


    rnn_size = 128
    keep_prob = 0.8
    num_layers = 1
    # Define Forward RNN Cell
    with tf.name_scope('fw_rnn'):
        fw_rnn_cell = tf.contrib.rnn.MultiRNNCell([basic_rnn_cell(rnn_size) for _ in range(num_layers)])
        fw_rnn_cell = tf.contrib.rnn.DropoutWrapper(fw_rnn_cell, output_keep_prob=keep_prob)

    # Define Backward RNN Cell
    with tf.name_scope('bw_rnn'):
        bw_rnn_cell = tf.contrib.rnn.MultiRNNCell([basic_rnn_cell(rnn_size) for _ in range(num_layers)])
        bw_rnn_cell = tf.contrib.rnn.DropoutWrapper(bw_rnn_cell, output_keep_prob=keep_prob)

    with tf.name_scope('bi_rnn'):
        # embedding_inputs shape: (batch_size, sequence_length, embedding_size)
        # rnn_output, _ = tf.nn.dynamic_rnn(fw_rnn_cell, inputs=embedding_inputs, sequence_length=seq_len, dtype=tf.float32)
        rnn_output, _ = tf.nn.bidirectional_dynamic_rnn(fw_rnn_cell, bw_rnn_cell, inputs=embedding_inputs,
                                                        sequence_length=seq_len, dtype=tf.float32)

    # In case of Bi-RNN, concatenate the forward and the backward RNN outputs
    if isinstance(rnn_output, tuple):
        rnn_output = tf.concat(rnn_output, 2)

    input_shape = rnn_output.shape  # (batch_size, sequence_length, hidden_size)
    sequence_size = input_shape[1].value  # the length of sequences processed in the RNN layer
    hidden_size = input_shape[2].value  # hidden size of the RNN layer

    # Attention Layer
    with tf.name_scope('attention'):
        attention_size = 100
        attention_w = tf.Variable(tf.truncated_normal([hidden_size, attention_size], stddev=0.1),
                                  name='attention_w')
        attention_b = tf.Variable(tf.constant(0.1, shape=[attention_size]), name='attention_b')
        attention_u = tf.Variable(tf.truncated_normal([attention_size], stddev=0.1), name='attention_u')
        z_list = []
        for t in range(sequence_size):
            u_t = tf.tanh(tf.matmul(rnn_output[:, t, :], attention_w) + tf.reshape(attention_b, [1, -1]))
            z_t = tf.matmul(u_t, tf.reshape(attention_u, [-1, 1]))
            z_list.append(z_t)
        alpha = tf.nn.softmax(tf.concat(z_list, axis=1))
        # Transform to batch_size * sequence_size * 1 , same rank as rnn_output
        attention_output = tf.reduce_sum(rnn_output * tf.reshape(alpha, [-1, sequence_size, 1]), 1)

    # Add dropout
    with tf.name_scope('dropout'):
        # attention_output shape: (batch_size, hidden_size)
        final_output = tf.nn.dropout(attention_output, keep_prob)

    # Fully connected layer
    with tf.name_scope('output'):
        fc_w = tf.Variable(tf.truncated_normal([hidden_size, num_classes], stddev=0.1), name='fc_w')
        fc_b = tf.Variable(tf.zeros([num_classes]), name='fc_b')
        logits = tf.matmul(final_output, fc_w) + fc_b
    tf.add_to_collection('logits', logits)
    predictions = tf.argmax(logits, -1, name='predictions')

    # Calculate cross-entropy loss
    with tf.name_scope('loss'):
        loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=input_y))
    tf.add_to_collection('loss_op', loss_op)

    # Create optimizer
    with tf.name_scope('optimizer'):
        optimizer = tf.train.AdamOptimizer(1e-3)
        gradients, variables = zip(*optimizer.compute_gradients(loss_op))
        gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
        optimizer_op = optimizer.apply_gradients(zip(gradients, variables),
                                                 global_step=global_step)
    tf.add_to_collection('optimizer_op', optimizer_op)

    # Calculate accuracy
    with tf.name_scope('accuracy'):
        correct_pred = tf.equal(tf.cast(predictions, tf.int32), y)
        accuracy_op = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    tf.add_to_collection('accuracy_op', accuracy_op)


def mkdir(filename):
    if os.path.exists(filename):
        pass
    else:
        os.mkdir(filename)


def load(sess, filename):
    # load
    print('load model %s' % (filename))
    filename = './%s/model.ckpt' % filename
    saver = tf.train.Saver()
    saver.restore(sess, filename)


def save(sess, filename):
    # save
    print('save model %s' % (filename))
    mkdir(filename)
    saver = tf.train.Saver()
    filename = './%s/model' % filename
    saver.save(sess, filename)


def load_graph(sess, filename, global_step):
    # load
    print('save model %s' % (filename))
    exit_flag = os.path.exists('./%s/model-%s.meta' % (filename, global_step))
    saver = tf.train.import_meta_graph('./%s/model-%s.meta' % (filename, global_step))
    saver.restore(sess, './%s/model-%s' % (filename, global_step))


def save_graph(sess, filename, global_step):
    # save
    print('save model %s' % (filename))
    mkdir(filename)
    saver = tf.train.Saver()
    filename = './%s/model' % filename
    saver.save(sess, filename, global_step)


BATCH_SIZE = 100

# Start running operations on the Graph.
config = tf.ConfigProto()
# 根据需要自动申请
config.gpu_options.allow_growth = True
with tf.Session(config=config, graph=graph) as sess:
    sess.run(tf.global_variables_initializer())


    def train(training_episodes, train_interval=50):
        print('-----train-----')
        batch_episodes = int(math.ceil(sequences.shape[0] / BATCH_SIZE))
        for i in xrange(1, 1 + training_episodes):
            print('Episode %s / %s' % (i, training_episodes))
            for start_index in xrange(batch_episodes):
                if (start_index + 1) * BATCH_SIZE > sequences.shape[0]:
                    batch_sequences = sequences[start_index * BATCH_SIZE:]
                    batch_labels = labels[start_index * BATCH_SIZE:]
                    print('end of this episode, length of data is', len(batch_sequences))
                else:
                    batch_sequences = sequences[start_index * BATCH_SIZE:(start_index + 1) * BATCH_SIZE]
                    batch_labels = labels[start_index * BATCH_SIZE:(start_index + 1) * BATCH_SIZE]
                batch_x_length = [len(seq) - sum(seq == 0) for seq in batch_sequences]
                feed_dict = {input_x: batch_sequences, y: batch_labels, seq_len: np.asarray(batch_x_length)}
                if start_index % train_interval == 0:
                    _, batch_loss, batch_acc = sess.run([optimizer_op, loss_op, accuracy_op],
                                                        feed_dict=feed_dict)
                    print('Time: %s, training %-5s, loss: %-.5f, accuracy: %-.5f' % (
                        datetime.now(), start_index, batch_loss, batch_acc))
                    save_graph(sess, filename=filename, global_step=episode)
                else:
                    _ = sess.run([optimizer_op], feed_dict=feed_dict)


    # Create summary writer
    writer = tf.summary.FileWriter('history', sess.graph)
    writer.add_graph(sess.graph)

    filename = './rnn_model_%s' % (model_type)
    episode = 10
    train(training_episodes=episode)
    # save_graph(sess, filename=filename, global_step=episode)
'''
tensorboard --logdir history
'''
