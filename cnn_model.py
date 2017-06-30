# Some code was borrowed from https://github.com/petewarden/tensorflow_makefile/blob/master/tensorflow/models/image/mnist/convolutional.py
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim

# Create model of CNN with slim api
def CNN(args, inputs, is_training=True):
    batch_norm_params = {'is_training': is_training, 'decay': 0.9, 'updates_collections': None}
    input_size = args.input_size
    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        normalizer_fn=slim.batch_norm,
                        normalizer_params=batch_norm_params):
        x = tf.reshape(inputs, [-1, input_size, input_size, 1])

        # For slim.conv2d, default argument values are like
        # normalizer_fn = None, normalizer_params = None, <== slim.arg_scope changes these arguments
        # padding='SAME', activation_fn=nn.relu,
        # weights_initializer = initializers.xavier_initializer(),
        # biases_initializer = init_ops.zeros_initializer,
        net = slim.conv2d(x, 16, [10, 10], stride=4, scope='conv1') # 246 -> 60
        net = slim.max_pool2d(net, [5, 5], scope='pool1')
        net = slim.conv2d(net, 32, [3, 3], scope='conv2')
        net = slim.max_pool2d(net, [3, 3], scope='pool2')
        net = slim.conv2d(net, 64, [3, 3], scope='conv3')
        net = slim.max_pool2d(net, [3, 3], scope='pool3')
        net = slim.flatten(net, scope='flatten4')

        # For slim.fully_connected, default argument values are like
        # activation_fn = nn.relu,
        # normalizer_fn = None, normalizer_params = None, <== slim.arg_scope changes these arguments
        # weights_initializer = initializers.xavier_initializer(),
        # biases_initializer = init_ops.zeros_initializer,
        net = slim.fully_connected(net, 1024, scope='fc3')
        net = slim.dropout(net, is_training=is_training, scope='dropout3')  # 0.5 by default
        if args.num_classes > 1:
            outputs = slim.fully_connected(net, args.num_classes, activation_fn=None, normalizer_fn=None, scope='fco')
        else: # regression
            outputs = slim.fully_connected(net, 1, activation_fn=tf.nn.relu, normalizer_fn=None, scope="fco")
    return outputs
'''
def vgg16(inputs):
    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                          activation_fn=tf.nn.relu,
                          weights_initializer=tf.truncated_normal_initializer(0.0, 0.01),
                          weights_regularizer=slim.l2_regularizer(0.0005)):
        net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')
        net = slim.max_pool2d(net, [2, 2], scope='pool1')
        net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
        net = slim.max_pool2d(net, [2, 2], scope='pool2')
        net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
        net = slim.max_pool2d(net, [2, 2], scope='pool3')
        net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
        net = slim.max_pool2d(net, [2, 2], scope='pool4')
        net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
        net = slim.max_pool2d(net, [2, 2], scope='pool5')
        net = slim.fully_connected(net, 4096, scope='fc6')
        net = slim.dropout(net, 0.5, scope='dropout6')
        net = slim.fully_connected(net, 4096, scope='fc7')
        net = slim.dropout(net, 0.5, scope='dropout7')
        net = slim.fully_connected(net, 1, activation_fn=tf.nn.relu, scope='fc8')
    return net
'''