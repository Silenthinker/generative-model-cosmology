# Some code was borrowed from https://github.com/petewarden/tensorflow_makefile/blob/master/tensorflow/models/image/mnist/convolutional.py

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy
import os
import csv
import tensorflow as tf
import pandas as pd
import tensorflow.contrib.slim as slim
from tensorflow.examples.tutorials.mnist import input_data

import cnn_model
import load_data

# user input
from argparse import ArgumentParser

# refernce argument values
MODEL_DIRECTORY = "model"
TEST_BATCH_SIZE = 5000
ENSEMBLE = True

# build parser

parser = ArgumentParser()

parser.add_argument('--model_dir', type=str, default="model", required=True, help='directory where model to be tested is stored')
parser.add_argument('--data_dir', type=str, default="data", required=True, help='test data directory [data]')
parser.add_argument('--batch_size', type=int, default=64, help='batch size [64]')
parser.add_argument('--use-ensemble',
                    dest='ensemble', help='boolean for usage of ensemble',
                    metavar='ENSEMBLE', required=True)
parser.add_argument('--input_size', type=int, required=True, help='input size')
parser.add_argument('--num_classes', type=int, required=True, help='number of classes')
parser.add_argument('--test', type=bool, required=True, help='True for real test without ground truth')
args = parser.parse_args()

def test(args, model_directory):
    # Import data
    input_size = args.input_size
    num_classes = args.num_classes
    batch_size = args.batch_size
    test_data, img_prefixes = load_data.prepare_cosmology_test_data(args)

    is_training = tf.placeholder(tf.bool, name='MODE')

    # tf Graph input
    x = tf.placeholder(tf.float32, [None, input_size*input_size])
    y = cnn_model.CNN(args, x, is_training=is_training)

    # Add ops to save and restore all the variables
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer(), feed_dict={is_training: True})

    ckpt = tf.train.get_checkpoint_state(args.model_dir)
    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
        saver = tf.train.Saver()
        saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        print("Created model with fresh parameters.")
        session.run(tf.global_variables_initializer())

    # Calculate accuracy for all mnist test images
    test_size = test_data.shape[0]
    total_batch = test_size/batch_size if test_size%batch_size==0 else int(test_size / batch_size) + 1

    res = []
    # Loop over all batches
    for i in range(total_batch):
        offset = (i * batch_size) % (test_size)
        batch_xs = test_data[offset:(offset + batch_size), :]
        y_final = sess.run(y, feed_dict={x: batch_xs, is_training: False})
        res.extend(y_final.flatten())
    pred = dict(zip(img_prefixes, res))
    print(pred)
    df = pd.DataFrame.from_dict(pred, orient="index")
    df = df.reset_index()
    df.to_csv(os.path.join(args.data_dir, "prediction.csv"), header=["Id", "Predicted"], index=False)

'''
# test with test data given by mnist_data.py
def test_org(model_directory, batch_size):
    # Import data
    PIXEL_DEPTH = mnist_data.PIXEL_DEPTH
    train_total_data, train_size, validation_data, validation_labels, test_data, test_labels = mnist_data.prepare_MNIST_data(
        False)

    is_training = tf.placeholder(tf.bool, name='MODE')

    # tf Graph input
    x = tf.placeholder(tf.float32, [None, 784])
    y_ = tf.placeholder(tf.float32, [None, 10])  # answer
    y = cnn_model.CNN(x, is_training=is_training)

    # Add ops to save and restore all the variables
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer(), feed_dict={is_training: True})

    # Restore variables from disk
    saver = tf.train.Saver()

    # Calculate accuracy for all mnist test images
    test_size = test_labels.shape[0]
    total_batch = int(test_size / batch_size)

    saver.restore(sess, model_directory)

    acc_buffer = []

    # Loop over all batches
    for i in range(total_batch):
        # Compute the offset of the current minibatch in the data.
        offset = (i * batch_size) % (test_size)
        batch_xs = test_data[offset:(offset + batch_size), :]
        batch_ys = test_labels[offset:(offset + batch_size), :]

        y_final = sess.run(y, feed_dict={x: batch_xs, y_: batch_ys, is_training: False})

        correct_prediction = numpy.equal(numpy.argmax(y_final, 1), numpy.argmax(batch_ys, 1))

        acc_buffer.append(numpy.sum(correct_prediction) / batch_size)

    print("test accuracy for the stored model: %g" % numpy.mean(acc_buffer))
'''

# For a given matrix, each row is converted into a one-hot row vector
def one_hot_matrix(a):
    a_ = numpy.zeros_like(a)
    for i, j in zip(numpy.arange(a.shape[0]), numpy.argmax(a, 1)): a_[i, j] = 1
    return a_

# test with test data given by mnist_data.py
def test_ensemble(args, model_directory_list, batch_size):
    # Import data
    input_size = args.input_size
    num_classes = args.num_classes
    PIXEL_DEPTH = mnist_data.PIXEL_DEPTH
    mnist = input_data.read_data_sets('data/', one_hot=True)

    is_training = tf.placeholder(tf.bool, name='MODE')

    # tf Graph input
    x = tf.placeholder(tf.float32, [None, input_size*input_size])
    y_ = tf.placeholder(tf.float32, [None, num_classes])  # answer
    y = cnn_model.CNN(args, x, is_training=is_training)

    # Add ops to save and restore all the variables
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer(), feed_dict={is_training: True})

    # Restore variables from disk
    saver = tf.train.Saver()

    # Calculate accuracy for all mnist test images
    test_size = mnist.test.num_examples
    total_batch = int(test_size / batch_size)

    acc_buffer = []
    # Loop over all batches
    for i in range(total_batch):

        batch = mnist.test.next_batch(batch_size)
        batch_xs = (batch[0] - (PIXEL_DEPTH / 2.0) / PIXEL_DEPTH)  # make zero-centered distribution as in mnist_data.extract_data()
        batch_ys = batch[1]

        y_final = numpy.zeros_like(batch_ys)

        for dir in model_directory_list:
            saver.restore(sess, dir+'/model.ckpt')
            pred = sess.run(y, feed_dict={x: batch_xs, y_: batch_ys, is_training: False})
            y_final += one_hot_matrix(pred) # take a majority vote as an answer

        correct_prediction = numpy.equal(numpy.argmax(y_final, 1), numpy.argmax(batch_ys, 1))

        acc_buffer.append(numpy.sum(correct_prediction) / batch_size)

    print("test accuracy for the stored model: %g" % numpy.mean(acc_buffer))

if __name__ == '__main__':
    ensemble = args.ensemble
    model_dir = args.model_dir
    batch_size = args.batch_size
    # Select ensemble test or a single model test
    if ensemble=='True': # use ensemble model
        model_directory_list = [x[0] for x in os.walk(model_dir)]
        test_ensemble(model_directory_list[1:], batch_size)
    else: # test a single model
        # test_org(model_dir, batch_size) #test with test data given by mnist_data.py
        test(args, model_dir+'/model.ckpt')  # test with test data given by tensorflow.examples.tutorials.mnist.input_data()