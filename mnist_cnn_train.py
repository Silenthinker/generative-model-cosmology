# Some code was borrowed from https://github.com/petewarden/tensorflow_makefile/blob/master/tensorflow/models/image/mnist/convolutional.py

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy
import argparse
import tensorflow as tf
import tensorflow.contrib.slim as slim

import mnist_data
import cnn_model


parser = argparse.ArgumentParser(description='Image classification')
parser.add_argument('--data_dir', type=str, default="data", help='data directory [data]')
parser.add_argument('--input_size', type=int, default=128, help='input size [128]')
parser.add_argument('--num_classes', type=int, default=2, help='number of classes [2]')
parser.add_argument('--training_epochs', type=int, default=10, help='epoch number [10]')
parser.add_argument('--batch_size', type=int, default=64, help='batch size [64]')
parser.add_argument('--display_step', type=int, default=100, help='display step [100]')
parser.add_argument('--validation_step', type=int, default=500, help='validation step [500]')
parser.add_argument('--augment', type=bool, default=True, help='augment data [True]')
parser.add_argument('--save_dir', type=str, default="model", help='save directory [model]')

args = parser.parse_args()

if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)
MODEL_DIRECTORY = os.path.join(args.save_dir, "model.ckpt")
LOGS_DIRECTORY = "logs/train"

# Params for Train
training_epochs = args.training_epochs # 10 for augmented training data, 20 for training data
display_step = args.display_step
validation_step = args.validation_step

# Params for test
TEST_BATCH_SIZE = args.batch_size

def train(args):
    # Some parameters
    batch_size = args.batch_size
    num_labels = mnist_data.NUM_LABELS
    input_size = args.input_size
    num_classes = args.num_classes

    # Prepare mnist data
    train_total_data, train_size, validation_data, validation_labels, test_data, test_labels = mnist_data.prepare_MNIST_data(args.augment)

    # Boolean for MODE of train or test
    is_training = tf.placeholder(tf.bool, name='MODE')

    # tf Graph input
    x = tf.placeholder(tf.float32, [None, input_size*input_size])
    y_ = tf.placeholder(tf.float32, [None, num_classes]) #answer

    # Predict
    y = cnn_model.CNN(args, x)

    # Get loss of model
    with tf.name_scope("LOSS"):
        loss = slim.losses.softmax_cross_entropy(y,y_)

    # Create a summary to monitor loss tensor
    tf.summary.scalar('loss', loss)

    # Define optimizer
    with tf.name_scope("ADAM"):
        # Optimizer: set up a variable that's incremented once per batch and
        # controls the learning rate decay.
        batch = tf.Variable(0)

        learning_rate = tf.train.exponential_decay(
            1e-4,  # Base learning rate.
            batch * batch_size,  # Current index into the dataset.
            train_size,  # Decay step.
            0.95,  # Decay rate.
            staircase=True)
        # Use simple momentum for the optimization.
        train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss,global_step=batch)

    # Create a summary to monitor learning_rate tensor
    tf.summary.scalar('learning_rate', learning_rate)

    # Get accuracy of model
    with tf.name_scope("ACC"):
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # Create a summary to monitor accuracy tensor
    tf.summary.scalar('acc', accuracy)

    # Merge all summaries into a single op
    merged_summary_op = tf.summary.merge_all()

    # Add ops to save and restore all the variables
    saver = tf.train.Saver()
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer(), feed_dict={is_training: True})

    # Training cycle
    total_batch = int(train_size / batch_size)

    # op to write logs to Tensorboard
    summary_writer = tf.summary.FileWriter(LOGS_DIRECTORY, graph=tf.get_default_graph())

    # Save the maximum accuracy value for validation data
    max_acc = 0.

    # Loop for epoch
    for epoch in range(training_epochs):

        # Random shuffling
        numpy.random.shuffle(train_total_data)
        train_data_ = train_total_data[:, :-num_labels]
        train_labels_ = train_total_data[:, -num_labels:]

        # Loop over all batches
        for i in range(total_batch):

            # Compute the offset of the current minibatch in the data.
            offset = (i * batch_size) % (train_size)
            batch_xs = train_data_[offset:(offset + batch_size), :]
            batch_ys = train_labels_[offset:(offset + batch_size), :]

            # Run optimization op (backprop), loss op (to get loss value)
            # and summary nodes
            _, train_accuracy, summary = sess.run([train_step, accuracy, merged_summary_op] , feed_dict={x: batch_xs, y_: batch_ys, is_training: True})

            # Write logs at every iteration
            summary_writer.add_summary(summary, epoch * total_batch + i)

            # Display logs
            if i % display_step == 0:
                print("Epoch:", '%04d,' % (epoch + 1),
                    "batch_index %4d/%4d, training accuracy %.5f" % (i, total_batch, train_accuracy))
                with open("log.out", "a") as log_file:
                    log_file.write(("Epoch: {}, batch_index: {}/{}, \
                        training accuracy: {}}\n".format(epoch + 1, i, total_batch, train_accuracy)))

            # Get accuracy for validation data
            if i % validation_step == 0:
                # Calculate accuracy
                validation_accuracy = sess.run(accuracy,
                feed_dict={x: validation_data, y_: validation_labels, is_training: False})

                print("Epoch:", '%04d,' % (epoch + 1),
                "batch_index %4d/%4d, validation accuracy %.5f" % (i, total_batch, validation_accuracy))

            # Save the current model if the maximum accuracy is updated
            if validation_accuracy > max_acc:
                max_acc = validation_accuracy
                save_path = saver.save(sess, MODEL_DIRECTORY)
                print("Model updated and saved in file: %s" % save_path)

    print("Optimization Finished!")

    # Restore variables from disk
    saver.restore(sess, MODEL_DIRECTORY)

    # Calculate accuracy for all mnist test images
    test_size = test_labels.shape[0]
    batch_size = TEST_BATCH_SIZE
    total_batch = int(test_size / batch_size)

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

if __name__ == '__main__':
    train(args)