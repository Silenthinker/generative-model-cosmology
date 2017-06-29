# Some code was borrowed from https://github.com/petewarden/tensorflow_makefile/blob/master/tensorflow/models/image/mnist/convolutional.py

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import os
import csv
import sys
import random
from PIL import Image

import numpy as np
from scipy import ndimage
import scipy as sp

from six.moves import urllib

import tensorflow as tf

SOURCE_URL = 'http://yann.lecun.com/exdb/mnist/'
DATA_DIRECTORY = "data"

# Params for MNIST
NUM_CHANNELS = 1
PIXEL_DEPTH = 255
VALIDATION_SIZE = 5000  # Size of the validation set.

def normalize(data, pixel_depth=255):
    data = (data - (pixel_depth / 2.0)) / pixel_depth # normalization
    return data

# Download MNIST data
def maybe_download(filename):
    """Download the data from Yann's website, unless it's already here."""
    if not tf.gfile.Exists(DATA_DIRECTORY):
        tf.gfile.MakeDirs(DATA_DIRECTORY)
    filepath = os.path.join(DATA_DIRECTORY, filename)
    if not tf.gfile.Exists(filepath):
        filepath, _ = urllib.request.urlretrieve(SOURCE_URL + filename, filepath)
        with tf.gfile.GFile(filepath) as f:
            size = f.size()
        print('Successfully downloaded', filename, size, 'bytes.')
    return filepath

# Extract the images
def extract_data(filename, num_images, input_size):
    """Extract the images into a 4D tensor [image index, y, x, channels].

    Values are rescaled from [0, 255] down to [-0.5, 0.5].
    """
    print('Extracting', filename)
    with gzip.open(filename) as bytestream:
        bytestream.read(16)
        buf = bytestream.read(input_size * input_size * num_images * NUM_CHANNELS)
        data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
        data = (data - (PIXEL_DEPTH / 2.0)) / PIXEL_DEPTH # normalization
        data = data.reshape(num_images, input_size, input_size, NUM_CHANNELS)
        data = np.reshape(data, [num_images, -1])
    return data

# Extract the labels
def extract_labels(filename, num_images, num_classes):
    """Extract the labels into a vector of int64 label IDs."""
    print('Extracting', filename)
    with gzip.open(filename) as bytestream:
        bytestream.read(8)
        buf = bytestream.read(1 * num_images)
        labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
        num_labels_data = len(labels)
        one_hot_encoding = np.zeros((num_labels_data,num_classes))
        one_hot_encoding[np.arange(num_labels_data),labels] = 1
        one_hot_encoding = np.reshape(one_hot_encoding, [-1, num_classes])
    return one_hot_encoding

# Augment training data
def expand_training_data(images, labels, input_size, args):

    expanded_images = []
    expanded_labels = []

    j = 0 # counter
    for x, y in zip(images, labels):
        j = j+1
        if j%100==0:
            print ('expanding data : %03d / %03d' % (j,np.size(images,0)))

        # register original data
        expanded_images.append(x)
        expanded_labels.append(y)

        # get a value for the background
        # zero is the expected value, but median() is used to estimate background's value 
        bg_value = np.median(x) # this is regarded as background's value        
        image = np.reshape(x, (-1, input_size))

        for i in range(args.augment):
            # rotate the image with random degree
            angle = np.random.randint(-180,180,1)
            new_img = ndimage.rotate(image,angle,reshape=False, cval=bg_value)

            # shift the image with random distance
            shift = np.random.randint(-2, 2, 2)
            new_img_ = ndimage.shift(new_img,shift, cval=bg_value)

            # register new training data
            expanded_images.append(np.reshape(new_img_, input_size*input_size))
            expanded_labels.append(y)

    # images and labels are concatenated for random-shuffle at each epoch
    # notice that pair of image and label should not be broken
    expanded_train_total_data = np.concatenate((expanded_images, expanded_labels), axis=1)
    np.random.shuffle(expanded_train_total_data)

    return expanded_train_total_data

# Prepare MNISt data
def prepare_MNIST_data(args):
    use_data_augmentation = True if args.augment > 0 else False
    input_size = args.input_size
    num_classes = args.num_classes
    # Get the data.
    train_data_filename = maybe_download('train-images-idx3-ubyte.gz')
    train_labels_filename = maybe_download('train-labels-idx1-ubyte.gz')
    test_data_filename = maybe_download('t10k-images-idx3-ubyte.gz')
    test_labels_filename = maybe_download('t10k-labels-idx1-ubyte.gz')

    # Extract it into np arrays.
    train_data = extract_data(train_data_filename, 60000, input_size)
    train_labels = extract_labels(train_labels_filename, 60000, num_classes)
    test_data = extract_data(test_data_filename, 10000, input_size)
    test_labels = extract_labels(test_labels_filename, 10000, num_classes)

    # Generate a validation set.
    validation_data = train_data[:VALIDATION_SIZE, :]
    validation_labels = train_labels[:VALIDATION_SIZE,:]
    train_data = train_data[VALIDATION_SIZE:, :]
    train_labels = train_labels[VALIDATION_SIZE:,:]

    # Concatenate train_data & train_labels for random shuffle
    if use_data_augmentation:
        train_total_data = expand_training_data(train_data, train_labels, input_size, args)
    else:
        train_total_data = np.concatenate((train_data, train_labels), axis=1)

    train_size = train_total_data.shape[0]

    return train_total_data, train_size, validation_data, validation_labels, test_data, test_labels

def prepare_cosmology_data(args):
    """ Prepare cosmology data
    """
    def csv_to_dict(csv_path):
        with open(csv_path,'r') as fp:
            csv_fp=csv.reader(fp)
            next(csv_fp)
            d = dict(filter(None, csv_fp))
            return d

    def one_hot_encoding(labels):
        res = np.zeros((labels.shape[0], args.num_classes))
        res[range(labels.shape[0]), labels] = 1
        res = np.reshape(res, (-1, args.num_classes))
        return res

    data_path = args.data_dir
    input_size = args.input_size
    use_data_augmentation = True if args.augment > 0 else False

    train_ratio, val_ratio = 0.7, 0.1
    test_ratio = 1 - train_ratio - val_ratio

    feat_size = input_size*input_size

    # Paths
    if args.num_classes > 1:
        resized_path = os.path.join(data_path, "labeled_resized")
        labeled_path=os.path.join(data_path,"labeled")
        label_file=os.path.join(data_path,"labeled.csv")
    else:
        resized_path = os.path.join(data_path, "scored_resized")
        labeled_path=os.path.join(data_path,"scored")
        label_file=os.path.join(data_path,"scored.csv")

    # Initialization
    label_dict=csv_to_dict(label_file)
    img_prefixes=list(label_dict.keys())
    # To determine if resize
    sample_image = Image.open(os.path.join(labeled_path,"{}.png".format(img_prefixes[0])))
    if np.array(sample_image.getdata()).shape[0] == input_size:
        resize = False
    else:
        resize = True
        if not os.path.exists(resized_path):
            os.makedirs(resized_path)
    random.shuffle(img_prefixes)
    n_train=int(train_ratio*len(img_prefixes))
    n_test=len(img_prefixes)-n_train
    n_val = int(val_ratio*len(img_prefixes))
    train_mat=np.zeros((n_train, feat_size))
    _train_y=np.zeros(n_train, dtype=int)
    test_mat=np.zeros((n_test,feat_size))
    _test_y=np.zeros(n_test, dtype=int)
    train_idx=0
    test_idx=0

    # Assemble train/test feature matrices / label vectors
    for idx,img_prefix in enumerate(img_prefixes):
        print("Image: {}/{}".format(idx+1,len(img_prefixes)))
        raw_image=Image.open(os.path.join(labeled_path,"{}.png".format(img_prefix)))
        if resize:
            resized_image = sp.misc.imresize(raw_image, (input_size, input_size))
            sp.misc.imsave(os.path.join(resized_path, "{}.png".format(img_prefix)), resized_image)
            img_arr=resized_image.astype(np.uint8)
        else:
            img_arr=np.array(raw_image.getdata()).reshape(raw_image.size[0],raw_image.size[1]).astype(np.uint8)
        img_arr = img_arr.flatten()
        img_arr = normalize(img_arr) # normalize to [-0.5, 0.5]
        label = float(label_dict[img_prefix])

        if idx<n_train:
            train_mat[train_idx,:]=img_arr
            _train_y[train_idx]=label
            train_idx+=1
        else:
            test_mat[test_idx,:] = img_arr
            _test_y[test_idx]=label
            test_idx+=1

    # generate validation data
    validation_data = test_mat[0:n_val, :]
    # generate test data
    test_data = test_mat[n_val:, :]
    # convert to one hot encoding
    if args.num_classes > 1: # if classification problem
        train_y = one_hot_encoding(_train_y)
        validation_y = one_hot_encoding(_test_y[0:n_val])
        test_y = one_hot_encoding(_test_y[n_val:])
    else: # if regression
        train_y = np.reshape(_train_y, (-1, 1))
        validation_y = np.reshape(_test_y[0:n_val], (-1, 1))
        test_y = np.reshape(_test_y[n_val:], (-1, 1))

    # Concatenate train_data & train_y for random shuffle
    if use_data_augmentation:
        train_total_data = expand_training_data(train_mat, train_y, input_size, args)
    else:
        train_total_data = np.concatenate((train_mat, train_y), axis=1)
    train_size = train_total_data.shape[0]

    return train_total_data, train_size, validation_data, validation_y, test_data, test_y

def prepare_cosmology_test_data(args):
    """ Prepare cosmology test data
    """

    data_path = args.data_dir
    input_size = args.input_size
    feat_size = input_size*input_size

    # Paths
    resized_path = os.path.join(data_path, "query_resized")
    query_path=os.path.join(data_path,"query")

    # Initialization
    img_prefixes = [f.split(".")[0] for f in os.listdir(query_path) if f.endswith(".png")]
    # To determine if resize
    sample_image = Image.open(os.path.join(query_path,"{}.png".format(img_prefixes[0])))
    if np.array(sample_image.getdata()).shape[0] == input_size:
        resize = False
    else:
        resize = True
        if not os.path.exists(resized_path):
            os.makedirs(resized_path)
    test_mat=np.zeros((len(img_prefixes), feat_size))

    # Assemble train/test feature matrices / label vectors
    for idx, img_prefix in enumerate(img_prefixes):
        print("Image: {}/{}".format(idx+1,len(img_prefixes)))
        raw_image=Image.open(os.path.join(query_path,"{}.png".format(img_prefix)))
        if resize:
            resized_image = sp.misc.imresize(raw_image, (input_size, input_size))
            sp.misc.imsave(os.path.join(resized_path, "{}.png".format(img_prefix)), resized_image)
            img_arr=resized_image.astype(np.uint8)
        else:
            img_arr=np.array(raw_image.getdata()).reshape(raw_image.size[0],raw_image.size[1]).astype(np.uint8)
        img_arr = img_arr.flatten()
        img_arr = normalize(img_arr) # normalize to [-0.5, 0.5]
        test_mat[idx,:] = img_arr

    return test_mat
