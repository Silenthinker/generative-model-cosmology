# Some code was borrowed from https://github.com/petewarden/tensorflow_makefile/blob/master/tensorflow/models/image/mnist/convolutional.py

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import os

import numpy
from scipy import ndimage

from six.moves import urllib

import tensorflow as tf

SOURCE_URL = 'http://yann.lecun.com/exdb/mnist/'
DATA_DIRECTORY = "data"

# Params for MNIST
NUM_CHANNELS = 1
PIXEL_DEPTH = 255
VALIDATION_SIZE = 5000  # Size of the validation set.

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
        data = numpy.frombuffer(buf, dtype=numpy.uint8).astype(numpy.float32)
        data = (data - (PIXEL_DEPTH / 2.0)) / PIXEL_DEPTH # normalization
        data = data.reshape(num_images, input_size, input_size, NUM_CHANNELS)
        data = numpy.reshape(data, [num_images, -1])
    return data

# Extract the labels
def extract_labels(filename, num_images, num_classes):
    """Extract the labels into a vector of int64 label IDs."""
    print('Extracting', filename)
    with gzip.open(filename) as bytestream:
        bytestream.read(8)
        buf = bytestream.read(1 * num_images)
        labels = numpy.frombuffer(buf, dtype=numpy.uint8).astype(numpy.int64)
        num_labels_data = len(labels)
        one_hot_encoding = numpy.zeros((num_labels_data,num_classes))
        one_hot_encoding[numpy.arange(num_labels_data),labels] = 1
        one_hot_encoding = numpy.reshape(one_hot_encoding, [-1, num_classes])
    return one_hot_encoding

# Augment training data
def expand_training_data(images, labels, input_size):

    expanded_images = []
    expanded_labels = []

    j = 0 # counter
    for x, y in zip(images, labels):
        j = j+1
        if j%100==0:
            print ('expanding data : %03d / %03d' % (j,numpy.size(images,0)))

        # register original data
        expanded_images.append(x)
        expanded_labels.append(y)

        # get a value for the background
        # zero is the expected value, but median() is used to estimate background's value 
        bg_value = numpy.median(x) # this is regarded as background's value        
        image = numpy.reshape(x, (-1, input_size))

        for i in range(4):
            # rotate the image with random degree
            angle = numpy.random.randint(-15,15,1)
            new_img = ndimage.rotate(image,angle,reshape=False, cval=bg_value)

            # shift the image with random distance
            shift = numpy.random.randint(-2, 2, 2)
            new_img_ = ndimage.shift(new_img,shift, cval=bg_value)

            # register new training data
            expanded_images.append(numpy.reshape(new_img_, input_size*input_size))
            expanded_labels.append(y)

    # images and labels are concatenated for random-shuffle at each epoch
    # notice that pair of image and label should not be broken
    expanded_train_total_data = numpy.concatenate((expanded_images, expanded_labels), axis=1)
    numpy.random.shuffle(expanded_train_total_data)

    return expanded_train_total_data

# Prepare MNISt data
def prepare_MNIST_data(args):
    use_data_augmentation = args.augment
    input_size = args.input_size
    num_classes = args.num_classes
    # Get the data.
    train_data_filename = maybe_download('train-images-idx3-ubyte.gz')
    train_labels_filename = maybe_download('train-labels-idx1-ubyte.gz')
    test_data_filename = maybe_download('t10k-images-idx3-ubyte.gz')
    test_labels_filename = maybe_download('t10k-labels-idx1-ubyte.gz')

    # Extract it into numpy arrays.
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
        train_total_data = expand_training_data(train_data, train_labels, input_size)
    else:
        train_total_data = numpy.concatenate((train_data, train_labels), axis=1)

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

    data_path = args.data_dir
    input_size = args.input_size
    use_data_augmentation = args.augment

    train_ratio, val_ratio = 0.7, 0.1
    test_ratio = 1 - train_ratio - val_ratio

    feat_size = input_size*input_size

    # Paths
    resized_path = os.path.join(data_path, "resized")
    labeled_path=os.path.join(data_path,"labeled")
    label_file=os.path.join(data_path,"labeled.csv")

    # Initialization
    label_dict=csv_to_dict(label_file)
    img_prefixes=list(label_dict.keys())
    # To determine if resize
    sample_image = Image.open(os.path.join(labeled_path,"{}.png".format(img_prefix[0])))
    if sample_image.shape[0] == input_size:
        resize = False
    else:
        resize = True
    random.shuffle(img_prefixes)
    n_train=int(train_ratio*len(img_prefixes))
    n_test=len(img_prefixes)-n_train
    n_val = int(val_ratio*len(img_prefixes))
    train_mat=np.zeros((n_train, feat_size))
    train_y=np.zeros(n_train)
    test_mat=np.zeros((n_test,feat_size))
    test_y=np.zeros(n_test)
    train_idx=0
    test_idx=0

    # Assemble train/test feature matrices / label vectors
    for idx,img_prefix in enumerate(img_prefixes):
        print("Image: {}/{}".format(idx+1,len(img_prefixes)))
        raw_image=Image.open(os.path.join(labeled_path,"{}.png".format(img_prefix)))
        if resize:
            resized_image = sp.misc.imresize(raw_image, (size, size))
            sp.misc.imsave(os.path.join(resized_path, "{}.png".format(img_prefix)), resized_image)
            img_arr=resized_image.astype(np.uint8)
        else:
            img_arr=np.array(raw_image.getdata()).reshape(raw_image.size[0],raw_image.size[1]).astype(np.uint8)
        img_arr = img_arr.flatten()
        label=float(label_dict[img_prefix])

        if idx<n_train:
            train_mat[train_idx,:]=img_arr
            train_y[train_idx]=label
            train_idx+=1
        else:
            test_mat[test_idx,:] = img_arr
            test_y[test_idx]=label
            test_idx+=1
    # generate validation data
    validation_data = test_mat[0:n_val, :]
    validation_labels = test_y[0:n_val, :]
    # generate test data
    test_data = test_mat[n_val:, :]
    test_labels = test_y[n_val:, :]

    # Concatenate train_data & train_labels for random shuffle
    if use_data_augmentation:
        train_total_data = expand_training_data(train_mat, train_y, input_size)
    else:
        train_total_data = numpy.concatenate((train_mat, train_y), axis=1)

    train_size = train_total_data.shape[0]

    return train_total_data, train_size, validation_data, validation_labels, test_data, test_labels
