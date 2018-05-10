'''Imports CIFAR-10 data.'''

import numpy as np
import pickle
import sys
import math

def load_CIFAR10_batch(filename):
    '''load data from single CIFAR-10 file'''

    with open(filename, 'rb') as f:
        if sys.version_info[0] < 3:
            dict = pickle.load(f)
        else:
            dict = pickle.load(f, encoding='latin1')
            x = dict['data']
            y = dict['labels']
            x = x.astype(float)
            y = np.array(y)
            return x, y

def load_data(data_dir):
    '''load all CIFAR-10 data and merge training batches'''

    batches_dir = data_dir + 'cifar-10-batches-py/'
  
    xs = []
    ys = []
    for i in range(1, 6):
        filename = batches_dir + 'data_batch_' + str(i)
        X, Y = load_CIFAR10_batch(filename)
        xs.append(X)
        ys.append(Y)

    x_train = np.concatenate(xs)
    y_train = np.concatenate(ys)
    del xs, ys

    x_test, y_test = load_CIFAR10_batch(batches_dir + 'test_batch')

    classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    data_dict = {
        'images_train': x_train.astype('float32'),
        'labels_train': y_train.astype('int32'),
        'images_test': x_test.astype('float32'),
        'labels_test': y_test.astype('int32'),
        'classes': classes
    }
    return data_dict

def gen_training_batch(data_dict, batch_size, num_iter=0):
    inputs = data_dict['images_train']
    labels = data_dict['labels_train']
    num_samples = len(inputs)
    if num_iter == 0:
        num_iter = int(num_samples / batch_size)
    index = num_samples
    for i in range(num_iter):
        index += batch_size
        if (index + batch_size > num_samples):
            index = 0
            shuffled_indices = np.random.permutation(np.arange(num_samples))
            inputs = inputs[shuffled_indices]
            labels = labels[shuffled_indices]
        yield inputs[index:index + batch_size], labels[index:index + batch_size]

def gen_inf_test_batch(data_dict, batch_size):
    inputs = data_dict['images_test']
    labels = data_dict['labels_test']
    num_samples = len(inputs)
    index = num_samples
    while True:
        index += batch_size
        if (index + batch_size > num_samples):
            index = 0
            shuffled_indices = np.random.permutation(np.arange(num_samples))
            inputs = inputs[shuffled_indices]
            labels = labels[shuffled_indices]
        yield inputs[index:index + batch_size], labels[index:index + batch_size]

def gen_test_batch(data_dict, batch_size):
    inputs = data_dict['images_test']
    labels = data_dict['labels_test']
    num_samples = len(inputs)
    num_iter = math.ceil(num_samples / batch_size)
    for i in range(num_iter):
        lower = i*batch_size
        upper = min((i+1)*batch_size, num_samples)
        yield inputs[lower:upper], labels[lower:upper]


def gen_training_batch_from_dataset(inputs, labels, batch_size, num_iter=0):
    num_samples = len(inputs)
    if num_iter == 0:
        num_iter = int(num_samples / batch_size)
    index = num_samples
    for i in range(num_iter):
        index += batch_size
        if (index + batch_size > num_samples):
            index = 0
            shuffled_indices = np.random.permutation(np.arange(num_samples))
            inputs = inputs[shuffled_indices]
            labels = labels[shuffled_indices]
        yield inputs[index:index + batch_size], labels[index:index + batch_size]

def gen_test_batch_from_dataset(inputs, labels, batch_size, num_iter=0):
    num_samples = len(inputs)
    num_iter = math.ceil(num_samples / batch_size)
    for i in range(num_iter):
        lower = i*batch_size
        upper = min((i+1)*batch_size, num_samples)
        yield inputs[lower:upper], labels[lower:upper]

def gen_inf_test_batch_from_dataset(inputs, labels, batch_size):
    num_samples = len(inputs)
    index = num_samples
    while True:
        index += batch_size
        if (index + batch_size > num_samples):
            index = 0
            shuffled_indices = np.random.permutation(np.arange(num_samples))
            inputs = inputs[shuffled_indices]
            labels = labels[shuffled_indices]
        yield inputs[index:index + batch_size], labels[index:index + batch_size]
