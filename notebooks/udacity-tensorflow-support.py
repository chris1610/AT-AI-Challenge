# From https://github.com/udacity/deep-learning
# Copyright (c) 2017 Udacity, Inc.
# MIT License https://opensource.org/licenses/MIT

from six.moves import cPickle as pickle
from scipy import ndimage
import math
import numpy as np
import os
from pathlib import Path
import tensorflow as tf

image_size = 28  # Pixel width and height.
num_labels = 10
pixel_depth = 255.0  # Number of levels per pixel.

def accuracy(predictions, labels):
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
          / predictions.shape[0])

def load_letter(folder, min_num_images):
    """Load the data for a single letter label."""
    image_files = list(folder.iterdir())
    dataset = np.ndarray(shape=(len(image_files), image_size, image_size),
                         dtype=np.float32)
    print(folder)
    num_images = 0
    for image in image_files:
        image_file = folder.joinpath(image)
        try:
            image_data = (ndimage.imread(image_file).astype(float) - 
                          pixel_depth / 2) / pixel_depth
            if image_data.shape != (image_size, image_size):
                raise Exception('Unexpected image shape: %s' % str(image_data.shape))
            dataset[num_images, :, :] = image_data
            num_images = num_images + 1
        except IOError as e:
            print('Could not read:', image_file, ':', e, '- it\'s ok, skipping.')
    
    dataset = dataset[0:num_images, :, :]
    if num_images < min_num_images:
        raise Exception('Many fewer images than expected: %d < %d' % 
                        (num_images, min_num_images))
    
    print('Full dataset tensor:', dataset.shape)
    print('Mean:', np.mean(dataset))
    print('Standard deviation:', np.std(dataset))
    return dataset
        
def maybe_pickle(data_folders, min_num_images_per_class, force=False):
    dataset_names = []
    data_folders = (i for i in data_folders if i.is_dir())
    for folder in data_folders:
        set_filename = str(folder) + '.pickle'
        dataset_names.append(set_filename)
        if os.path.exists(set_filename) and not force:
            # You may override by setting force=True.
            print('%s already present - Skipping pickling.' % set_filename)
        else:
            print('Pickling %s.' % set_filename)
            dataset = load_letter(folder, min_num_images_per_class)
            try:
                with open(set_filename, 'wb') as f:
                    pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)
            except Exception as e:
                print('Unable to save data to', set_filename, ':', e) 
    return dataset_names

def make_arrays(nb_rows, img_size):
    if nb_rows:
        dataset = np.ndarray((nb_rows, img_size, img_size), dtype=np.float32)
        labels = np.ndarray(nb_rows, dtype=np.int32)
    else:
        dataset, labels = None, None
    return dataset, labels

def randomize(dataset, labels):
    permutation = np.random.permutation(labels.shape[0])
    shuffled_dataset = dataset[permutation,:,:]
    shuffled_labels = labels[permutation]
    return shuffled_dataset, shuffled_labels

def reformat(dataset, labels):
    dataset = dataset.reshape((-1, image_size * image_size)).astype(np.float32)
    # Map 0 to [1.0, 0.0, 0.0 ...], 1 to [0.0, 1.0, 0.0 ...]
    labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
    return dataset, labels

# Update of udacity content.
def merge_datasets(pickle_files, dataset_size):
    """
    Merge multiple glyph pickle files into nd-array dataset and nd-array labels
    for model evaluation.
    Simplification from https://github.com/udacity/deep-learning
    """
    num_classes = len(pickle_files)
    dataset, labels = make_arrays(dataset_size, image_size)
    size_per_class = dataset_size // num_classes
    
    start_t = 0
    for label, pickle_file in enumerate(pickle_files):       
        try:
            with open(pickle_file, 'rb') as f:
                letter_set = pickle.load(f)
                end_t = start_t + size_per_class
                np.random.shuffle(letter_set)
                dataset[start_t:end_t, :, :] = letter_set[0:size_per_class]
                labels[start_t:end_t] = label
                start_t = end_t
        except Exception as e:
            print('Unable to process data from', pickle_file, ':', e)
            raise
    
    return dataset, labels