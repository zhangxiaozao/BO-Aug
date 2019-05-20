# Copyright 2018 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Data utils for SVHN."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from scipy.io import loadmat as load
import copy
import os
import augmentation_transforms
import numpy as np
import bo_policies as found_policies
import tensorflow as tf


# pylint:disable=logging-format-interpolation


class DataSet(object):
  """Dataset object that produces augmented training and eval data."""

  def __init__(self, hparams):
    self.hparams = hparams
    self.epochs = 0
    self.curr_train_index = 0

    self.good_policies = found_policies.good_policies()

    assert hparams.train_size + hparams.validation_size <= 1000

    # Determine how many images we have loaded
    train_dataset_size = 1000
    test_dataset_size = 800
    all_data = np.empty((train_dataset_size, 32, 32, 3), dtype=np.uint8)
    test_data = np.empty((test_dataset_size, 32, 32, 3), dtype=np.uint8)
    tf.logging.info('SVHN')
    datafile = 'reduced_train_32x32.mat'
    num_classes = 10

    d = load(os.path.join(hparams.data_path, datafile))
    all_data = copy.deepcopy(d['X'])
    all_labels = np.array(d['y'][0])

    d_test = load(os.path.join(hparams.data_path, 'validation_32x32.mat'))
    test_data = copy.deepcopy(d_test['X'])
    test_labels = np.array(d_test['y'][0])

    all_data = all_data / 255.0
    test_data = test_data / 255.0
    mean = augmentation_transforms.MEANS
    std = augmentation_transforms.STDS
    tf.logging.info('mean:{}    std: {}'.format(mean, std))

    all_data = (all_data - mean) / std
    all_labels = np.eye(num_classes)[np.array(all_labels, dtype=np.int32)]
    test_data = (test_data - mean) / std
    test_labels = np.eye(num_classes)[np.array(test_labels, dtype=np.int32)]
    assert len(all_data) == len(all_labels)
    tf.logging.info(
        'In SVHN loader, number of images: {}'.format(len(all_data)+len(test_data)))

    # Break off test data
    if hparams.eval_test:
      self.test_images = test_data
      self.test_labels = test_labels

    # Shuffle the rest of the data
    all_data = all_data[:train_dataset_size]
    all_labels = all_labels[:train_dataset_size]
    np.random.seed(0)
    perm = np.arange(len(all_data))
    np.random.shuffle(perm)
    all_data = all_data[perm]
    all_labels = all_labels[perm]

    # Break into train and val
    train_size, val_size = hparams.train_size, hparams.validation_size
    assert 1000 >= train_size + val_size
    self.train_images = all_data[:train_size]
    self.train_labels = all_labels[:train_size]
    self.val_images = all_data[train_size:train_size + val_size]
    self.val_labels = all_labels[train_size:train_size + val_size]
    self.num_train = self.train_images.shape[0]

  def next_batch(self):
    """Return the next minibatch of augmented data."""
    next_train_index = self.curr_train_index + self.hparams.batch_size
    if next_train_index > self.num_train:
      # Increase epoch number
      epoch = self.epochs + 1
      self.reset()
      self.epochs = epoch
    batched_data = (
        self.train_images[self.curr_train_index:
                          self.curr_train_index + self.hparams.batch_size],
        self.train_labels[self.curr_train_index:
                          self.curr_train_index + self.hparams.batch_size])
    final_imgs = []

    images, labels = batched_data
    for data in images:
      '''epoch_policy = self.good_policies[np.random.choice(
          len(self.good_policies))]
      final_img = augmentation_transforms.apply_policy(
          epoch_policy, data)'''
      final_img = augmentation_transforms.random_flip(
          augmentation_transforms.zero_pad_and_crop(data, 4))
      # Apply cutout
      #final_img = augmentation_transforms.cutout_numpy(final_img)
      final_imgs.append(final_img)
    batched_data = (np.array(final_imgs, np.float32), labels)
    self.curr_train_index += self.hparams.batch_size
    return batched_data

  def reset(self):
    """Reset training data and index into the training data."""
    self.epochs = 0
    # Shuffle the training data
    perm = np.arange(self.num_train)
    np.random.shuffle(perm)
    assert self.num_train == self.train_images.shape[
        0], 'Error incorrect shuffling mask'
    self.train_images = self.train_images[perm]
    self.train_labels = self.train_labels[perm]
    self.curr_train_index = 0