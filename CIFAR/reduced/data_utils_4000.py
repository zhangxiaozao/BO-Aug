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

"""Data utils for CIFAR-10 and CIFAR-100."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import pickle
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

    all_labels = []
    test_labels = []

    self.good_policies = found_policies.good_policies()

    # Determine how many databatched to load
    num_data_batches_to_load = 5
    train_batches_to_load = num_data_batches_to_load
    assert hparams.train_size + hparams.validation_size <= 4000

    # Determine how many images we have loaded
    train_dataset_size = 800 * num_data_batches_to_load

    if hparams.dataset == 'cifar10':
      #all_data = np.empty((train_batches_to_load, 800, 3072), dtype=np.uint8)
      all_data = np.empty((1, train_dataset_size, 3072), dtype=np.uint8)
      test_data = np.empty((1, 10000, 3072), dtype=np.uint8)
    if hparams.dataset == 'cifar10':
      tf.logging.info('Cifar10')
      datafiles = ['reduced_train_batch']
      num_classes = 10

    if hparams.dataset != 'test':
      for file_num, f in enumerate(datafiles):
        d = unpickle(os.path.join(hparams.data_path, f))
        if f == 'test':
          test_data[0] = copy.deepcopy(d['data'])
          all_data = np.concatenate([all_data, test_data], axis=1)
        else:
          all_data[file_num] = copy.deepcopy(d['data'])
        if hparams.dataset == 'cifar10':
          labels = np.array(d['labels'])
        else:
          labels = np.array(d['fine_labels'])
        nsamples = len(labels)
        for idx in range(nsamples):
          all_labels.append(labels[idx])
    d_test = unpickle(os.path.join(hparams.data_path, 'test_batch'))
    test_data[0] = copy.deepcopy(d_test['data'])
    t_labels = np.array(d_test['labels'])
    for i in range(10000):
      test_labels.append(t_labels[i])

    all_data = all_data.reshape(train_dataset_size, 3072)
    all_data = all_data.reshape(-1, 3, 32, 32)
    all_data = all_data.transpose(0, 2, 3, 1).copy()
    all_data = all_data / 255.0
    test_data = test_data.reshape(10000, 3072)
    test_data = test_data.reshape(-1, 3, 32, 32)
    test_data = test_data.transpose(0, 2, 3, 1).copy()
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
        'In CIFAR10 loader, number of images: {}'.format(len(all_data)+len(test_data)))

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
    assert 4000 >= train_size + val_size
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
      epoch_policy = self.good_policies[np.random.choice(
          len(self.good_policies))]
      final_img = augmentation_transforms.apply_policy(
          epoch_policy, data)
      final_img = augmentation_transforms.random_flip(
          augmentation_transforms.zero_pad_and_crop(final_img, 4))
      # Apply cutout
      final_img = augmentation_transforms.cutout_numpy(final_img)
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


def unpickle(f):
  tf.logging.info('loading file: {}'.format(f))
  fo = tf.gfile.Open(f, 'rb')
  d = pickle.load(fo, encoding='latin1')
  fo.close()
  return d
