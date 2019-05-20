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

"""AutoAugment Train/Eval module.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import contextlib
import os
os.environ['CUDA_VISIBLE_DEVICES']='1'
import time

import custom_ops as ops
import data_utils_4000
import helper_utils
import numpy as np
from shake_drop import build_shake_drop_model
from shake_shake import build_shake_shake_model
import tensorflow as tf
from wrn import build_wrn_model
import bo_policies

tf.flags.DEFINE_string('model_name', 'wrn',
                       'wrn, shake_shake_32, shake_shake_96, shake_shake_112, '
                       'pyramid_net')
tf.flags.DEFINE_string('checkpoint_dir', '../tmp', 'Training Directory.')
tf.flags.DEFINE_string('data_path', '../dataset',
                       'Directory where dataset is located.')
tf.flags.DEFINE_string('dataset', 'cifar10',
                       'Dataset to train with. Either cifar10 or cifar100')
tf.flags.DEFINE_integer('use_cpu', 0, '1 if use CPU, else GPU.')

FLAGS = tf.flags.FLAGS

arg_scope = tf.contrib.framework.arg_scope


def setup_arg_scopes(is_training):
  """Sets up the argscopes that will be used when building an image model.

  Args:
    is_training: Is the model training or not.

  Returns:
    Arg scopes to be put around the model being constructed.
  """

  batch_norm_decay = 0.9
  batch_norm_epsilon = 1e-5
  batch_norm_params = {
      # Decay for the moving averages.
      'decay': batch_norm_decay,
      # epsilon to prevent 0s in variance.
      'epsilon': batch_norm_epsilon,
      'scale': True,
      # collection containing the moving mean and moving variance.
      'is_training': is_training,
  }

  scopes = []

  scopes.append(arg_scope([ops.batch_norm], **batch_norm_params))
  return scopes


def build_model(inputs, num_classes, is_training, hparams):
  """Constructs the vision model being trained/evaled.

  Args:
    inputs: input features/images being fed to the image model build built.
    num_classes: number of output classes being predicted.
    is_training: is the model training or not.
    hparams: additional hyperparameters associated with the image model.

  Returns:
    The logits of the image model.
  """
  scopes = setup_arg_scopes(is_training)
  with contextlib.ExitStack() as stack:
    tuple(stack.enter_context(cm) for cm in scopes)
    if hparams.model_name == 'pyramid_net':
      logits = build_shake_drop_model(
          inputs, num_classes, is_training)
    elif hparams.model_name == 'wrn':
      logits = build_wrn_model(
          inputs, num_classes, hparams.wrn_size)
    elif hparams.model_name == 'shake_shake':
      logits = build_shake_shake_model(
          inputs, num_classes, hparams, is_training)
  return logits


class CifarModel(object):
  """Builds an image model for Cifar10/Cifar100."""

  def __init__(self, hparams):
    self.hparams = hparams

  def build(self, mode):
    """Construct the cifar model."""
    assert mode in ['train', 'eval']
    self.mode = mode
    self._setup_misc(mode)
    self._setup_images_and_labels()
    self._build_graph(self.images, self.labels, mode)

    self.init = tf.group(tf.global_variables_initializer(),
                         tf.local_variables_initializer())

  def _setup_misc(self, mode):
    """Sets up miscellaneous in the cifar model constructor."""
    self.lr_rate_ph = tf.Variable(0.0, name='lrn_rate', trainable=False)
    self.reuse = None if (mode == 'train') else True
    self.batch_size = self.hparams.batch_size
    if mode == 'eval':
      self.batch_size = 25

  def _setup_images_and_labels(self):
    """Sets up image and label placeholders for the cifar model."""
    if FLAGS.dataset == 'cifar10':
      self.num_classes = 10
    else:
      self.num_classes = 100
    self.images = tf.placeholder(tf.float32, [self.batch_size, 32, 32, 3])
    self.labels = tf.placeholder(tf.float32,
                                 [self.batch_size, self.num_classes])

  def assign_epoch(self, session, epoch_value):
    session.run(self._epoch_update, feed_dict={self._new_epoch: epoch_value})

  def _build_graph(self, images, labels, mode):
    """Constructs the TF graph for the cifar model.

    Args:
      images: A 4-D image Tensor
      labels: A 2-D labels Tensor.
      mode: string indicating training mode ( e.g., 'train', 'valid', 'test').
    """
    is_training = 'train' in mode
    if is_training:
      self.global_step = tf.train.get_or_create_global_step()

    logits = build_model(
        images,
        self.num_classes,
        is_training,
        self.hparams)
    self.predictions, self.cost = helper_utils.setup_loss(
        logits, labels)
    self.accuracy, self.eval_op = tf.metrics.accuracy(
        tf.argmax(labels, 1), tf.argmax(self.predictions, 1))
    self._calc_num_trainable_params()

    # Adds L2 weight decay to the cost
    self.cost = helper_utils.decay_weights(self.cost,
                                           self.hparams.weight_decay_rate)

    if is_training:
      self._build_train_op()

    # Setup checkpointing for this child model
    # Keep 2 or more checkpoints around during training.
    with tf.device('/cpu:0'):
      self.saver = tf.train.Saver(max_to_keep=2)

    self.init = tf.group(tf.global_variables_initializer(),
                         tf.local_variables_initializer())

  def _calc_num_trainable_params(self):
    self.num_trainable_params = np.sum([
        np.prod(var.get_shape().as_list()) for var in tf.trainable_variables()
    ])
    tf.logging.info('number of trainable params: {}'.format(
        self.num_trainable_params))

  def _build_train_op(self):
    """Builds the train op for the cifar model."""
    hparams = self.hparams
    tvars = tf.trainable_variables()
    grads = tf.gradients(self.cost, tvars)
    if hparams.gradient_clipping_by_global_norm > 0.0:
      grads, norm = tf.clip_by_global_norm(
          grads, hparams.gradient_clipping_by_global_norm)
      tf.summary.scalar('grad_norm', norm)

    # Setup the initial learning rate
    initial_lr = self.lr_rate_ph
    optimizer = tf.train.MomentumOptimizer(
        initial_lr,
        0.9,
        use_nesterov=True)

    self.optimizer = optimizer
    apply_op = optimizer.apply_gradients(
        zip(grads, tvars), global_step=self.global_step, name='train_step')
    train_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies([apply_op]):
      self.train_op = tf.group(*train_ops)


class CifarModelTrainer(object):
  """Trains an instance of the CifarModel class."""

  def __init__(self, hparams):
    self._session = None
    self.hparams = hparams

    self.model_dir = os.path.join(FLAGS.checkpoint_dir, 'model')
    self.log_dir = os.path.join(FLAGS.checkpoint_dir, 'log')
    # Set the random seed to be sure the same validation set
    # is used for each model
    np.random.seed(0)
    self.data_loader = data_utils_4000.DataSet(hparams)
    np.random.seed()  # Put the random seed back to random
    self.data_loader.reset()

  def save_model(self, step=None):
    """Dumps model into the backup_dir.

    Args:
      step: If provided, creates a checkpoint with the given step
        number, instead of overwriting the existing checkpoints.
    """
    model_save_name = os.path.join(self.model_dir, 'model.ckpt')
    if not tf.gfile.IsDirectory(self.model_dir):
      tf.gfile.MakeDirs(self.model_dir)
    self.saver.save(self.session, model_save_name, global_step=step)
    tf.logging.info('Saved child model')

  def extract_model_spec(self):
    """Loads a checkpoint with the architecture structure stored in the name."""
    checkpoint_path = tf.train.latest_checkpoint(self.model_dir)
    if checkpoint_path is not None:
      self.saver.restore(self.session, checkpoint_path)
      tf.logging.info('Loaded child model checkpoint from %s',
                      checkpoint_path)
    else:
      self.save_model(step=0)

  def eval_child_model(self, model, data_loader, mode):
    """Evaluate the child model.

    Args:
      model: image model that will be evaluated.
      data_loader: dataset object to extract eval data from.
      mode: will the model be evalled on train, val or test.

    Returns:
      Accuracy of the model on the specified dataset.
    """
    tf.logging.info('Evaluating child model in mode %s', mode)
    while True:
      try:
        with self._new_session(model):
          accuracy = helper_utils.eval_child_model(
              self.session,
              model,
              data_loader,
              mode)
          tf.logging.info('Eval child model accuracy: {}'.format(accuracy))
          # If epoch trained without raising the below errors, break
          # from loop.
          break
      except (tf.errors.AbortedError, tf.errors.UnavailableError) as e:
        tf.logging.info('Retryable error caught: %s.  Retrying.', e)

    return accuracy

  @contextlib.contextmanager
  def _new_session(self, m):
    """Creates a new session for model m."""
    # Create a new session for this model, initialize
    # variables, and save / restore from
    # checkpoint.
    self._session = tf.Session(
        '',
        config=tf.ConfigProto(
            allow_soft_placement=True, log_device_placement=False))
    self.session.run(m.init)

    # Load in a previous checkpoint, or save this one
    self.extract_model_spec()
    try:
      yield
    finally:
      tf.Session.reset('')
      self._session = None

  def _build_models(self):
    """Builds the image models for train and eval."""
    # Determine if we should build the train and eval model. When using
    # distributed training we only want to build one or the other and not both.
    with tf.variable_scope('model', use_resource=False):
      m = CifarModel(self.hparams)
      m.build('train')
      self._num_trainable_params = m.num_trainable_params
      self._saver = m.saver
    with tf.variable_scope('model', reuse=True, use_resource=False):
      meval = CifarModel(self.hparams)
      meval.build('eval')
    return m, meval

  def _calc_starting_epoch(self, m):
    """Calculates the starting epoch for model m based on global step."""
    hparams = self.hparams
    batch_size = hparams.batch_size
    steps_per_epoch = int(hparams.train_size / batch_size)
    with self._new_session(m):
      curr_step = self.session.run(m.global_step)
    total_steps = steps_per_epoch * hparams.num_epochs
    epochs_left = (total_steps - curr_step) // steps_per_epoch
    starting_epoch = hparams.num_epochs - epochs_left
    return starting_epoch

  def _run_training_loop(self, m, curr_epoch):
    """Trains the cifar model `m` for one epoch."""
    start_time = time.time()
    while True:
      try:
        with self._new_session(m):
          train_accuracy = helper_utils.run_epoch_training(
              self.session, m, self.data_loader, curr_epoch)
          tf.logging.info('Saving model after epoch')
          self.save_model(step=curr_epoch)
          break
      except (tf.errors.AbortedError, tf.errors.UnavailableError) as e:
        tf.logging.info('Retryable error caught: %s.  Retrying.', e)
    tf.logging.info('Finished epoch: {}'.format(curr_epoch))
    tf.logging.info('Epoch time(min): {}'.format(
        (time.time() - start_time) / 60.0))
    return train_accuracy

  def _compute_final_accuracies(self, meval):
    """Run once training is finished to compute final val/test accuracies."""
    valid_accuracy = self.eval_child_model(meval, self.data_loader, 'val')
    if self.hparams.eval_test:
      test_accuracy = self.eval_child_model(meval, self.data_loader, 'test')
    else:
      test_accuracy = 0
    tf.logging.info('Test Accuracy: {}'.format(test_accuracy))
    return valid_accuracy, test_accuracy

  def run_model(self):
    """Trains and evalutes the image model."""
    hparams = self.hparams

    # Build the child graph
    with tf.Graph().as_default(), tf.device(
        '/cpu:0' if FLAGS.use_cpu else '/gpu:0'):
      m, meval = self._build_models()

      # Figure out what epoch we are on
      starting_epoch = self._calc_starting_epoch(m)

      # Run the validation error right at the beginning
      valid_accuracy = self.eval_child_model(
          meval, self.data_loader, 'val')
      tf.logging.info('Before Training Epoch: {}     Val Acc: {}'.format(
          starting_epoch, valid_accuracy))
      training_accuracy = None

      for curr_epoch in range(starting_epoch, hparams.num_epochs):

        # Run one training epoch
        training_accuracy = self._run_training_loop(m, curr_epoch)

        valid_accuracy = self.eval_child_model(
            meval, self.data_loader, 'test')
        tf.logging.info('Epoch: {}    test Acc: {}'.format(
            curr_epoch, valid_accuracy))

      valid_accuracy, test_accuracy = self._compute_final_accuracies(
          meval)

    tf.logging.info(
        'Train Acc: {}    Valid Acc: {}     Test Acc: {}'.format(
            training_accuracy, valid_accuracy, test_accuracy))

  @property
  def saver(self):
    return self._saver

  @property
  def session(self):
    return self._session

  @property
  def num_trainable_params(self):
    return self._num_trainable_params


def main(_):
  if FLAGS.dataset not in ['cifar10', 'cifar100']:
    raise ValueError('Invalid dataset: %s' % FLAGS.dataset)
  hparams = tf.contrib.training.HParams(
      train_size=4000,
      validation_size=0,
      eval_test=1,
      dataset=FLAGS.dataset,
      data_path=FLAGS.data_path,
      batch_size=128,
      gradient_clipping_by_global_norm=5.0)
  if FLAGS.model_name == 'wrn':
    hparams.add_hparam('model_name', 'wrn')
    hparams.add_hparam('num_epochs', 200)
    hparams.add_hparam('wrn_size', 160)
    hparams.add_hparam('lr', 0.02)
    hparams.add_hparam('weight_decay_rate', 1e-3)
  elif FLAGS.model_name == 'shake_shake_32':
    hparams.add_hparam('model_name', 'shake_shake')
    hparams.add_hparam('num_epochs', 1800)
    hparams.add_hparam('shake_shake_widen_factor', 2)
    hparams.add_hparam('lr', 0.01)
    hparams.add_hparam('weight_decay_rate', 0.001)
  elif FLAGS.model_name == 'shake_shake_96':
    hparams.add_hparam('model_name', 'shake_shake')
    hparams.add_hparam('num_epochs', 1800)
    hparams.add_hparam('shake_shake_widen_factor', 6)
    hparams.add_hparam('lr', 0.02)
    hparams.add_hparam('weight_decay_rate', 0.001)
  elif FLAGS.model_name == 'shake_shake_112':
    hparams.add_hparam('model_name', 'shake_shake')
    hparams.add_hparam('num_epochs', 1800)
    hparams.add_hparam('shake_shake_widen_factor', 7)
    hparams.add_hparam('lr', 0.01)
    hparams.add_hparam('weight_decay_rate', 0.001)
  elif FLAGS.model_name == 'pyramid_net':
    hparams.add_hparam('model_name', 'pyramid_net')
    hparams.add_hparam('num_epochs', 1800)
    hparams.add_hparam('lr', 0.05)
    hparams.add_hparam('weight_decay_rate', 5e-5)
    hparams.batch_size = 64
  else:
    raise ValueError('Not Valid Model Name: %s' % FLAGS.model_name)
  policies = [109.20106953, 0.68744719, 3.5974516, 0.34782689, 6.69234296,
              137.19266635, 0.28751239, 7.89317224, 0.96544629, 4.98142085,
              74.98771347, 0.99302917, 3.03588378, 0.63758538, 3.09574188,
              9.40884823e+01, 1.35866024e-01, 2.23605996e+00, 8.54421859e-01, 3.39767442e+00,
              5.10162967e+01, 5.51995875e-01, 4.15257666e+00, 1.43381437e-02, 3.70342795e+00,
              2.37403561e+00, 9.46327006e-01, 8.98322018e+00, 5.74013210e-01, 1.93171616e+00,
              1.84314328e+02, 3.05114167e-01, 6.19262659e+00, 6.21558722e-01, 8.88480359e-03,
              5.37311256e+01, 4.52710724e-01, 4.32227023e+00, 9.64577141e-01, 4.67866033e+00,
              1.24243716e+01, 8.46156673e-01, 2.66230654e+00, 5.05679308e-01, 2.53413119e+00,
              1.55676984e+02, 5.95669582e-01, 8.16849530e+00, 7.34205692e-01, 6.58333647e+00,
              1.25925553e+02, 8.95386018e-02, 7.03711050e-01, 5.10269755e-01, 2.39371388e+00,
              1.73103566e+02, 1.96604196e-01, 7.38150520e+00, 3.62209250e-01, 2.98331999e+00,
              16.13123165, 0.53446554, 5.04162728, 0.99780608, 7.44855232,
              174.41834826, 0.18125575, 8.12656322, 0.99633167, 8.75760733,
              93.08033121, 0.59277896, 8.1990227, 0.96245571, 5.89129844,
              1.35014605e+02, 5.03736991e-01, 3.24525604e+00, 9.85316144e-01, 4.19673992e+00,
              4.57709886e+01, 7.71666943e-02, 4.50353200e+00, 8.80749965e-01, 2.26248302e-01,
              2.86799252e+01, 2.53072454e-01, 4.37616276e+00, 3.43747964e-01, 1.70618167e+00,
              165.40196028, 0.71150485, 4.38419689, 0.18726134, 7.09883013,
              187.1155305, 0.77083213, 8.40102888, 0.236383, 1.35294306,
              81.05222221, 0.7149217, 5.33828718, 0.78634629, 6.47043838,
              1.57754846e+02, 2.12182984e-01, 6.92002239e+00, 8.09418927e-03, 6.52557457e+00,
              1.84751368e+02, 1.36123941e-01, 6.05405157e+00, 9.48045261e-01, 3.63816330e+00,
              3.72569965e+01, 1.69496072e-01, 5.57341552e+00, 5.92098224e-01, 1.20581358e+00]
  bo_policies.construct_good_policies(policies)
  cifar_trainer = CifarModelTrainer(hparams)
  cifar_trainer.run_model()

if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run()
