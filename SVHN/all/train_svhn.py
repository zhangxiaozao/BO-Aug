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
os.environ['CUDA_VISIBLE_DEVICES']='0'
import time

import custom_ops as ops
import data_utils
import helper_utils
import numpy as np
import tensorflow as tf
from wrn import build_wrn_model
from shake_drop import build_shake_drop_model
from shake_shake import build_shake_shake_model
import bo_policies

tf.flags.DEFINE_string('model_name', 'shake_shake_96',
                       'wrn, shake_shake_32, shake_shake_96, shake_shake_112, '
                       'pyramid_net')
tf.flags.DEFINE_string('checkpoint_dir', '../tmp', 'Training Directory.')
tf.flags.DEFINE_string('data_path', '../dataset',
                       'Directory where dataset is located.')
tf.flags.DEFINE_string('dataset', 'SVHN',
                       'Dataset to train with.')
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


class SVHNModel(object):
  """Builds an image model for SVHN."""

  def __init__(self, hparams):
    self.hparams = hparams

  def build(self, mode):
    """Construct the SVHN model."""
    assert mode in ['train', 'eval']
    self.mode = mode
    self._setup_misc(mode)
    self._setup_images_and_labels()
    self._build_graph(self.images, self.labels, mode)

    self.init = tf.group(tf.global_variables_initializer(),
                         tf.local_variables_initializer())

  def _setup_misc(self, mode):
    """Sets up miscellaneous in the SVHN model constructor."""
    self.lr_rate_ph = tf.Variable(0.0, name='lrn_rate', trainable=False)
    self.reuse = None if (mode == 'train') else True
    self.batch_size = self.hparams.batch_size
    if mode == 'eval':
      self.batch_size = 16

  def _setup_images_and_labels(self):
    """Sets up image and label placeholders for the SVHN model."""
    self.num_classes = 10
    self.images = tf.placeholder(tf.float32, [self.batch_size, 32, 32, 3])
    self.labels = tf.placeholder(tf.float32,
                                 [self.batch_size, self.num_classes])

  def assign_epoch(self, session, epoch_value):
    session.run(self._epoch_update, feed_dict={self._new_epoch: epoch_value})

  def _build_graph(self, images, labels, mode):
    """Constructs the TF graph for the SVHN model.

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
    """Builds the train op for the SVHN model."""
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


class SVHNModelTrainer(object):
  """Trains an instance of the SVHNModel class."""

  def __init__(self, hparams):
    self._session = None
    self.hparams = hparams

    self.model_dir = os.path.join(FLAGS.checkpoint_dir, 'model')
    self.log_dir = os.path.join(FLAGS.checkpoint_dir, 'log')
    # Set the random seed to be sure the same validation set
    # is used for each model
    np.random.seed(0)
    self.data_loader = data_utils.DataSet(hparams)
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
      m = SVHNModel(self.hparams)
      m.build('train')
      self._num_trainable_params = m.num_trainable_params
      self._saver = m.saver
    with tf.variable_scope('model', reuse=True, use_resource=False):
      meval = SVHNModel(self.hparams)
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
    """Trains the SVHN model `m` for one epoch."""
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
  if FLAGS.dataset not in ['SVHN']:
    raise ValueError('Invalid dataset: %s' % FLAGS.dataset)
  hparams = tf.contrib.training.HParams(
      train_size=604388,
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
    hparams.add_hparam('lr', 0.01)
    hparams.add_hparam('weight_decay_rate', 3e-2)
  elif FLAGS.model_name == 'shake_shake_32':
    hparams.add_hparam('model_name', 'shake_shake')
    hparams.add_hparam('num_epochs', 1800)
    hparams.add_hparam('shake_shake_widen_factor', 2)
    hparams.add_hparam('lr', 0.01)
    hparams.add_hparam('weight_decay_rate', 0.001)
  elif FLAGS.model_name == 'shake_shake_96':
    hparams.add_hparam('model_name', 'shake_shake')
    hparams.add_hparam('num_epochs', 160)
    hparams.add_hparam('shake_shake_widen_factor', 6)
    hparams.add_hparam('lr', 0.02)
    hparams.add_hparam('weight_decay_rate', 0.00005)
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
  policies = [1.45522131e+02, 8.45678333e-01, 4.60949614e+00, 3.67976894e-01, 1.83036823e-01,
              1.80641248e+02, 6.80922088e-01, 8.99819506e+00, 8.37348975e-01, 4.55625847e+00,
              7.02834630e+01, 3.31686134e-01, 7.42863238e+00, 2.87297799e-02, 3.39666749e+00,
              97.999995,      0.5,            1.5,            0.83333333,     4.5,
              163.333325,     0.16666667,     7.5,            0.5,            4.5,
              97.999995,      0.5,            4.5,            0.5,            1.5,
              9.83142099e+01, 2.60399343e-02, 3.52886524e+00, 9.88750529e-01, 1.54048298e+00,
              6.75492734e+01, 8.90070866e-01, 1.77199734e+00, 3.41633169e-01, 2.21034172e+00,
              7.08153528e+01, 9.61761820e-02, 3.36569549e+00, 1.00264871e-01, 3.26389001e+00,
              1.83076901e+02, 8.56486275e-01, 6.25760726e+00, 6.59635963e-01, 3.78382978e-01,
              1.82390970e+02, 4.33487639e-01, 2.64658982e+00, 9.04660680e-01, 2.42482708e+00,
              1.61532705e+02, 1.29753357e-02, 6.78490504e+00, 1.85556674e-01, 1.28215456e+00,
              154.48458228,   0.80396899,     9.,             0.72748873,     6.50008488,
              114.86504097,   0.76100067,     1.60808054,     0.54140569,     7.35203369,
              68.75617049,    0.73635751,     4.08114415,     0.91400328,     5.93128857,
              171.72478549,   1.,             1.59970262,     1.,             2.51771727,
              11.09116256,    0.18575471,     2.08929737,     1.,             9.,
              65.21873297,    1.,             5.30298646,     0.77797216,     2.12326594,
              23.9726703,     0.56901906,     6.86111495,     0.7820346,      2.29774103,
              106.46584923,   0.30513824,     5.81118573,     0.9950806,      1.6216652,
              29.08702704,    0.59834883,     6.89900005,     0.65553929,     1.0488647,
              9.92287105e+01, 2.12627562e-01, 1.49527328e+00, 6.46882043e-01, 5.81398546e+00,
              1.42053160e+02, 8.83175903e-01, 1.46498517e+00, 6.81907420e-03, 5.50908918e+00,
              1.47931282e+02, 9.48393530e-01, 8.66660511e+00, 3.97058084e-02, 3.69045357e+00]
  bo_policies.construct_good_policies(policies)
  SVHN_trainer = SVHNModelTrainer(hparams)
  SVHN_trainer.run_model()

if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run()
