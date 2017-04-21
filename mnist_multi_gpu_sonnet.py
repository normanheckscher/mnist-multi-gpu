# Copyright 2017 Norman Heckscher. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""A binary to train MNIST using multiple GPU's with synchronous updates.

Accuracy:

Done training with 2 GPUs, for 20 epochs, 11000 steps.
Total Duration (327.396 sec)
2017-04-21 20:46:18.466392: precision = 9848.000
Done training with 1 GPUs, for 20 epochs, 22000 steps.
Total Duration (500.122 sec)
2017-04-21 20:56:40.639580: precision = 9884.000

Speed: With batch_size 50.

System        | Step Time (sec/batch)  | Accuracy
-------------------------------------------------------------------------
1 GTX 1080    | 258.136 sec            | ~94.58% at 11K steps
2 GTX 1080    | 189.572 sec            | ~94.59% at 11K steps

Usage:
Please see the TensorFlow website for how to download the MNIST
data set, compile and train models.

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path
import re
import time
import numpy as np
from datetime import datetime

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf
import sonnet as snt

# Constants used for dealing with the files, matches convert_to_records.
TRAIN_FILE = 'train.tfrecords'
VALIDATION_FILE = 'validation.tfrecords'
# If a model is trained with multiple GPUs, prefix all Op names with tower_name
# to differentiate the operations. Note that this prefix is removed from the
# names of the summaries when visualizing a model.
TOWER_NAME = 'tower'
IMAGE_PIXELS = 784

# Constants describing the training process.
MOVING_AVERAGE_DECAY = 0.9999  # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 20.0  # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.
INITIAL_LEARNING_RATE = 0.1  # Initial learning rate.

# Global constants describing the MNIST data set.
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 50000
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 10000

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('batch_size', 50,
                            """Number of images to process in a batch.""")
tf.app.flags.DEFINE_string('data_dir', '/home/norman/MNIST_data',
                           """Path to the MNIST data directory.""")
tf.app.flags.DEFINE_string('train_dir', '/home/norman/MNIST_train',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_integer('num_gpus', 2,
                            """How many GPUs to use.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")
tf.app.flags.DEFINE_boolean('tb_logging', False,
                            """Whether to log to Tensorboard.""")
tf.app.flags.DEFINE_integer('num_epochs', 20,
                            """Number of epochs to run trainer.""")


def read_and_decode(filename_queue):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        # Defaults are not specified since both keys are required.
        features={
            'image_raw': tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([], tf.int64),
        })

    # Convert from a scalar string tensor (whose single string has
    # length mnist.IMAGE_PIXELS) to a uint8 tensor with shape
    # [mnist.IMAGE_PIXELS].
    image = tf.decode_raw(features['image_raw'], tf.uint8)
    image.set_shape([IMAGE_PIXELS])

    # OPTIONAL: Could reshape into a 28x28 image and apply distortions
    # here.  Since we are not applying any distortions in this
    # example, and the next step expects the image to be flattened
    # into a vector, we don't bother.

    # Convert from [0, 255] -> [-0.5, 0.5] floats.
    image = tf.cast(image, tf.float32) * (1. / 255) - 0.5

    # Convert label from a scalar uint8 tensor to an int32 scalar.
    label = tf.cast(features['label'], tf.int32)

    return image, label


def inputs(train, batch_size, num_epochs):
    """Reads input data num_epochs times.
    Args:
      train: Selects between the training (True) and validation (False) data.
      batch_size: Number of examples per returned batch.
      num_epochs: Number of times to read the input data, or 0/None to
         train forever.
    Returns:
      A tuple (images, labels), where:
      * images is a float tensor with shape [batch_size, mnist.IMAGE_PIXELS]
        in the range [-0.5, 0.5].
      * labels is an int32 tensor with shape [batch_size] with the true label,
        a number in the range [0, mnist.NUM_CLASSES).
      Note that an tf.train.QueueRunner is added to the graph, which
      must be run using e.g. tf.train.start_queue_runners().
    """
    if not num_epochs: num_epochs = None
    filename = os.path.join(FLAGS.data_dir,
                            TRAIN_FILE if train else VALIDATION_FILE)

    with tf.name_scope('input'):
        filename_queue = tf.train.string_input_producer(
            [filename], num_epochs=num_epochs)

        # Even when reading in multiple threads, share the filename
        # queue.
        image, label = read_and_decode(filename_queue)

        # Shuffle the examples and collect them into batch_size batches.
        # (Internally uses a RandomShuffleQueue.)
        # We run this in two threads to avoid being a bottleneck.
        images, sparse_labels = tf.train.shuffle_batch(
            [image, label], batch_size=batch_size, num_threads=2,
            capacity=1000 + 3 * batch_size,
            # Ensures a minimum amount of shuffling of examples.
            min_after_dequeue=1000)

        return images, sparse_labels


def custom_build(inputs, is_training, keep_prob):
  x_inputs = tf.reshape(inputs, [-1, 28, 28, 1])
  """A custom build method to wrap into a sonnet Module."""
  outputs = snt.Conv2D(output_channels=32, kernel_shape=4, stride=2)(x_inputs)
  outputs = snt.BatchNorm()(outputs, is_training=is_training)
  outputs = tf.nn.relu(outputs)
  outputs = tf.nn.max_pool(outputs, ksize=[1, 2, 2, 1],
                           strides=[1, 2, 2, 1], padding='SAME')
  outputs = snt.Conv2D(output_channels=64, kernel_shape=4, stride=2)(outputs)
  outputs = snt.BatchNorm()(outputs, is_training=is_training)
  outputs = tf.nn.relu(outputs)
  outputs = tf.nn.max_pool(outputs, ksize=[1, 2, 2, 1],
                           strides=[1, 2, 2, 1], padding='SAME')
  outputs = snt.Conv2D(output_channels=1024, kernel_shape=1, stride=1)(outputs)
  outputs = snt.BatchNorm()(outputs, is_training=is_training)
  outputs = tf.nn.relu(outputs)
  outputs = snt.BatchFlatten()(outputs)
  outputs = tf.nn.dropout(outputs, keep_prob=keep_prob)
  outputs = snt.Linear(output_size=10)(outputs)
#  _activation_summary(outputs)
  return outputs


def _variable_with_weight_decay(name, shape, stddev, wd):
    """Helper to create an initialized Variable with weight decay.

    Note that the Variable is initialized with a truncated normal distribution.
    A weight decay is added only if one is specified.

    Args:
      name: name of the variable
      shape: list of ints
      stddev: standard deviation of a truncated Gaussian
      wd: add L2Loss weight decay multiplied by this float. If None, weight
          decay is not added for this Variable.

    Returns:
      Variable Tensor
    """
    dtype = tf.float32
    var = _variable_on_cpu(
        name,
        shape,
        tf.truncated_normal_initializer(stddev=stddev, dtype=dtype))
    if wd is not None:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var


def _variable_on_cpu(name, shape, initializer):
    """Helper to create a Variable stored on CPU memory.

    Args:
      name: name of the variable
      shape: list of ints
      initializer: initializer for Variable

    Returns:
      Variable Tensor
    """
    with tf.device('/cpu:0'):
        dtype = tf.float32
        var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
    return var


def _activation_summary(x):
    """Helper to create summaries for activations.

    Creates a summary that provides a histogram of activations.
    Creates a summary that measures the sparsity of activations.

    Args:
      x: Tensor
    Returns:
      nothing
    """
    # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
    # session. This helps the clarity of presentation on tensorboard.
    if FLAGS.tb_logging:
        tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
        tf.summary.histogram(tensor_name + '/activations', x)
        tf.summary.scalar(tensor_name + '/sparsity',
                          tf.nn.zero_fraction(x))


def loss(logits, labels):
    """Add L2Loss to all the trainable variables.

    Add summary for "Loss" and "Loss/avg".
    Args:
      logits: Logits from inference().
      labels: Labels from distorted_inputs or inputs(). 1-D tensor
              of shape [batch_size]

    Returns:
      Loss tensor of type float.
    """
    # Calculate the average cross entropy loss across the batch.
    # labels = tf.cast(labels, tf.int64)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=labels, logits=logits, name='cross_entropy_per_example')
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    tf.add_to_collection('losses', cross_entropy_mean)

    # The total loss is defined as the cross entropy loss plus all of the weight
    # decay terms (L2 loss).
    return tf.add_n(tf.get_collection('losses'), name='total_loss')


def average_gradients(tower_grads):
    """Calculate average gradient for each shared variable across all towers.

    Note that this function provides a synchronization point across all towers.

    Args:
      tower_grads: List of lists of (gradient, variable) tuples. The outer list
        is over individual gradients. The inner list is over the gradient
        calculation for each tower.
    Returns:
       List of pairs of (gradient, variable) where the gradient has been 
       averaged across all towers.
    """
    # for m in xrange(len(tower_grads)):
    #     for n in xrange(len(tower_grads[m])):
    #         print(type(tower_grads[0][n][0]))
    # for gg in tower_grads:
    #     for x in gg:
    #         print(type(x[0]))
    #     print(tower_grads)

    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
            if g != None:
            # Add 0 dimension to the gradients to represent the tower.
                expanded_g = tf.expand_dims(g, 0)

            # Append on a 'tower' dimension which we will average over below.
                grads.append(expanded_g)

        # Average over the 'tower' dimension.
        grad = tf.concat(grads, 0)
        grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads


def tower_loss(scope):
    """Calculate the total loss on a single tower running the MNIST model.

    Args:
      scope: unique prefix string identifying the MNIST tower, e.g. 'tower_0'

    Returns:
       Tensor of shape [] containing the total loss for a batch of data
    """
    # Input images and labels.

    images, labels = inputs(train=True, batch_size=FLAGS.batch_size,
                            num_epochs=(FLAGS.num_epochs / FLAGS.num_gpus))
    # Build inference Graph.
    # The line below takes custom_build and
    # wraps it to construct a sonnet Module.
    module_with_build_args = snt.Module(custom_build, name='simple_net')

    train_model_outputs = module_with_build_args(images, is_training=True,
                                                 keep_prob=tf.constant(0.5))

    # Build the portion of the Graph calculating the losses. Note that we will
    # assemble the total_loss using a custom function below.
    _ = loss(train_model_outputs, labels)

    # Assemble all of the losses for the current tower only.
    losses = tf.get_collection('losses', scope)

    # Calculate the total loss for the current tower.
    total_loss = tf.add_n(losses, name='total_loss')

    # Attach a scalar summary to all individual losses and the total loss; do
    # the same for the averaged version of the losses.
    if FLAGS.tb_logging:
        for l in losses + [total_loss]:
            # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU
            # training session. This helps the clarity of presentation on
            # tensorboard.
            loss_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', l.op.name)
            tf.summary.scalar(loss_name, l)

    return total_loss


def train():
    with tf.Graph().as_default(), tf.device('/cpu:0'):
        # Create a variable to count the number of train() calls. This equals
        # the number of batches processed * FLAGS.num_gpus.
        global_step = tf.get_variable(
            'global_step', [],
            initializer=tf.constant_initializer(0), trainable=False)

        # Calculate the learning rate schedule.
        num_batches_per_epoch = (NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN /
                                 (FLAGS.batch_size * FLAGS.num_gpus))
        decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)

        # Decay the learning rate exponentially based on the number of steps.
        lr = tf.train.exponential_decay(learning_rate=INITIAL_LEARNING_RATE,
                                        global_step=global_step,
                                        decay_steps=decay_steps,
                                        decay_rate=LEARNING_RATE_DECAY_FACTOR,
                                        staircase=True)

        opt = tf.train.MomentumOptimizer(lr, 0.9, use_nesterov=True,
                                        use_locking=True)
        #opt = tf.train.AdamOptimizer(1e-4)
        # opt = tf.train.ProximalAdagradOptimizer(1e-2)

        # Calculate the gradients for each model tower.
        tower_grads = []
        with tf.variable_scope(tf.get_variable_scope()):
            for i in xrange(FLAGS.num_gpus):
                with tf.device('/gpu:%d' % i):
                    with tf.name_scope(
                                    '%s_%d' % (TOWER_NAME, i)) as scope:
                        # Calculate the loss for one tower of the CIFAR model.
                        # This function constructs the entire CIFAR model but
                        # shares the variables across all towers.
                        loss = tower_loss(scope)

                        # Reuse variables for the next tower.
                        # No need for this with Sonnet?
                        #tf.get_variable_scope().reuse_variables()

                        # Retain the summaries from the final tower.
                        summaries = tf.get_collection(tf.GraphKeys.SUMMARIES,
                                                      scope)

                        # Calculate the gradients for the batch of data on this
                        # MNIST tower.
                        grads = opt.compute_gradients(loss, gate_gradients=2)
                        # for x in grads:
                        #     print(type(x[0]))
                        # print (grads)

                        # Keep track of the gradients across all towers.
                        tower_grads.append(grads)

        # We must calculate the mean of each gradient. Note that this is the
        # synchronization point across all towers.
        grads = average_gradients(tower_grads)

        # Add histograms for gradients.
        if FLAGS.tb_logging:
            for grad, var in grads:
                if grad is not None:
                    summaries.append(
                        tf.summary.histogram(var.op.name + '/gradients', grad))
            # Add a summary to track the learning rate.
            summaries.append(tf.summary.scalar('learning_rate', lr))

        train_op = opt.apply_gradients(grads, global_step=global_step)

        # Add histograms for trainable variables.
        if FLAGS.tb_logging:
            for var in tf.trainable_variables():
                summaries.append(tf.summary.histogram(var.op.name, var))

        # Create a saver.
        saver = tf.train.Saver(tf.global_variables(), sharded=True)

        # Build the summary operation from the last tower summaries.
        summary_op = tf.summary.merge(summaries)

        # Build an initialization operation to run below.
        # init = tf.global_variables_initializer()

        # The op for initializing the variables.
        init_op = tf.group(tf.global_variables_initializer(),
                           tf.local_variables_initializer())

        # Start running operations on the Graph. allow_soft_placement must be
        # set to True to build towers on GPU, as some of the ops do not have GPU
        # implementations.
        gpu_options = tf.GPUOptions(allow_growth=True)
        sess = tf.Session(config=tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=FLAGS.log_device_placement,
            gpu_options=gpu_options))
        sess.run(init_op)

        # Start input enqueue threads.
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        summary_writer = tf.summary.FileWriter(FLAGS.train_dir, sess.graph)

        try:
            step = 0
            while not coord.should_stop():
                start_time = time.time()

                # Run one step of the model.  The return values are
                # the activations from the `train_op` (which is
                # discarded) and the `loss` op.  To inspect the values
                # of your ops or variables, you may include them in
                # the list passed to sess.run() and the value tensors
                # will be returned in the tuple from the call.
                _, loss_value = sess.run([train_op, loss])

                duration = time.time() - start_time

                assert not np.isnan(
                    loss_value), 'Model diverged with loss = NaN'

                # Print an overview fairly often.
                if step % 100 == 0:
                    num_examples_per_step = FLAGS.batch_size * FLAGS.num_gpus
                    examples_per_sec = num_examples_per_step / duration
                    sec_per_batch = duration / FLAGS.num_gpus
                    format_str = (
                        '%s: step %d, epochs %d, loss = %.3f '
                        '(%.1f examples/sec; %.3f sec/batch)')
                    print(format_str % (datetime.now(), step,
                                        # step * gpu / batchsize * gpu = 100
                                        FLAGS.num_epochs,
                                        loss_value,
                                        examples_per_sec, sec_per_batch))
                if FLAGS.tb_logging:
                    if step % 10 == 0:
                        summary_str = sess.run(summary_op)
                        summary_writer.add_summary(summary_str, step)

                # Save the model checkpoint periodically.
                if step % 1000 == 0 or (
                            step + 1) == FLAGS.num_epochs * FLAGS.batch_size:
                    checkpoint_path = os.path.join(FLAGS.train_dir,
                                                   'model.ckpt')
                    saver.save(sess, checkpoint_path, global_step=step)

                step += 1
        except tf.errors.OutOfRangeError:
            print('Done training with %d GPUs, for %d epochs, %d steps.' % (
                FLAGS.num_gpus, FLAGS.num_epochs, step))
        finally:
            # When done, ask the threads to stop.
            coord.request_stop()

        # Wait for threads to finish.
        coord.join(threads)
        sess.close()


def evaluate():
    """Eval MNIST for a number of steps."""
    with tf.Graph().as_default():
        # Get images and labels for MNIST.
        mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=False)
        images = mnist.test.images
        labels = mnist.test.labels

        # Build a Graph that computes the logits predictions from the
        # inference model.
        # The line below takes custom_build and wraps it to construct a sonnet Module.
        module_with_build_args = snt.Module(custom_build, name='simple_net')
        test_model_outputs = module_with_build_args(images, is_training=False,
                                                    keep_prob=tf.constant(1.0))

        # Calculate predictions.
        top_k_op = tf.nn.in_top_k(predictions=test_model_outputs, targets=labels, k=1)

        # Create saver to restore the learned variables for eval.
        saver = tf.train.Saver()

        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
            if ckpt and ckpt.model_checkpoint_path:
                # Restores from checkpoint
                saver.restore(sess, ckpt.model_checkpoint_path)
            else:
                print('No checkpoint file found')
                return

            predictions = np.sum(sess.run([top_k_op]))

            # Compute precision.
            print('%s: precision = %.3f' % (datetime.now(), predictions))


def main(argv=None):  # pylint: disable=unused-argument
    start_time = time.time()
    train()
    duration = time.time() - start_time
    print('Total Duration (%.3f sec)' % duration)
    evaluate()


if __name__ == '__main__':
    tf.app.run()
