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
mnist_multi_gpu_train.py achieves ~xx% accuracy after 20K steps (xxx
epochs of data) as judged by mnist_multi_gpu_batching_eval.py.

Speed: With batch_size 50.

System        | Step Time (sec/batch)  |     Accuracy
--------------------------------------------------------------------
1 GTX 1080    | 0.08-0.10              | ~xx% at 20K steps  (x hours)
2 GTX 1080    | 0.08-0.10              | ~xx% at 20K steps  (x hours)

Usage:
Please see the tutorial and website for how to download the MNIST
data set, compile the program and train the model.

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import os.path
import re
import time

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

import model



FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('batch_size', 1000,
                            """Number of images to process in a batch.""")
tf.app.flags.DEFINE_string('data_dir', '/home/norman/MNIST_data',
                           """Path to the MNIST data directory.""")
tf.app.flags.DEFINE_boolean('use_fp16', False,
                            """Train the model using fp16.""")
tf.app.flags.DEFINE_string('train_dir', '/home/norman/MNIST_train',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_integer('max_steps', 20000,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_integer('num_gpus', 2,
                            """How many GPUs to use.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")
tf.app.flags.DEFINE_boolean('tb_logging', False,
                            """Whether to log to Tensorboard.""")

def tower_loss(scope):
    """Calculate the total loss on a single tower running the MNIST model.
  
    Args:
      scope: unique prefix string identifying the MNIST tower, e.g. 'tower_0'
  
    Returns:
       Tensor of shape [] containing the total loss for a batch of data
    """
    # Get images and labels for MSNIT.
    images, labels = model.inputs(FLAGS.batch_size)

    # Build inference Graph.
    logits = model.inference(images, keep_prob=0.5)

    # Build the portion of the Graph calculating the losses. Note that we will
    # assemble the total_loss using a custom function below.
    _ = model.loss(logits, labels)

    # Assemble all of the losses for the current tower only.
    losses = tf.get_collection('losses', scope)

    # Calculate the total loss for the current tower.
    total_loss = tf.add_n(losses, name='total_loss')

    # Attach a scalar summary to all individual losses and the total loss; do
    # the same for the averaged version of the losses.
    if (FLAGS.tb_logging):
        for l in losses + [total_loss]:
            # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU
            # training session. This helps the clarity of presentation on
            # tensorboard.
            loss_name = re.sub('%s_[0-9]*/' % model.TOWER_NAME, '', l.op.name)
            tf.summary.scalar(loss_name, l)

    return total_loss


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
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
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

def train():
    """Train MNIST for a number of steps."""
    with tf.Graph().as_default(), tf.device('/cpu:0'):

        # Create a variable to count the number of train() calls. This equals
        # the number of batches processed * FLAGS.num_gpus.
        global_step = tf.get_variable(
            'global_step', [],
            initializer=tf.constant_initializer(0), trainable=False)

        # Use AdamOptimizer.
        opt = tf.train.AdamOptimizer(model.INITIAL_LEARNING_RATE)

        # Calculate the gradients for each model tower.
        tower_grads = []
        with tf.variable_scope(tf.get_variable_scope()):
            for i in xrange(FLAGS.num_gpus):
                with tf.device('/gpu:%d' % i):
                    with tf.name_scope(
                                    '%s_%d' % (model.TOWER_NAME, i)) as scope:
                        # Calculate the loss for one tower of the MNIST model.
                        # This function constructs the entire MNIST model but
                        # shares the variables across all towers.
                        loss = tower_loss(scope)

                        # Reuse variables for the next tower.
                        tf.get_variable_scope().reuse_variables()

                        # Retain the summaries from the final tower.
                        summaries = tf.get_collection(tf.GraphKeys.SUMMARIES,
                                                      scope)

                        # Calculate the gradients for the batch of data on this
                        # MNIST tower.
                        grads = opt.compute_gradients(loss)

                        # Keep track of the gradients across all towers.
                        tower_grads.append(grads)

        # We must calculate the mean of each gradient. Note that this is the
        # synchronization point across all towers.
        grads = average_gradients(tower_grads)

        # Add histograms for gradients.
        if (FLAGS.tb_logging):
            for grad, var in grads:
                if grad is not None:
                    summaries.append(
                        tf.summary.histogram(var.op.name + '/gradients', grad))

        # Apply the gradients to adjust the shared variables.
        train_op = opt.apply_gradients(grads, global_step=global_step)

        # Add histograms for trainable variables.
        if (FLAGS.tb_logging):
            for var in tf.trainable_variables():
                summaries.append(tf.summary.histogram(var.op.name, var))

        # Create a saver.
        saver = tf.train.Saver(tf.global_variables())

        # Build the summary operation from the last tower summaries.
        summary_op = tf.summary.merge(summaries)

        # Build an initialization operation to run below.
        init = tf.global_variables_initializer()

        # Start running operations on the Graph. allow_soft_placement must be
        # set to True to build towers on GPU, as some of the ops do not have GPU
        # implementations.
        sess = tf.Session(config=tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=FLAGS.log_device_placement))
        sess.run(init)

        # Start the queue runners.
        tf.train.start_queue_runners(sess=sess)

        summary_writer = tf.summary.FileWriter(FLAGS.train_dir, sess.graph)

        for step in xrange(FLAGS.max_steps):
            start_time = time.time()
            _, loss_value = sess.run([train_op, loss])
            duration = time.time() - start_time

            assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

            if step % 50 == 0:
                num_examples_per_step = FLAGS.batch_size * FLAGS.num_gpus
                examples_per_sec = num_examples_per_step / duration
                sec_per_batch = duration / FLAGS.num_gpus

                format_str = (
                    '%s: step %d, loss = %.4f (%.1f examples/sec; %.3f '
                    'sec/batch)')
                print(format_str % (datetime.now(), step, loss_value,
                                    examples_per_sec, sec_per_batch))
            if (FLAGS.tb_logging):
                if step % 5 == 0:
                    summary_str = sess.run(summary_op)
                    summary_writer.add_summary(summary_str, step)

            # Save the model checkpoint periodically.
            if step % 1000 == 0 or (step + 1) == FLAGS.max_steps:
                checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)


def main(argv=None):  # pylint: disable=unused-argument
    train()


if __name__ == '__main__':
    tf.app.run()
