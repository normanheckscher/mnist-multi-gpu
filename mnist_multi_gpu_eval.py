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

"""Evaluation for MNIST.

Accuracy:

Speed:

Usage:

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import math
import time

import numpy as np
import tensorflow as tf

import model

from tensorflow.examples.tutorials.mnist import input_data

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('batch_size', 50,
                            """Number of images to process in a batch.""")
tf.app.flags.DEFINE_string('eval_dir', '/home/norman/MNIST_train',
                           """Directory where to write event logs.""")
tf.app.flags.DEFINE_string('eval_data', 'test',
                           """Either 'test' or 'train_eval'.""")
tf.app.flags.DEFINE_string('data_dir', '/home/norman/MNIST_data',
                           """Path to the MNIST data directory.""")
tf.app.flags.DEFINE_string('checkpoint_dir', '/home/norman/MNIST_train',
                           """Directory where to read model checkpoints.""")
tf.app.flags.DEFINE_integer('eval_interval_secs', 5,
                            """How often to run the eval.""")
tf.app.flags.DEFINE_integer('num_examples', 10000,
                            """Number of examples to run.""")
tf.app.flags.DEFINE_boolean('run_once', False,
                            """Whether to run eval only once.""")
tf.app.flags.DEFINE_boolean('use_fp16', False,
                            """Train the model using fp16.""")


def eval_once(saver, summary_writer, top_k_op):
    """Run Eval once.
  
    Args:
      saver: Saver.
      summary_writer: Summary writer.
      top_k_op: Top K op.
    """
    with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            # Restores from checkpoint
            saver.restore(sess, ckpt.model_checkpoint_path)
            # Assuming model_checkpoint_path looks something like:
            #   /my-favorite-path/MNIST_train/model.ckpt-0,
            # extract global_step from it.
            global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[
                -1]
        else:
            print('No checkpoint file found')
            return

        predictions = np.sum(sess.run([top_k_op]))

        # Compute precision @ 1.
        print('%s: precision @ 1 = %.3f' % (datetime.now(), predictions))

def evaluate():
    """Eval MNIST for a number of steps."""
    with tf.Graph().as_default() as g:
        # Get images and labels for MNIST.
        mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=False)
        images = mnist.test.images
        labels = tf.cast(mnist.test.labels, dtype=tf.int32)

        # Build a Graph that computes the logits predictions from the
        # inference model.
        logits = model.inference(images, keep_prob=1.0)

        # Calculate predictions.
        top_k_op = tf.nn.in_top_k(predictions=logits, targets=labels, k=1)

        # Create saver to restore the learned variables for eval.
        saver = tf.train.Saver()

        # Build the summary operation based on the TF collection of Summaries.
        summary_op = tf.summary.merge_all()

        summary_writer = tf.summary.FileWriter(FLAGS.eval_dir, g)

        eval_once(saver, summary_writer, top_k_op)

def main(argv=None):  # pylint: disable=unused-argument
    evaluate()


if __name__ == '__main__':
    tf.app.run()
