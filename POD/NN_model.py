import tensorflow as tf
import math
import numpy as np

input_units = 14600
hidden1_units = 256
hidden2_units = 64
hidden3_units = 16
class_num = 2


def inference(input_x):

    # Hidden 1
    with tf.name_scope('hidden1'):

        weights = tf.Variable(tf.truncated_normal([input_units, hidden1_units], stddev=1.0 / math.sqrt(float(input_units))), name='weights')

        biases = tf.Variable(tf.zeros([hidden1_units]), name='biases')

        hidden1 = tf.nn.relu(tf.matmul(input_x, weights) + biases)

    # Hidden 2
    with tf.name_scope('hidden2'):

        weights = tf.Variable(tf.truncated_normal([hidden1_units, hidden2_units], stddev=1.0 / math.sqrt(float(hidden1_units))), name='weights')

        biases = tf.Variable(tf.zeros([hidden2_units]), name='biases')

        hidden2 = tf.nn.relu(tf.matmul(hidden1, weights) + biases)

    # Hidden 3
    with tf.name_scope('hidden3'):

        weights = tf.Variable(tf.truncated_normal([hidden2_units, hidden3_units], stddev=1.0 / math.sqrt(float(hidden2_units))), name='weights')

        biases = tf.Variable(tf.zeros([hidden3_units]), name='biases')

        hidden3 = tf.nn.relu(tf.matmul(hidden2, weights) + biases)

    # Linear
    with tf.name_scope('softmax_linear'):

        weights = tf.Variable(tf.truncated_normal([hidden3_units, class_num], stddev=1.0 / math.sqrt(float(hidden3_units))), name='weights')

        biases = tf.Variable(tf.zeros([class_num]), name='biases')

        logits = tf.matmul(hidden3, weights) + biases

    return logits


def loss(logits, labels):

    # softmax层
    softmax = tf.nn.softmax(logits)

    # 定义各个类别的损失权重
    loss_weight = tf.constant([[1.0], [3.0]])

    # 计算交叉熵
    cross_entropy_vector = tf.matmul(tf.cast(labels, tf.float32) * tf.log(softmax), loss_weight)
    cross_entropy = -tf.reduce_sum(cross_entropy_vector, axis=1)
    # 平均交叉熵
    cross_entropy_mean = tf.reduce_mean(cross_entropy)

    return cross_entropy_mean


def training(loss, learning_rate):
    """Sets up the training Ops.
    Creates a summarizer to track the loss over time in TensorBoard.
    Creates an optimizer and applies the gradients to all trainable variables.
    The Op returned by this function is what must be passed to the
    `sess.run()` call to cause the model to train.
    Args:
      loss: Loss tensor, from loss().
      learning_rate: The learning rate to use for gradient descent.
    Returns:
      train_op: The Op for training.
    """

    # Add a scalar summary for the snapshot loss.
    tf.summary.scalar('loss', loss)

    # Create the gradient descent optimizer with the given learning rate.
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)

    # Create a variable to track the global step.
    global_step = tf.Variable(0, name='global_step', trainable=False)

    # Use the optimizer to apply the gradients that minimize the loss
    # (and also increment the global step counter) as a single training step.
    train_op = optimizer.minimize(loss, global_step=global_step)

    return train_op


def evaluation(logits, labels):
    """Evaluate the quality of the logits at predicting the label.
    Args:
      logits: Logits tensor, float - [batch_size, NUM_CLASSES].
      labels: Labels tensor, int32 - [batch_size], with values in the
        range [0, NUM_CLASSES).
    Returns:
      A scalar int32 tensor with the number of examples (out of batch_size)
      that were predicted correctly.
    """
    # For a classifier model, we can use the in_top_k Op.
    # It returns a bool tensor with shape [batch_size] that is true for
    # the examples where the label is in the top k (here k=1)
    # of all logits for that example.
    y_predict = tf.argmax(logits, axis=1)
    y_label = tf.argmax(labels, axis=1)

    # Return the number of true entries.
    return y_label, y_predict
