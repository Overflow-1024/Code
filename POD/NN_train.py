import tensorflow as tf
import pandas as pd
import numpy as np
import os
import time
import csv
import sys
from six.moves import xrange
import NN_model
import model
import preprocessing
import matplotlib.pyplot as plt

log_dir = "NN_model"
# 设置训练参数
batch_size = 50
learning_rate = 0.001
max_steps = 100000


class DataSet(object):

    def __init__(self, x, y):
        self._x = x
        self._y = y
        self._epochs_completed = 0
        self._index_in_epoch = 0
        self._samples_num = y.shape[0]

        print("dataset shape: ")
        print(self._x.shape)
        print(self._y.shape)

    @property
    def x(self):
        return self._x

    @property
    def y(self):
        return self._y

    @property
    def samples_num(self):
        return self._samples_num

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size, shuffle=True):
        start = self._index_in_epoch

        # 第一轮训练之前洗牌
        if self._epochs_completed == 0 and start == 0 and shuffle:
            perm0 = np.arange(self._samples_num)
            np.random.shuffle(perm0)
            self._x = self.x[perm0]
            self._y = self.y[perm0]

        # Go to the next epoch
        if start + batch_size > self._samples_num:
            # Finished epoch
            self._epochs_completed += 1

            # Get the rest examples in this epoch
            rest_num_examples = self._samples_num - start
            x_rest_part = self._x[start:self._samples_num]
            y_rest_part = self._y[start:self._samples_num]

            # Shuffle the data
            if shuffle:
                perm = np.arange(self._samples_num)
                np.random.shuffle(perm)
                self._x = self.x[perm]
                self._y = self.y[perm]

            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size - rest_num_examples
            end = self._index_in_epoch
            x_new_part = self._x[start:end]
            y_new_part = self._y[start:end]

            return np.concatenate((x_rest_part, x_new_part), axis=0), np.concatenate((y_rest_part, y_new_part), axis=0)

        else:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
            return self._x[start:end], self._y[start:end]


# 划分训练集和测试集，比例3:1
def divide_dataset(data_x, data_y):

    cut_point = int(data_y.shape[0] * 0.75) // batch_size * batch_size
    end = data_y.shape[0] // batch_size * batch_size
    x_train = data_x[:cut_point]
    y_train = data_y[:cut_point]
    x_test = data_x[cut_point: end]
    y_test = data_y[cut_point: end]

    return x_train, y_train, x_test, y_test


def fill_feed_dict(data_set, x_placeholder, y_placeholder, shuffle_flag):
    """Fills the feed_dict for training the given step.
    A feed_dict takes the form of:
     feed_dict = {
        <placeholder>: <tensor of values to be passed for placeholder>,
         ....
    }
    Args:
        data_set: The set of images and labels, from input_data.read_data_sets()
    Returns:
        feed_dict: The feed dictionary mapping from placeholders to values.
    """
    # Create the feed_dict for the placeholders filled with the next
    # `batch size` examples.
    x_feed, y_feed = data_set.next_batch(batch_size, shuffle=shuffle_flag)

    feed_dict = {
        x_placeholder: x_feed,
        y_placeholder: y_feed,
    }
    return feed_dict


def do_eval(sess, y_label, y_predict, x_placeholder, y_placeholder, data_set):
    """Runs one evaluation against the full epoch of data.
    Args:
        sess: The session in which the model has been trained.
        eval_correct: The Tensor that returns the number of correct predictions.
        data_set: The set of images and labels to evaluate, from
        input_data.read_data_sets().
    """
    # And run one epoch of eval.
    true_count = 0  # Counts the number of correct predictions.

    # 计算实际参与计算的样本数量(最后数量不到1个batch的那堆样本就扔掉了）
    steps_per_epoch = data_set.samples_num // batch_size
    num_examples = steps_per_epoch * batch_size

    y_predict_all = np.zeros((steps_per_epoch, batch_size))
    y_label_all = np.zeros((steps_per_epoch, batch_size))

    for step in xrange(steps_per_epoch):
        feed_dict = fill_feed_dict(data_set, x_placeholder, y_placeholder, shuffle_flag=False)

        y_label_batch, y_predict_batch = sess.run([y_label, y_predict], feed_dict=feed_dict)

        y_predict_all[step] = y_predict_batch
        y_label_all[step] = y_label_batch

    accuracy, confusion_matrix, precision, recall, Fscore = model.evaluate(y_predict_all.flatten(), y_label_all.flatten())

    print('Num examples: %d  Num correct: %d  Accuracy : %0.04f' % (num_examples, true_count, accuracy))
    print(confusion_matrix)
    print("Precision: %0.04f  Recall: %0.04f  Fscore: %0.04f" % (precision, recall, Fscore))


def run_training(data_x, data_y):
    """Train MNIST for a number of steps."""
    # Get the sets of images and labels for training, validation, and
    # test on MNIST.

    # 划分原数据集，得到训练集，测试集
    x_train, y_train, x_test, y_test = divide_dataset(data_x, data_y)
    dataset_train = DataSet(x_train, y_train)
    dataset_test = DataSet(x_test, y_test)

    # Tell TensorFlow that the model will be built into the default Graph.
    with tf.Graph().as_default():
        # Generate placeholders for the images and labels.

        x_placeholder = tf.placeholder(tf.float32, shape=(batch_size, NN_model.input_units))

        y_placeholder = tf.placeholder(tf.int32, shape=(batch_size, 2))

        # Build a Graph that computes predictions from the inference model.
        logits = NN_model.inference(x_placeholder)

        # Add to the Graph the Ops for loss calculation.
        loss = NN_model.loss(logits, y_placeholder)

        # Add to the Graph the Ops that calculate and apply gradients.
        train_op = NN_model.training(loss, learning_rate)

        # Add the Op to compare the logits to the labels during evaluation.
        y_label, y_predict = NN_model.evaluation(logits, y_placeholder)

        # Build the summary Tensor based on the TF collection of Summaries.
        summary = tf.summary.merge_all()

        # Add the variable initializer Op.
        init = tf.global_variables_initializer()

        # Create a saver for writing training checkpoints.
        saver = tf.train.Saver()

        # Create a session for running Ops on the Graph.
        sess = tf.Session()

        # Instantiate a SummaryWriter to output summaries and the Graph.
        summary_writer = tf.summary.FileWriter(log_dir, sess.graph)

        # And then after everything is built:

        # Run the Op to initialize the variables.
        sess.run(init)

        step_list = []
        loss_list = []

        # Start the training loop.
        for step in xrange(max_steps):
            start_time = time.time()

        # Fill a feed dictionary with the actual set of images and labels
        # for this particular training step.
            feed_dict = fill_feed_dict(dataset_train, x_placeholder, y_placeholder, shuffle_flag=True)

        # Run one step of the model.  The return values are the activations
        # from the `train_op` (which is discarded) and the `loss` Op.  To
        # inspect the values of your Ops or variables, you may include them
        # in the list passed to sess.run() and the value tensors will be
        # returned in the tuple from the call.
            _, loss_value, _logits = sess.run([train_op, loss, logits], feed_dict=feed_dict)

            duration = time.time() - start_time

        # Write the summaries and print an overview fairly often.
            if step % 100 == 0:
                # Print status to stdout.
                print('Step %d: loss = %.3f (%.3f sec)' % (step, loss_value, duration))
                step_list.append(step)
                loss_list.append(loss_value)
                # Update the events file.
                summary_str = sess.run(summary, feed_dict=feed_dict)
                summary_writer.add_summary(summary_str, step)
                summary_writer.flush()

            # Save a checkpoint and evaluate the model periodically.
            if (step + 1) % 1000 == 0 or (step + 1) == max_steps:

                checkpoint_file = os.path.join(log_dir, 'model.ckpt')

                saver.save(sess, checkpoint_file, global_step=step)

                # Evaluate against the training set.
                print('Training Data Eval:')
                do_eval(sess, y_label, y_predict, x_placeholder, y_placeholder, dataset_train)

                # Evaluate against the test set.
                print('Test Data Eval:')
                do_eval(sess, y_label, y_predict, x_placeholder, y_placeholder, dataset_test)

            # 如果损失已小于目标值，提前结束训练
            if loss_value < 0.001:
                break

        # 画loss的变化图
        plt.plot(step_list, loss_list, color='blue', linestyle="-", marker=".", linewidth=1)

        plt.xlabel("step")
        plt.ylabel("loss")

        plt.show()



if __name__ == '__main__':

    data_path = "D:/毕设数据/第一层/ISall_NN.csv"

    maxInt = sys.maxsize
    while True:
        # decrease the maxInt value by factor 10
        # as long as the OverflowError occurs.

        try:
            csv.field_size_limit(maxInt)
            break
        except OverflowError:
            maxInt = int(maxInt / 10)

    # 读取数据文件
    df = pd.read_csv(data_path, engine='python', encoding='gb2312')
    df_key = df.loc[:, ['name', 'pod']]
    df_attr = df.drop(['name', 'pod'], axis=1, inplace=False)
    data_X = df_attr.values
    data_y = df['pod'].values
    attr_name = df_attr.columns.values.tolist()

    print("start training...")

    X_resample, y_resample = preprocessing.over_sampling_smote(data_X, data_y)
    X_resample, y_resample = preprocessing.shuffle(X_resample, y_resample)

    y_resample_onehot = np.zeros((y_resample.shape[0], 2))
    for row in range(y_resample.shape[0]):
        if y_resample[row] == 1:
            y_resample_onehot[row] = [0, 1]
        else:
            y_resample_onehot[row] = [1, 0]

    run_training(X_resample, y_resample_onehot)
