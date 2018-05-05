import os

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sn

# 加载数据集
from tensorflow.python.framework import graph_util


def read_data(file_path):
    column_names = ['Time_Stamp', 'Ax', 'Ay', 'Az', 'Gx', 'Gy', 'Gz', 'Mx', 'My', 'Mz', 'Activity_Label']
    # data = pd.read_csv(file_path, header=None, names=column_names)
    data = pd.read_excel(file_path, names=column_names)
    return data


# 数据标准化，标准分数：变量值与其平均数的离差除以标准差后的值
# 标准分数具有平均数为0，标准差为1的特性
def feature_normalize(dataset):
    mu = np.mean(dataset, axis=0)
    sigma = np.std(dataset, axis=0)
    # print(dataset)
    return (dataset - mu) / sigma


# 创建时间窗口，90 × 50ms，也就是 4.5 秒，每次前进 45 条记录，半重叠的方式。
def windows(data, size):
    start = 0
    while start < data.count():
        yield start, start + size
        start += (size / 2)


# 创建输入数据，每一组数据包含 x, y, z 三个轴的 90 条连续记录，
# 用 `stats.mode` 方法获取这 90 条记录中出现次数最多的行为
# 作为该组行为的标签，这里有待商榷，其实可以完全使用同一种行为的数据记录
# 来创建一组数据用于输入的。
def segment_signal(data, window_size=90):
    segments = np.empty((0, window_size, 9))
    labels = np.empty(0)
    print(len(data['Time_Stamp']))
    count = 0
    for (start, end) in windows(data['Time_Stamp'], window_size):
        count += 1
        ax = data["Ax"][int(start):int(end)]
        ay = data["Ay"][int(start):int(end)]
        az = data["Az"][int(start):int(end)]
        gx = data["Gx"][int(start):int(end)]
        gy = data["Gy"][int(start):int(end)]
        gz = data["Gz"][int(start):int(end)]
        mx = data["Mx"][int(start):int(end)]
        my = data["My"][int(start):int(end)]
        mz = data["Mz"][int(start):int(end)]
        if len(dataset['Time_Stamp'][int(start):int(end)]) == window_size:
            segments = np.vstack([segments, np.dstack([ax, ay, az, gx, gy, gz, mx, my, mz])])
            labels = np.append(labels, stats.mode(data["Activity_Label"][int(start):int(end)])[0][0])
    return segments, labels


# 初始化神经网络参数
def weight_variable(shape, name):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name=name)


# 初始化神经网络参数
def bias_variable(shape, name):
    initial = tf.constant(0.0, shape=shape)
    return tf.Variable(initial, name=name)


# 执行卷积操作
def depthwise_conv2d(x, W):
    return tf.nn.depthwise_conv2d(x, W, [1, 1, 1, 1], padding='VALID')


# 为输入数据的每个 channel 执行一维卷积，并输出到 ReLU 激活函数
def apply_depthwise_conv(x, kernel_size, num_channels, depth):
    weights = weight_variable([1, kernel_size, num_channels, depth], name='weight')
    biases = bias_variable([depth * num_channels], name='bias')
    return tf.nn.relu(tf.add(depthwise_conv2d(x, weights), biases))


# 在卷积层输出进行一维 max pooling
def apply_max_pool(x, kernel_size, stride_size):
    return tf.nn.max_pool(x, ksize=[1, 1, kernel_size, 1],
                          strides=[1, 1, stride_size, 1], padding='VALID')


# 参数概要
def variable_summaries(var):
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)  # 平均值
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)  # 标准差
        tf.summary.scalar('max', tf.reduce_max(var))  # 最大值
        tf.summary.scalar('min', tf.reduce_min(var))  # 最小值
        tf.summary.histogram('histogram', var)  # 直方图


def plot_axis(ax, x, y, title):
    ax.plot(x, y)
    ax.set_title(title)
    ax.xaxis.set_visible(False)
    ax.set_ylim([min(y) - np.std(y), max(y) + np.std(y)])
    ax.set_xlim([min(x), max(x)])
    ax.grid(True)


def plot_activity(activity, data):
    fig, (ax0, ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8) = plt.subplots(nrows=9, figsize=(15, 20), sharex=True)
    plot_axis(ax0, data['Time_Stamp'], data['Ax'], 'Ax')
    plot_axis(ax1, data['Time_Stamp'], data['Ay'], 'Ay')
    plot_axis(ax2, data['Time_Stamp'], data['Az'], 'Az')
    plot_axis(ax3, data['Time_Stamp'], data['Gx'], 'Gx')
    plot_axis(ax4, data['Time_Stamp'], data['Gy'], 'Gy')
    plot_axis(ax5, data['Time_Stamp'], data['Gz'], 'Gz')
    plot_axis(ax6, data['Time_Stamp'], data['Mx'], 'Mx')
    plot_axis(ax7, data['Time_Stamp'], data['My'], 'My')
    plot_axis(ax8, data['Time_Stamp'], data['Mz'], 'Mz')
    plt.subplots_adjust(hspace=0.2)
    fig.suptitle(activity)
    plt.subplots_adjust(top=0.90)
    plt.show()


dataset = read_data('shoab_data/Wrist.xlsx')
dataset['Ax'] = feature_normalize(dataset['Ax'])
dataset['Ay'] = feature_normalize(dataset['Ay'])
dataset['Az'] = feature_normalize(dataset['Az'])
dataset['Gx'] = feature_normalize(dataset['Gx'])
dataset['Gy'] = feature_normalize(dataset['Gy'])
dataset['Gz'] = feature_normalize(dataset['Gz'])
dataset['Mx'] = feature_normalize(dataset['Mx'])
dataset['My'] = feature_normalize(dataset['My'])
dataset['Mz'] = feature_normalize(dataset['Mz'])

for activity in np.unique(dataset["Activity_Label"]):
    subset = dataset[dataset["Activity_Label"] == activity][:180]
    plot_activity(activity, subset)

segments, labels = segment_signal(dataset)
labels = np.asarray(pd.get_dummies(labels), dtype=np.int8)
# 创建输入
reshaped_segments = segments.reshape(len(segments), 1, 90, 9)

# 在准备好的输入数据中，分别抽取训练数据和测试数据，按照 70/30 原则来做。
train_test_split = np.random.rand(len(reshaped_segments)) < 0.70
train_x = reshaped_segments[train_test_split]
train_y = labels[train_test_split]
test_x = reshaped_segments[~train_test_split]
test_y = labels[~train_test_split]

# 定义输入数据的维度和标签个数
input_height = 1
input_width = 90
num_labels = 6
num_channels = 9

batch_size = 100
kernel_size = 60
depth = 120

# 隐藏层神经元个数
num_hidden = 1000

learning_rate = 0.0001

# 降低 cost 的迭代次数
training_epochs = 120

# 输入图片是28*28
n_inputs = 90  # 输入一行，一行有28个数据
max_time = 9  # 一共28行
lstm_size = 100  # 隐层单元
n_classes = 6  # 10个分类
n_batch = reshaped_segments.shape[0] // batch_size  # 计算一共有多少个批次

pb_file_path = os.getcwd()

# 下面是使用 Tensorflow 创建神经网络的过程。
with tf.name_scope('input'):
    X = tf.placeholder(tf.float32, shape=[None, input_height, input_width, num_channels])
    Y = tf.placeholder(tf.float32, shape=[None, num_labels])

# 初始化权值
weights = tf.Variable(tf.truncated_normal([lstm_size, n_classes], stddev=0.1))
# 初始化偏置值
biases = tf.Variable(tf.constant(0.1, shape=[n_classes]))


# 定义RNN网络
def RNN(X, weights, biases):
    # inputs=[batch_size, max_time, n_inputs]
    with tf.name_scope('reshape'):
        inputs = tf.reshape(X, [-1, max_time, n_inputs])
    # 定义LSTM基本CELL
    with tf.name_scope('basicLSTMCell'):
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(lstm_size)
    with tf.name_scope('dynamicRNN'):
        outputs, final_state = tf.nn.dynamic_rnn(lstm_cell, inputs, dtype=tf.float32)
    with tf.name_scope('softmax'):
        results = tf.nn.softmax(tf.matmul(final_state[1], weights) + biases)
    return results


y_ = RNN(X, weights, biases)

with tf.name_scope('loss'):
    loss = -tf.reduce_sum(Y * tf.log(y_))
    tf.summary.scalar('loss', loss)
with tf.name_scope('train'):
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

with tf.name_scope('accuracy'):
    with tf.name_scope('correct_prediction'):
        # 结果存放在一个布尔型列表中
        correct_prediction = tf.equal(tf.argmax(y_, 1), tf.argmax(Y, 1))
    with tf.name_scope('accuracy'):
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar('accuracy', accuracy)

cost_history = np.empty(shape=[1], dtype=float)

# 合并所有的summary
merged = tf.summary.merge_all()

# 开始训练
with tf.Session() as session:
    tf.global_variables_initializer().run()
    writer = tf.summary.FileWriter('logs/', session.graph)
    # convert_variables_to_constants 需要指定output_node_names，list()，可以多个
    # 开始迭代
    for epoch in range(training_epochs):
        for b in range(n_batch):
            offset = (b * batch_size) % (train_y.shape[0] - batch_size)
            batch_x = train_x[offset:(offset + batch_size), :, :, :]
            batch_y = train_y[offset:(offset + batch_size), :]
            _, c, summary = session.run([optimizer, loss, merged], feed_dict={X: batch_x, Y: batch_y})
            # _, c = session.run([optimizer, loss], feed_dict={X: batch_x, Y: batch_y})
            cost_history = np.append(cost_history, c)
        writer.add_summary(summary, epoch)
        print("Epoch {}: Training Loss = {}, Training Accuracy = {}".format(
            epoch, c, session.run(accuracy, feed_dict={X: train_x, Y: train_y})))
        tf.train.write_graph(session.graph_def, pb_file_path, 'expert-graph.pb', as_text=False)

    y_p = tf.argmax(y_, 1)
    y_true = np.argmax(test_y, 1)
    final_acc, y_pred = session.run([accuracy, y_p], feed_dict={X: test_x, Y: test_y})
    print("Testing Accuracy: {}".format(final_acc))

    # tf.train.write_graph(session.graph_def, pb_file_path, 'expert-graph.pb', as_text=False)

    temp_y_true = np.unique(y_true)
    temp_y_pred = np.unique(y_pred)
    np.save("y_true", y_true)
    np.save("y_pred", y_pred)
    print("temp_y_true", temp_y_true)
    print("temp_y_pred", temp_y_pred)
    # 计算模型的 metrics
    print("Precision", precision_score(y_true.tolist(), y_pred.tolist(), average='weighted'))
    print("Recall", recall_score(y_true, y_pred, average='weighted'))
    print("f1_score", f1_score(y_true, y_pred, average='weighted'))
    print("confusion_matrix")
    conf_mat = confusion_matrix(y_true, y_pred)
    print(conf_mat)

    activity_lables = ['downstairs', 'running', 'sitting', 'standing', 'upstairs', 'walking']
    df_cm = pd.DataFrame(conf_mat, index=[i for i in activity_lables],
                         columns=[i for i in activity_lables])
    plt.figure(figsize=(10, 7))
    sn.heatmap(df_cm, annot=True, fmt="d", cmap="YlGnBu")
    plt.title('confusion matrix')
    plt.show()
