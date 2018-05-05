import os

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sn
from tensorflow.python.framework import graph_util

MODEL_DIR = "model/pb"
MODEL_NAME = "frozen_model.pb"

if not tf.gfile.Exists(MODEL_DIR):  # 创建目录
    tf.gfile.MakeDirs(MODEL_DIR)


def freeze_graph(model_folder):
    checkpoint = tf.train.get_checkpoint_state(model_folder)  # 检查目录下ckpt文件状态是否可用
    input_checkpoint = checkpoint.model_checkpoint_path  # 得ckpt文件路径
    output_graph = os.path.join(MODEL_DIR, MODEL_NAME)  # PB模型保存路径

    output_node_names = "accuracy/accuracy/_y"  # 原模型输出操作节点的名字
    saver = tf.train.import_meta_graph(input_checkpoint + '.meta',
                                       clear_devices=True)  # 得到图、clear_devices ：Whether or not to clear the device field for an `Operation` or `Tensor` during import.

    graph = tf.get_default_graph()  # 获得默认的图
    input_graph_def = graph.as_graph_def()  # 返回一个序列化的图代表当前的图

    with tf.Session() as sess:
        saver.restore(sess, input_checkpoint)  # 恢复图并得到数据

        print("accuracy/accuracy/_y : ", sess.run("accuracy/accuracy/_y:0", feed_dict={
            "input_holder_x:0": train_x}))

        output_graph_def = graph_util.convert_variables_to_constants(  # 模型持久化，将变量值固定
            sess,
            input_graph_def,
            output_node_names.split(",")  # 如果有多个输出节点，以逗号隔开
        )
        with tf.gfile.GFile(output_graph, "wb") as f:  # 保存模型
            f.write(output_graph_def.SerializeToString())  # 序列化输出
        print("%d ops in the final graph." % len(output_graph_def.node))  # 得到当前图有几个操作节点

        for op in graph.get_operations():
            print(op.name, op.values())


freeze_graph("model/ckpt")


# 加载数据集
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
# 来创建一组数据用于输入的。
def segment_signal(data, window_size=90):
    segments = np.empty((0, window_size, 3))
    labels = np.empty(0)
    print(len(data['Time_Stamp']))
    count = 0
    for (start, end) in windows(data['Time_Stamp'], window_size):
        count += 1
        ax = data["Ax"][int(start):int(end)]
        ay = data["Ay"][int(start):int(end)]
        az = data["Az"][int(start):int(end)]
        if len(dataset['Time_Stamp'][int(start):int(end)]) == window_size:
            segments = np.vstack([segments, np.dstack([ax, ay, az])])
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
    fig, (ax0, ax1, ax2) = plt.subplots(nrows=3, figsize=(15, 20), sharex=True)
    plot_axis(ax0, data['Time_Stamp'], data['Ax'], 'Ax')
    plot_axis(ax1, data['Time_Stamp'], data['Ay'], 'Ay')
    plot_axis(ax2, data['Time_Stamp'], data['Az'], 'Az')
    plt.subplots_adjust(hspace=0.2)
    fig.suptitle(activity)
    plt.subplots_adjust(top=0.90)
    plt.show()


dataset = read_data('shoab_data/Wrist.xlsx')
dataset['Ax'] = feature_normalize(dataset['Ax'])
dataset['Ay'] = feature_normalize(dataset['Ay'])
dataset['Az'] = feature_normalize(dataset['Az'])

for activity in np.unique(dataset["Activity_Label"]):
    subset = dataset[dataset["Activity_Label"] == activity][:180]
    plot_activity(activity, subset)

segments, labels = segment_signal(dataset)
labels = np.asarray(pd.get_dummies(labels), dtype=np.int8)
# 创建输入
reshaped_segments = segments.reshape(len(segments), 1, 90, 3)

# 在准备好的输入数据中，分别抽取训练数据和测试数据，按照 70/30 原则来做。
train_test_split = np.random.rand(len(reshaped_segments)) < 0.70
train_x = reshaped_segments[train_test_split]
train_y = labels[train_test_split]
test_x = reshaped_segments[~train_test_split]
test_y = labels[~train_test_split]

pb_file_path = os.getcwd()

# 定义输入数据的维度和标签个数
input_height = 1
input_width = 90
num_labels = 6
num_channels = 3

batch_size = 100
kernel_size = 60
depth = 120

# 隐藏层神经元个数
num_hidden = 100

learning_rate = 0.0001

# 降低 cost 的迭代次数
training_epochs = 45

total_batchs = reshaped_segments.shape[0] // batch_size

# 下面是使用 Tensorflow 创建神经网络的过程。
with tf.name_scope('input'):
    X = tf.placeholder(tf.float32, shape=[None, input_height, input_width, num_channels], name="input_holder")
    Y = tf.placeholder(tf.float32, shape=[None, num_labels])

with tf.name_scope('conv1'):
    c = apply_depthwise_conv(X, kernel_size, num_channels, depth)
    p = apply_max_pool(c, 20, 2)
with tf.name_scope('conv2'):
    c = apply_depthwise_conv(p, 6, depth * num_channels, depth // 10)
    p = apply_max_pool(c, 1, 2)

shape = p.get_shape().as_list()
with tf.name_scope('input_reshape'):
    p_flat = tf.reshape(p, [-1, shape[1] * shape[2] * shape[3]])

with tf.name_scope('layer1'):
    with tf.name_scope('weights1'):
        f_weights_l1 = weight_variable([shape[1] * shape[2] * depth *
                                        num_channels * (depth // 10), num_hidden],
                                       name='weight')
        variable_summaries(f_weights_l1)
    with tf.name_scope('biases1'):
        f_biases_l1 = bias_variable([num_hidden], name='bias')
        variable_summaries(f_biases_l1)
    with tf.name_scope('tanh'):
        f = tf.nn.tanh(tf.add(tf.matmul(p_flat, f_weights_l1), f_biases_l1))

with tf.name_scope('layer2'):
    with tf.name_scope('weights2'):
        out_weights = weight_variable([num_hidden, num_labels], name='weight')
        variable_summaries(out_weights)
    with tf.name_scope('biases2'):
        out_biases = bias_variable([num_labels], name='bias')
        variable_summaries(out_biases)
    with tf.name_scope('softmax'):
        y_ = tf.nn.softmax(tf.matmul(f, out_weights) + out_biases)

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
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name="_y")
        tf.summary.scalar('accuracy', accuracy)

cost_history = np.empty(shape=[1], dtype=float)

# 合并所有的summary
merged = tf.summary.merge_all()

output_graph = "model/pb/model.pb"

# saver = tf.train.Saver()  # 声明saver用于保存模型

# 开始训练
with tf.Session() as session:
    tf.global_variables_initializer().run()
    writer = tf.summary.FileWriter('logs/', session.graph)
    # 开始迭代
    for epoch in range(training_epochs):
        for b in range(total_batchs):
            offset = (b * batch_size) % (train_y.shape[0] - batch_size)
            batch_x = train_x[offset:(offset + batch_size), :, :, :]
            batch_y = train_y[offset:(offset + batch_size), :]
            _, c, summary = session.run([optimizer, loss, merged], feed_dict={X: batch_x, Y: batch_y})
            # _, c = session.run([optimizer, loss], feed_dict={X: batch_x, Y: batch_y})
            cost_history = np.append(cost_history, c)
        writer.add_summary(summary, epoch)
        print("Epoch {}: Training Loss = {}, Training Accuracy = {}".format(
            epoch, c, session.run(accuracy, feed_dict={X: train_x, Y: train_y})))
    y_p = tf.argmax(y_, 1)
    y_true = np.argmax(test_y, 1)
    final_acc, y_pred = session.run([accuracy, y_p], feed_dict={X: test_x, Y: test_y})
    print("Testing Accuracy: {}".format(final_acc))

    graph_def = tf.get_default_graph().as_graph_def()  # 得到当前的图的 GraphDef 部分，通过这个部分就可以完成重输入层到输出层的计算过程

    output_graph_def = graph_util.convert_variables_to_constants(session, graph_def, ["accuracy/accuracy/_y"])
    with tf.gfile.GFile(output_graph, "wb") as f:  # 保存模型
        f.write(output_graph_def.SerializeToString())  # 序列化输出
    print("%d ops in the final graph." % len(output_graph_def.node))

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

for op in tf.get_default_graph().get_operations():  # 打印模型节点信息
    print(op.name, op.values())
