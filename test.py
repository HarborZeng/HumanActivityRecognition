import numpy as np
import pandas as pd
import tensorflow as tf
from scipy import stats
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix


# 加载数据集
def read_data(file_path):
    column_names = ['user-id', 'activity', 'timestamp', 'x-axis', 'y-axis', 'z-axis']
    # data = pd.read_csv(file_path, header=None, names=column_names)
    data = pd.read_csv(file_path, header=None, names=column_names, comment=';')
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
    segments = np.empty((0, window_size, 3))
    labels = np.empty(0)
    print(len(data['timestamp']))
    count = 0
    for (start, end) in windows(data['timestamp'], window_size):
        count += 1
        x = data["x-axis"][int(start):int(end)]
        y = data["y-axis"][int(start):int(end)]
        z = data["z-axis"][int(start):int(end)]
        if len(dataset['timestamp'][int(start):int(end)]) == window_size:
            segments = np.vstack([segments, np.dstack([x, y, z])])
            labels = np.append(labels, stats.mode(data["activity"][int(start):int(end)])[0][0])
    return segments, labels


# 初始化神经网络参数
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


# 初始化神经网络参数
def bias_variable(shape):
    initial = tf.constant(0.0, shape=shape)
    return tf.Variable(initial)


# 执行卷积操作
def depthwise_conv2d(x, W):
    return tf.nn.depthwise_conv2d(x, W, [1, 1, 1, 1], padding='VALID')


# 为输入数据的每个 channel 执行一维卷积，并输出到 ReLU 激活函数
def apply_depthwise_conv(x, kernel_size, num_channels, depth):
    weights = weight_variable([1, kernel_size, num_channels, depth])
    biases = bias_variable([depth * num_channels])
    return tf.nn.relu(tf.add(depthwise_conv2d(x, weights), biases))


# 在卷积层输出进行一维 max pooling
def apply_max_pool(x, kernel_size, stride_size):
    return tf.nn.max_pool(x, ksize=[1, 1, kernel_size, 1],
                          strides=[1, 1, stride_size, 1], padding='VALID')


dataset = read_data('WISDM_ar_v1.1_raw.txt')
dataset['x-axis'] = feature_normalize(dataset['x-axis'])
dataset['y-axis'] = feature_normalize(dataset['y-axis'])
dataset['z-axis'] = feature_normalize(dataset['z-axis'])

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

# 定义输入数据的维度和标签个数
input_height = 1
input_width = 90
num_labels = 6
num_channels = 3

batch_size = 10
kernel_size = 60
depth = 60

# 隐藏层神经元个数
num_hidden = 1000

learning_rate = 0.0001

# 降低 cost 的迭代次数
training_epochs = 12

total_batchs = reshaped_segments.shape[0] // batch_size

# 下面是使用 Tensorflow 创建神经网络的过程。
X = tf.placeholder(tf.float32, shape=[None, input_height, input_width, num_channels])
Y = tf.placeholder(tf.float32, shape=[None, num_labels])

c = apply_depthwise_conv(X, kernel_size, num_channels, depth)
p = apply_max_pool(c, 20, 2)
c = apply_depthwise_conv(p, 6, depth * num_channels, depth // 10)

shape = c.get_shape().as_list()
c_flat = tf.reshape(c, [-1, shape[1] * shape[2] * shape[3]])

f_weights_l1 = weight_variable([shape[1] * shape[2] * depth * num_channels * (depth // 10), num_hidden])
f_biases_l1 = bias_variable([num_hidden])
f = tf.nn.tanh(tf.add(tf.matmul(c_flat, f_weights_l1), f_biases_l1))

out_weights = weight_variable([num_hidden, num_labels])
out_biases = bias_variable([num_labels])

y_ = tf.nn.softmax(tf.matmul(f, out_weights) + out_biases)

loss = -tf.reduce_sum(Y * tf.log(y_))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)

correct_prediction = tf.equal(tf.argmax(y_, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

cost_history = np.empty(shape=[1], dtype=float)

# 开始训练
with tf.Session() as session:
    tf.global_variables_initializer().run()
    # 开始迭代
    for epoch in range(training_epochs):
        for b in range(total_batchs):
            offset = (b * batch_size) % (train_y.shape[0] - batch_size)
            batch_x = train_x[offset:(offset + batch_size), :, :, :]
            batch_y = train_y[offset:(offset + batch_size), :]
            _, c = session.run([optimizer, loss], feed_dict={X: batch_x, Y: batch_y})
            cost_history = np.append(cost_history, c)
        print("Epoch {}: Training Loss = {}, Training Accuracy = {}".format(
            epoch, c, session.run(accuracy, feed_dict={X: train_x, Y: train_y})))
    y_p = tf.argmax(y_, 1)
    y_true = np.argmax(test_y, 1)
    final_acc, y_pred = session.run([accuracy, y_p], feed_dict={X: test_x, Y: test_y})
    print("Testing Accuracy: {}".format(final_acc))
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
    print(confusion_matrix(y_true, y_pred))
