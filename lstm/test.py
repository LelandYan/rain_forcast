import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt



CSV_FILE_PATH = '02.csv'
df = pd.read_csv(CSV_FILE_PATH)
shapes = df.values.shape
data = df.values[:, 4:shapes[1] - 1]

CSV_FILE_PATH = '02.csv'
df = pd.read_csv(CSV_FILE_PATH)
shapes = df.values.shape
data = df.values[:, 4:shapes[1] - 3]
result = df.values[:, shapes[1] - 1:shapes[1]]
train_x, test_x, train_y, test_y = train_test_split(data, result, test_size=0.3)
n_features = train_x.shape[1]
train_y = np.array(train_y.flatten())
test_y = np.array(test_y.flatten())


n_classes = 1
batch_size = 200


def get_batch(x, y, batch):
    n_samples = len(x)
    for i in range(batch, n_samples, batch):
        yield x[i - batch:i], y[i - batch:i]


x_input = tf.placeholder(tf.float32, shape=[None, n_features], name='x_input')
y_input = tf.placeholder(tf.int32, shape=[None], name='y_input')

W1 = tf.Variable(tf.truncated_normal([n_features, 100]), name='W')
b1 = tf.Variable(tf.zeros([100]) + 0.1, name='b')

logits1 = tf.sigmoid(tf.matmul(x_input, W1) + b1)

W = tf.Variable(tf.truncated_normal([100, n_classes]), name='W')
b = tf.Variable(tf.zeros([n_classes]), name='b')

logits = tf.tanh(tf.matmul(logits1, W) + b)

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_input, logits=logits))
# 使用AdamOptimizer进行优化
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
# 结果放入一个布尔型列表中
correct_prediction = tf.equal(tf.argmax(y_input, 0), tf.argmax(logits, 0))
# 求准确率
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(25):  # 训练次数
        for tx, ty in get_batch(train_x, train_y, batch_size):  # 得到一个batch的数据

            # tx = tx.tolist()
            # ty = ty[:, np.newaxis].tolist()
            sess.run(train_step, feed_dict={x_input: tx, y_input: ty})
    #print(sess.run(logits,feed_dict={x_input: test_x, y_input: test_y}))
    acc,cost = sess.run([accuracy,cross_entropy], feed_dict={x_input: test_x, y_input: test_y})
    print("Iter " + str(epoch) + ",Testing Accuracy= " + str(acc))
    print(cost)
