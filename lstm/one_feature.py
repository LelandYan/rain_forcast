
# _*_ coding: utf-8 _*_
__author__ = 'LelandYan'
__date__ = '2018/11/22 21:05'

import tensorflow as tf

X = tf.placeholder(tf.float32,[None,1])
w = tf.Variable(tf.zeros([1,1]))
b = tf.Variable(tf.zeros([1]))
y = tf.matmul(X,w)+b
Y = tf.placeholder(tf.float32,[None,1])
