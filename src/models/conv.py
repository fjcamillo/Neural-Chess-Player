import tensorflow as tf
import numpy as np
import pandas as pd


def cnn_model(device='/cpu:0'):
    with tf.device(device):
        x = tf.placeholder(tf.float32, [None, 48, 48, 3])
        y = tf.placeholder(tf.float32, [None, 12])

        conv1_w = tf.Variable(tf.truncate_normal([8, 8, 3, 8]), tf.float32)
        conv1_b = tf.Variable(tf.zeros([8]))

        conv2_w = tf.Variable(tf.truncate_normal([4, 4, 8, 16]), tf.float32)
        conv2_b = tf.Variable(tf.zeros([16]))

        conv3_w = tf.Variable(tf.truncate_normal([4, 4, 16, 32]), tf.float32)
        conv3_b = tf.Variable(tf.zeros[32])

        fc1_w = tf.Variable(tf.truncate_normal([48*48*32, 200]), tf.float32)
        fc1_b = tf.Variable(tf.ones[200])

        fc2_w = tf.Variable(tf.truncate_normal([200, 200]), tf.float32)
        fc2_b = tf.Variable(tf.ones[200])

        fc3_w = tf.Variable(tf.truncate_normal([200, 12]), tf.float32)
        fc3_b = tf.Variable(tf.ones[12])

        #MODEL

        conv1 = tf.nn.relu(tf.nn.conv2d(x, conv1_w, strides=[1, 1, 1, 1], padding='SAME'))
        conv2 = tf.nn.relu(tf.nn.conv2d(conv1, conv2_w, strides=[1, 1, 1, 1], padding='SAME'))
        con
