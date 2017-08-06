import tensorflow as tf
import numpy as np
import pandas as pd


def cnn_model(device='/cpu:0', epoch=1):
    with tf.device(device):
        x = tf.placeholder(tf.float32, [None, 48, 48, 3])
        y = tf.placeholder(tf.float32, [None, 12])
        keep_prob = tf.placeholder(tf.float32)

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
        max_1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        conv2 = tf.nn.relu(tf.nn.conv2d(max_1, conv2_w, strides=[1, 1, 1, 1], padding='SAME'))
        max_2 = tf.nn.max_pool(conv2, kstrides = [1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        conv3 = tf.nn.relu(tf.nn.conv2d(max_2, conv3_w, stride[1, 1, 1, 1], padding='SAME'))
        max_3 = tf.nn.max_pool(conv3, kstrides=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        reshaped = tf.reshape(max_3, [-1, 48*48*32])

        fc1 = tf.nn.dropout(tf.nn.relu(tf.matmul(reshaped, fc1_w) + fc1_b), keep_prob)
        fc2 = tf.nn.dropout(tf.nn.relu(tf.matmul(fc1, fc2_w) + fc2_b), keep_prob)
        fc3 = tf.matmul(fc2, tf3_w) + fc3_b
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=fc3))

        train = tf.train.AdamOptimizer().minimize(cross_entropy)

        cross_prediction = tf.equal(tf.argmax(fc3, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(cross_prediction, tf.float32))

    sess = tf.InteractiveSession().run()
