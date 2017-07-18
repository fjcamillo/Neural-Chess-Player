import tensorflow as tf
import numpy as np

def convolutional_layer(data, weight, strides, padding):
    conv = tf.conv2d(data, weight, strides, padding)
    

def main():
    with tf.device('./cpu:0'):


if __name__ == '__main__':
    main()
