import tensorflow as tf
import numpy as np
import pandas as pd

def create_conv2d(device='/cpu:0'):
    with tf.device(device):
        
