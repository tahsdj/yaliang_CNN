import numpy as np
import random
import os
import tensorflow as tf
import keras

def weight_variable(shape, name="W"):
#     initial = tf.truncated_normal(shape, stddev=0.1)
#     return tf.Variable(initial)
    return tf.get_variable(name, shape=shape, initializer=tf.contrib.layers.xavier_initializer())

def bias_variable(shape, name='b'):
    initial = tf.constant(0.01, shape=shape)
#     return tf.Variable(initial)
    return tf.get_variable(name, shape=shape, initializer=tf.constant_initializer(0.01))

def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)

def batch_norm(w_plus_b, out_size=1, name='batch_norm'):
    fc_mean, fc_var = tf.nn.moments(
            w_plus_b,
            axes=[0],   # 想要 normalize 的维度, [0] 代表 batch 维度
                        # 如果是图像数据, 可以传入 [0, 1, 2], 相当于求[batch, height, width] 的均值/方差, 注意不要加入 channel 维度
        )
    #   scale = tf.Variable(tf.ones([1]))
    #   shift = tf.Variable(tf.zeros([1]))
    scale = tf.get_variable(name, tf.ones([1]))
    shift = tf.get_variable(name, tf.zeros([1]))
    epsilon = 0.001
    return tf.nn.batch_normalization(w_plus_b, fc_mean, fc_var, shift, scale, epsilon)

def batch_norm(x,axis=3,training=True):
    return tf.layers.batch_normalization(x, axis=3, training=training)

def max_pool_2x2(x):
    # stride [1, x_movement, y_movement, 1]
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
  
def max_pool_3x3(x):
    # stride [1, x_movement, y_movement, 1]
    return tf.nn.max_pool(x, ksize=[1,3,3,1], strides=[1,3,3,1], padding='SAME')

def max_pool_4x4(x):
    # stride [1, x_movement, y_movement, 1]
    return tf.nn.max_pool(x, ksize=[1,4,4,1], strides=[1,4,4,1], padding='SAME')

def max_pool_8x8(x):
    # stride [1, x_movement, y_movement, 1]
    return tf.nn.max_pool(x, ksize=[1,8,8,1], strides=[1,8,8,1], padding='SAME')

def avg_pool_6x6(x):
    # stride [1, x_movement, y_movement, 1]
    return tf.nn.avg_pool(x, ksize=[1,6,6,1], strides=[1,6,6,1], padding='SAME')

def avg_pool_global2D(x):
    return tf.reduce_mean(x, axis=[1,2])

def conv2D(x,kernel_size,out_size, name="con2d"):
    in_size = x.get_shape().as_list()[-1]
    w = weight_variable([kernel_size, kernel_size, in_size, out_size], name=("w"+name))
    b = bias_variable([out_size], name=("b_"+ name))
    
    return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME') + b
  
def fc(x , out_size, name="fc"):
    in_size = x.get_shape().as_list()[-1]
    w = weight_variable([in_size, out_size], name=("w_"+name))
    b = bias_variable([out_size], name=("b_"+name))
    
    return tf.matmul(x,w) + b

def dropout(x,prob):
    return tf.nn.dropout(x, prob)

def relu(x):
    return tf.nn.relu(x)