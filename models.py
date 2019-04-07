import numpy as np
import random
import tensorflow as tf
import keras
from layers import *

# Network in Network
class NiN():
    def __init__(self,temperature=7):
        self.temperature = temperature # for knowledge distallation
        self.keep_prob = tf.placeholder(tf.float32) # dropout probability 
        self.learning_rate = tf.placeholder(tf.float32) 
        self.alpha = 0.7

    def build(self,x,reuse=False):
        with tf.variable_scope('NiN', reuse=reuse):
        
            # It is a 3 conv layers NiN
            
            # conv1
            
            x = conv2D(x,5,96, name="conv1")
            x = batch_norm(x)
            x = relu(x)
            
            with tf.variable_scope('conv1-ccp1', reuse=reuse):
                x = conv2D(x,1,80)
                x = batch_norm(x)
                x = relu(x)
            
            with tf.variable_scope('conv1-ccp2', reuse=reuse):
                x = conv2D(x,1,48)
                attention_map.append(x) # low level attention maps
                x = batch_norm(x)
                x = relu(x)
            
            x = max_pool_3x3(x)
            x = dropout(x,self.keep_prob)
            
            # conv2
            
            x = conv2D(x,5,96,name="conv2")
            x = batch_norm(x)
            x = relu(x)
            
            with tf.variable_scope('conv2-ccp1', reuse=reuse):
                x = conv2D(x,1,96)
                x = batch_norm(x)
                x = relu(x)

            with tf.variable_scope('conv2-ccp2', reuse=reuse):
                x = conv2D(x,1,96)
                x = batch_norm(x)
                x = relu(x)
            
            x = max_pool_3x3(x)
            x = dropout(x,self.keep_prob)
            
            # conv3
            
            x = conv2D(x,3,96, name="conv3")
            x = batch_norm(x)
            x = relu(x)
            
            # conv2 cccp1
            
            with tf.variable_scope('conv3-ccp1', reuse=reuse):
                x = conv2D(x,1,96)
                x = batch_norm(x)
                x = relu(x)
            
            # conv2 cccp2
            
            with tf.variable_scope('conv3-ccp2', reuse=reuse):
                x = conv2D(x,1,3)
                x = batch_norm(x)
                x = relu(x)
            
            x = avg_pool_global2D(x)
            
            
            # flatten
            x = tf.layers.flatten(x)
            
            # classification
            self.pred = x
            self.softmax_soft = tf.nn.softmax(self.pred/self.temperature)
            self.drop_pred = dropout(self.pred, self.keep_prob) # with dropout
            self.attention_map = attention_map

            return self.softmax_soft, self.pred, self.attention_map
        
    def build_loss_fn(self,y):
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.pred, labels=y))
        self.train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
        
    # attention map
    def build_AT(self,x,shape=(11,11)):
        x = tf.abs(x)
        x = tf.reduce_sum(x, axis=3)
        x = tf.reshape(x, [-1, shape[0], shape[1], 1])
        return x

    def compute_accuracy(self, v_xs, v_ys):
        y_pre = sess.run(self.pred, feed_dict={xs: v_xs, self.keep_prob: 1})
        correct_prediction = tf.equal(tf.argmax(y_pre,1), tf.argmax(v_ys,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        acc = sess.run(accuracy, feed_dict={xs: v_xs})
        return acc