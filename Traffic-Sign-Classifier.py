#Importing Libraries
import numpy as np
import math
import os
import random 
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
#tensorflow
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib.layers import flatten

#downloading data

#defining some hyperparameters
n_classes = 43
epochs = 10
batch_size = 128
rate = 0.001

#defining the features and labels
x = tf.placeholder(tf.float32, (None, 32, 32, 1))
y = tf.placeholder(tf.int32, (None))
one_hot_y = tf.one_hot(y, 43)

#preprocess the data

#identifyign the Convolution Neural Network layers
#following the same architecture as the LeNet but with different output
def traffic_classifier(x):
    #definining the mean and the standard deviation
    mu = 0
    sigma = 0.1

    #First convolution
    '''The Input for the images should be 28*28*6 for the first convolution layer'''
    conv1_w = tf.Variable(tf.truncated_normal(shape=(5,5,1,6), mean = mu, stddev = sigma))
    conv1_b = tf.Variable(tf.zeros(6))
    conv1 = tf.nn.conv2d(x, conv1_w, strides=[1,1,1,1], padding='VALID') + conv1_b

    #Activation function
    conv1 = tf.nn.relu(conv1)
    '''The Pooling for the first convolution layer should be 14*14*6'''
    conv1 = tf.nn.max_pool(conv1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')
     
    #Second Convolution
    '''The Input for the images should be 10*10*16 for the first convolution layer'''
    conv2_w = tf.Variable(tf.truncated_normal(shape=(5,5,6,16), mean=mu, stddev=sigma))
    conv2_b = tf.Variable(tf.zeros(6))
    conv2 = tf.nn.conv2d(conv1, conv2_w, strides=[1,1,1,1], padding='VALID') + conv2_b

    #Activation Function
    conv2 = tf.nn.relu(conv2)
    '''The Pooling for the first convolution layer should be 5*5*16'''
    conv2 = tf.nn.max_pool(conv2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')

    #Flatten the Output 
    '''This to flatten the output to 1D rather than 3D'''
    fc0 = flatten(conv2)

    #First Fully Connected Layer
    '''The Input for the images should be 28*28*6 for the first convolution layer'''
    fc1_w = tf.Variable(tf.truncated_normal(shape =(400,120), mean=mu, stddev=sigma))
    fc1_b = tf.Variable(tf.zeros(120))
    fc1 = tf.matmul(fc0, fc1_w) + fc1_b

    '''The output should be 120'''
    fc1 = tf.nn.relu(fc1)

    #Second Fully Connected Layer
    '''The Input for the images should be 28*28*6 for the first convolution layer'''
    fc2_w = tf.Variable(tf.truncated_normal(shape=(120, 84), mean=mu, stddev=sigma))
    fc2_b = tf.Variable(tf.zeros(84))
    fc2 = tf.matmul(fc1, fc2_w) + fc2_b

    '''The output should be 84'''
    fc2 = tf.nn.relu(fc2)

    #Third Fully Connected Layer
    '''The Input for the images should be 28*28*6 for the first convolution layer'''
    fc3_w = tf.Variable(tf.truncated_normal(shape=(84,10), mean=mu, stddev=sigma))
    fc3_b = tf.Variable(tf.zeros(10))
    logits = tf.matmul(fc2, fc3_w) + fc3_b
    '''The Final output should be 10 as per the MNIST data'''
    return logits

logits = traffic_classifier(x)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_y, logits=logits)
loss_function = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate=rate)
training = optimizer.minimize(loss_function)

def evaluate(x_data, y_data):
    