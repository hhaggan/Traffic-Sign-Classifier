#Importing Libraries
import numpy as np
import math
import os
import random 
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import time
from multiprocessing import Queue
import pickle
#tensorflow
import tensorflow as tf
from tensorflow.contrib.layers import flatten

#loading data
training_file = "traffic-signs-data/train.p"
testing_file = "traffic-signs-data/test.p"

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)
    
X_train, y_train = train['features'], train['labels']
X_test, y_test = test['features'], test['labels']

#dividing the training dataset into training and validation
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

#defining some hyperparameters
n_classes = 43
epochs = 10
batch_size = 128
rate = 0.001

#defining the features and labels
x = tf.placeholder(tf.float32, (None, 32, 32, 3))
y = tf.placeholder(tf.int32, (None))
one_hot_y = tf.one_hot(y, 43)

#preprocess the data
X_train, y_train = shuffle(X_train, y_train)

#identifyign the Convolution Neural Network layers
#following the same architecture as the LeNet but with different output
def traffic_classifier(x):
    #definining the mean and the standard deviation
    mu = 0
    sigma = 0.1

    #First convolution
    '''The Input for the images should be 28*28*6 for the first convolution layer'''
    conv1_w = tf.Variable(tf.truncated_normal(shape=(5,5,3,6), mean = mu, stddev = sigma))
    conv1_b = tf.Variable(tf.zeros(6))
    conv1 = tf.nn.conv2d(x, conv1_w, strides=[1,1,1,1], padding='VALID') + conv1_b

    #Activation function
    conv1 = tf.nn.relu(conv1)
    '''The Pooling for the first convolution layer should be 14*14*6'''
    conv1 = tf.nn.max_pool(conv1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')
     
    #Second Convolution
    '''The Input for the images should be 10*10*16 for the first convolution layer'''
    conv2_w = tf.Variable(tf.truncated_normal(shape=(5,5,6,16), mean=mu, stddev=sigma))
    conv2_b = tf.Variable(tf.zeros(16))
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
    fc3_w = tf.Variable(tf.truncated_normal(shape=(84,43), mean=mu, stddev=sigma))
    fc3_b = tf.Variable(tf.zeros(43))
    logits = tf.matmul(fc2, fc3_w) + fc3_b
    '''The Final output should be 43 as per the traffic sign data data'''
    return logits

#Deep Learning details
logits = traffic_classifier(x)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_y, logits=logits)
loss_function = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate=rate)
training = optimizer.minimize(loss_function)

#parameters for the correction of the training 
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver()

#function to evaluate
def evaluate(x_data, y_data):
    num_examples = len(x_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range (0, num_examples, batch_size):
        batch_x, batch_y = x_data[offset:offset+batch_size], y_data[offset:offset+batch_size]
        accuracy = sess.run(accuracy_operation, feed_dict = {x:batch_x, y:batch_y})
        total_accuracy += (accuracy*len(batch_x))
    return total_accuracy/num_examples

#Train The Model
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    num_examples = len(X_train)

    print("Training...")
    print()
    for i in range(epochs):
        X_train, y_train = shuffle(X_train, y_train)
        for offset in range(0, num_examples, batch_size):
            end = offset + batch_size
            batch_x, batch_y = X_train[offset:end], y_train[offset:end]
            sess.run(training, feed_dict={x: batch_x, y: batch_y})
            
        validation_accuracy = evaluate(X_val, y_val)
        print("EPOCH {} ...".format(i+1))
        print("Validation Accuracy = {:.3f}".format(validation_accuracy))
        print()
        
    saver.save(sess, './lenet')
    print("Model saved")

'''with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('.'))

    test_accuracy = evaluate(X_test, y_test)
    print("Test Accuracy = {:.3f}".format(test_accuracy))'''