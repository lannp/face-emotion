import sys
import numpy as np
import matplotlib.pyplot as plt
import glob
import math
import tensorflow as tf
import cv2
import os
from sklearn.preprocessing import LabelEncoder


sess = tf.Session()
train_folder = "normalized_images"
test_folder = "test"
image_size=64
emotions = ['neutral','fear', 'happy', 'sadness', 'surprise']

def load_sequence(folder):
    X = []
    name = []
    for emotion in emotions:
        sequence_folder = glob.glob(folder+ "/" +str(emotion)+ "/*")

        for path in sequence_folder:
            img = cv2.imread(path,0)
            img = cv2.resize(img, (image_size, image_size))
            img = np.reshape(img, (image_size*image_size))
            X.append(img)
            name.append(emotion)
    return X, name

def random_batch(x_train, y_train, batch_size):
    rnd_indices = np.random.randint(0, len(x_train), batch_size)
    x_batch = x_train[rnd_indices]
    y_batch = y_train[rnd_indices]
    return x_batch, y_batch

def conv_net(x):
    with tf.variable_scope('ConvNet'):
        x = tf.reshape(x, shape=[-1, 64, 64, 1])
        conv1 = tf.layers.conv2d(x, 8, 5, activation=tf.nn.relu, padding="SAME")
        conv1 = tf.layers.max_pooling2d(conv1, 2, 2, padding="SAME")
        conv2 = tf.layers.conv2d(conv1, 16, 3, activation=tf.nn.relu, padding="SAME")
        conv2 = tf.layers.max_pooling2d(conv2, 2, 2, padding="SAME")
        conv3 = tf.layers.conv2d(conv2, 32, 5, activation=tf.nn.relu, padding="SAME")
        conv3 = tf.layers.max_pooling2d(conv3, 2, 2, padding="SAME")

        fc1 = tf.contrib.layers.flatten(conv2)
        fc1 = tf.layers.dense(fc1, 1024, activation=tf.nn.relu)
        fc1 = tf.layers.dropout(fc1, rate=dropout, training=training)
        out = tf.layers.dense(fc1, num_classes)

    return out

x_train,y_train=load_sequence(train_folder)
x_train=np.asarray(x_train)
y_train=np.asarray(y_train)
print(x_train.shape)
x_test,y_test=load_sequence(test_folder)
x_test=np.asarray(x_test) #.reshape((-1, 64, 64))
y_test=np.asarray(y_test)
print(x_test.shape)
labelencoder_X=LabelEncoder()
y_train = labelencoder_X.fit_transform(y_train)
y_test=labelencoder_X.transform(y_test)

# Training Parameters
learning_rate = 0.001
num_steps = 500
batch_size = 128
display_step = 10

# Network Parameters
num_input = 4096
num_classes = len(emotions)

dropout=0.3
# tf Graph input
x = tf.placeholder(tf.float32, [None, num_input])
y = tf.placeholder(tf.int32, [None])
training = tf.placeholder_with_default(False, shape=(), name='training')


pred = conv_net(x)

y_pred=tf.argmax(tf.nn.softmax(pred),1)
xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred, labels=y)
cost = tf.reduce_mean(xentropy)

optimizer = tf.train.AdamOptimizer()
training_op=optimizer.minimize(cost)

correct = tf.nn.in_top_k(pred, y, 1)
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
# Initializing the variables
init = tf.global_variables_initializer()
sess= tf.Session()

sess.run(init)
for step in range(1, num_steps+1):
    batch_x, batch_y = random_batch(x_train, y_train, batch_size)
    sess.run(training_op, feed_dict={x: batch_x, y: batch_y,training: True})
    if step % display_step == 0 or step == 1:
        acc = sess.run( accuracy, feed_dict={x: batch_x,y: batch_y})
        test_acc=sess.run(accuracy, feed_dict={x: x_test,y: y_test})
        print('Step:',step, ', Accuracy:',acc, ", Testing Accuracy:", test_acc)

print("Optimization Finished!")
