import numpy as np
import pandas as pd

import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import metrics

import os
import sys
import shutil

from sklearn.datasets import load_breast_cancer

data = load_breast_cancer()
target = 1 - data.target

# MUST reshape target (y) to be list of lists for TensorFlow 
target = data.target.reshape(-1,1)

# Split the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(data.data, target)

ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)

tf.reset_default_graph()

# shape None == allow dynamic number of rows; X_train.shape[1] == number of features
X = tf.placeholder(dtype=tf.float32, shape=(None, X_train.shape[1]), name='inputs')
y = tf.placeholder(dtype=tf.float32, shape=(None), name='targets')

# First hidden layer
# X == features coming in, 30 == number of features == neurons
h1 = tf.layers.dense(X, 30, name='input_layer_1', activation=tf.nn.relu)

# Second hidden layer
# h1 == features coming in, 26 == number of neurons == arbitrary
h2 = tf.layers.dense(h1, 26, name='input_layer_2', activation=tf.nn.relu)

# Last/output layar always has 1 neuron for binary classification problems
y_hat = tf.layers.dense(h2, 1, name='y_hat', activation=tf.nn.sigmoid)
final_output = tf.identity(y_hat, name='classifications') 

loss = tf.losses.log_loss(y, y_hat)

# Define training operation
training_op = tf.train.AdamOptimizer(.01).minimize(loss)

# Initialize
init = tf.global_variables_initializer()

# If model already exists, delete it
try:
    if os.path.isfile('./resources/models/BreastCancer/saved_model.pb'):
        shutil.rmtree('./resources/models/BreastCancer')
except OSError as e:
    print('OSError: ', e.strerror)

with tf.Session() as sess:
    init.run()
    # Train model with 1000 epochs
    for epoch in range(1000):
        sess.run(training_op, feed_dict={X: X_train, y: y_train})
        if (epoch % 100 == 0):
            training_loss = sess.run(loss, feed_dict={X: X_train, y: y_train})
            test_loss = sess.run(loss, feed_dict={X: X_test, y: y_test})
            print('epoch:',epoch,' | training loss:', training_loss, ' | test loss:', test_loss)

    # Get the tensors needed for serving
    graph = tf.get_default_graph()
    inputs = graph.get_tensor_by_name('inputs:0')
    classifications = graph.get_tensor_by_name('classifications:0')
    
    # Create tensors info needed for serving
    model_input = tf.saved_model.utils.build_tensor_info(inputs)
    model_output = tf.saved_model.utils.build_tensor_info(classifications)

    # Build signature definition needed for serving
    signature_definition = tf.saved_model.signature_def_utils.build_signature_def(
        inputs={'inputs': model_input},
        outputs={'outputs': model_output},
        method_name= tf.saved_model.signature_constants.CLASSIFY_METHOD_NAME)

    builder = tf.saved_model.builder.SavedModelBuilder('./resources/models/BreastCancer')

    builder.add_meta_graph_and_variables(
        sess, [tf.saved_model.tag_constants.SERVING],
        signature_def_map={
            tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                signature_definition
        })
    
    # Save the model so it can be served with a TF model server
    builder.save()
    
    # Use model to classify test data
    classifications_on_test_data = sess.run(y_hat, feed_dict={X: X_test})
    
    # Check model accuracy
    classes = (classifications_on_test_data > .5).astype(int)
    
    print('\nAccuracy Score:',metrics.accuracy_score(y_test, classes))

