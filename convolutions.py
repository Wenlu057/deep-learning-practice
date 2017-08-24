
# coding: utf-8

# In[1]:

# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
from __future__ import print_function
import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle
from six.moves import range


# In[2]:

pickle_file = 'notMNIST.pickle'

with open(pickle_file, 'rb') as f:
  save = pickle.load(f)
  train_dataset = save['train_dataset']
  train_labels = save['train_labels']
  valid_dataset = save['valid_dataset']
  valid_labels = save['valid_labels']
  test_dataset = save['test_dataset']
  test_labels = save['test_labels']
  del save  # hint to help gc free up memory
  print('Training set', train_dataset.shape, train_labels.shape)
  print('Validation set', valid_dataset.shape, valid_labels.shape)
  print('Test set', test_dataset.shape, test_labels.shape)


# In[3]:

image_size = 28
num_labels = 10
num_channels = 1

import numpy as np

def reformat(dataset, labels):
    dataset = dataset.reshape(-1, image_size, image_size, num_channels).astype(np.float32)
    labels = (labels[:, None] == np.arange(num_labels)).astype(np.float32)
    return dataset, labels
train_dataset, train_labels = reformat(train_dataset, train_labels)
valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
test_dataset, test_labels = reformat(test_dataset, test_labels)

print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)


# In[4]:

def accuracy(predictions, labels):
    return (100 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))/predictions.shape[0])


# In[25]:

batch_size = 16
patch_size = 5
depth = 16
num_hidden = 64

graph = tf.Graph()

with graph.as_default():
    
    #Input data.
    tf_train_dataset = tf.placeholder(tf.float32, shape = (batch_size, image_size, image_size, num_channels))
    tf_train_labels = tf.placeholder(tf.float32, shape = (batch_size, num_labels))
    tf_valid_dataset = tf.constant(valid_dataset)
    tf_test_dataset = tf.constant(test_dataset)
    
    #Variables.
    layer1_weights = tf.Variable(tf.truncated_normal([patch_size, patch_size, num_channels, depth], stddev = 0.1))
    layer1_biases = tf.Variable(tf.zeros([depth]))
    layer2_weights = tf.Variable(tf.truncated_normal([patch_size, patch_size, depth, depth], stddev = 0.1))
    layer2_biases = tf.Variable(tf.constant(1.0, shape = [depth]))
    layer3_weights = tf.Variable(tf.truncated_normal([image_size//4 *  image_size//4*depth, num_hidden],stddev = 0.1))
    layer3_biases = tf.Variable(tf.constant(1.0, shape = [num_hidden]))
    layer4_weights = tf. Variable(tf.truncated_normal([num_hidden, num_labels], stddev=0.1))
    layer4_biases = tf.Variable(tf.constant(1.0, shape =[num_labels]))
    
    #Model.
    def model(data):
        conv = tf.nn.conv2d(data, layer1_weights, [1,2,2,1], padding = 'SAME')
        hidden = tf.nn.relu(conv+layer1_biases)
        pool = tf.nn.max_pool(
           hidden, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        conv = tf.nn. conv2d(hidden, layer2_weights, [1,2,2,1], padding = 'SAME')
        hidden = tf.nn.relu(conv+layer2_biases)
        shape = hidden.get_shape().as_list()
        reshape = tf.reshape(hidden, [shape[0], shape[1] * shape[2] * shape[3]])
        hidden = tf.nn.relu(tf.matmul(reshape, layer3_weights) + layer3_biases)
        return tf.matmul(hidden, layer4_weights) + layer4_biases
    
    #Training computation.
    logits = model(tf_train_dataset)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=logits))
    
    #Optimizer
    optimizer = tf.train.GradientDescentOptimizer(0.05).minimize(loss)
    
    # Predictions for the training, validation, and test data.
    train_prediction = tf.nn.softmax(logits)
    valid_prediction = tf.nn.softmax(model(tf_valid_dataset))
    test_prediction = tf.nn.softmax(model(tf_test_dataset))
    


# In[26]:

num_steps = 1001

with tf.Session(graph=graph) as session:
  tf.global_variables_initializer().run()
  print('Initialized')
  for step in range(num_steps):
    offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
    batch_data = train_dataset[offset:(offset + batch_size), :, :, :]
    batch_labels = train_labels[offset:(offset + batch_size), :]
    feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
    _, l, predictions = session.run(
      [optimizer, loss, train_prediction], feed_dict=feed_dict)
    if (step % 50 == 0):
      print('Minibatch loss at step %d: %f' % (step, l))
      print('Minibatch accuracy: %.1f%%' % accuracy(predictions, batch_labels))
      print('Validation accuracy: %.1f%%' % accuracy(
        valid_prediction.eval(), valid_labels))
  print('Test accuracy: %.1f%%' % accuracy(test_prediction.eval(), test_labels))


# In[33]:


# Problem 1
# The convolutional model above uses convolutions with stride 2 to reduce the dimensionality. 
# Replace the strides by a max pooling operation (nn.max_pool()) of stride 2 and kernel size 2.

batch_size = 16
patch_size = 5
depth = 16
num_hidden = 64

graph = tf.Graph()

with graph.as_default():
    
    #Input data.
    tf_train_dataset = tf.placeholder(tf.float32, shape = (batch_size, image_size, image_size, num_channels))
    tf_train_labels = tf.placeholder(tf.float32, shape = (batch_size, num_labels))
    tf_valid_dataset = tf.constant(valid_dataset)
    tf_test_dataset = tf.constant(test_dataset)
    
    #Variables.
    layer1_weights = tf.Variable(tf.truncated_normal([patch_size, patch_size, num_channels, depth], stddev = 0.1))
    layer1_biases = tf.Variable(tf.zeros([depth]))
    layer2_weights = tf.Variable(tf.truncated_normal([patch_size, patch_size, depth, depth], stddev = 0.1))
    layer2_biases = tf.Variable(tf.constant(1.0, shape = [depth]))
    layer3_weights = tf.Variable(tf.truncated_normal([image_size//4 *  image_size//4*depth, num_hidden],stddev = 0.1))
    layer3_biases = tf.Variable(tf.constant(1.0, shape = [num_hidden]))
    layer4_weights = tf. Variable(tf.truncated_normal([num_hidden, num_labels], stddev=0.1))
    layer4_biases = tf.Variable(tf.constant(1.0, shape =[num_labels]))
    
    #Model.
    def model(data):
        conv = tf.nn.conv2d(data, layer1_weights, [1,1,1,1], padding = 'SAME')
        hidden = tf.nn.relu(conv+layer1_biases)
        pool = tf.nn.max_pool(
           hidden, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        conv = tf.nn. conv2d(pool, layer2_weights, [1,1,1,1], padding = 'SAME')
        hidden = tf.nn.relu(conv+layer2_biases)
        pool = tf.nn.max_pool(
           hidden, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        shape = pool.get_shape().as_list()
        reshape = tf.reshape(pool, [shape[0], shape[1] * shape[2] * shape[3]])
        hidden = tf.nn.relu(tf.matmul(reshape, layer3_weights) + layer3_biases)
        return tf.matmul(hidden, layer4_weights) + layer4_biases
    
    #Training computation.
    logits = model(tf_train_dataset)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=logits))
    
    #Optimizer
    optimizer = tf.train.GradientDescentOptimizer(0.05).minimize(loss)
    
    # Predictions for the training, validation, and test data.
    train_prediction = tf.nn.softmax(logits)
    valid_prediction = tf.nn.softmax(model(tf_valid_dataset))
    test_prediction = tf.nn.softmax(model(tf_test_dataset))
    


# In[32]:

num_steps = 1001

with tf.Session(graph=graph) as session:
  tf.global_variables_initializer().run()
  print('Initialized')
  for step in range(num_steps):
    offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
    batch_data = train_dataset[offset:(offset + batch_size), :, :, :]
    batch_labels = train_labels[offset:(offset + batch_size), :]
    feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
    _, l, predictions = session.run(
      [optimizer, loss, train_prediction], feed_dict=feed_dict)
    if (step % 50 == 0):
      print('Minibatch loss at step %d: %f' % (step, l))
      print('Minibatch accuracy: %.1f%%' % accuracy(predictions, batch_labels))
      print('Validation accuracy: %.1f%%' % accuracy(
        valid_prediction.eval(), valid_labels))
  print('Test accuracy: %.1f%%' % accuracy(test_prediction.eval(), test_labels))


# In[70]:

batch_size = 16
patch_size = 5
depth1 = 6
depth2 = 16
num_hidden1 = 120
num_hidden2 = 84

graph = tf.Graph()

with graph.as_default():
    
    #Input data.
    tf_train_dataset = tf.placeholder(tf.float32, shape = (batch_size, image_size, image_size, num_channels))
    tf_train_labels = tf.placeholder(tf.float32, shape = (batch_size, num_labels))
    tf_valid_dataset = tf.constant(valid_dataset)
    tf_test_dataset = tf.constant(test_dataset)
    global_step = tf.Variable(0)
    
    #Variables.
    layer1_weights = tf.Variable(tf.truncated_normal([patch_size, patch_size, num_channels, depth1], stddev = 0.1))
    layer1_biases = tf.Variable(tf.zeros([depth1]))
    layer2_weights = tf.Variable(tf.truncated_normal([patch_size, patch_size, depth1, depth2], stddev = 0.1))
    layer2_biases = tf.Variable(tf.constant(1.0, shape = [depth2]))
    layer3_weights = tf.Variable(tf.truncated_normal([image_size//7 *  image_size//7*depth2, num_hidden1],stddev = 0.1))
    layer3_biases = tf.Variable(tf.constant(1.0, shape = [num_hidden1]))
    layer4_weights = tf.Variable(tf.truncated_normal([num_hidden1, num_hidden2],stddev = 0.1))
    layer4_biases = tf.Variable(tf.constant(1.0, shape = [num_hidden2]))
    layer5_weights = tf.Variable(tf.truncated_normal([num_hidden2, num_labels], stddev=0.1))
    layer5_biases = tf.Variable(tf.constant(1.0, shape =[num_labels]))
    
    #Model.
    def model(data):
        conv = tf.nn.conv2d(data, layer1_weights, [1,1,1,1], padding = 'VALID')
        hidden = tf.nn.relu(conv+layer1_biases) #output 24*24 , depth :16
        pool = tf.nn.avg_pool(
           hidden, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
        conv = tf.nn. conv2d(pool, layer2_weights, [1,1,1,1], padding = 'VALID')
        hidden = tf.nn.relu(conv+layer2_biases)
        pool = tf.nn.avg_pool(
           hidden, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
        shape = pool.get_shape().as_list()
        reshape = tf.reshape(pool, [shape[0], shape[1] * shape[2] * shape[3]])
        hidden = tf.nn.relu(tf.matmul(reshape, layer3_weights) + layer3_biases)
        hidden = tf.nn.relu(tf.matmul(hidden, layer4_weights) + layer4_biases)
        return tf.matmul(hidden, layer5_weights) + layer5_biases
    
    #Training computation.
    logits = model(tf_train_dataset)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=logits))
    
    #Optimizer
    learning_rate = tf.train.exponential_decay(0.05, global_step, 1000, 0.85, staircase = True)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step = global_step)
    
    # Predictions for the training, validation, and test data.
    train_prediction = tf.nn.softmax(logits)
    valid_prediction = tf.nn.softmax(model(tf_valid_dataset))
    test_prediction = tf.nn.softmax(model(tf_test_dataset))
    


# In[72]:

num_steps = 5001

with tf.Session(graph=graph) as session:
  tf.global_variables_initializer().run()
  print('Initialized')
  for step in range(num_steps):
    offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
    batch_data = train_dataset[offset:(offset + batch_size), :, :, :]
    batch_labels = train_labels[offset:(offset + batch_size), :]
    feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
    _, l, predictions = session.run(
      [optimizer, loss, train_prediction], feed_dict=feed_dict)
    if (step % 50 == 0):
      print('Minibatch loss at step %d: %f' % (step, l))
      print('Minibatch accuracy: %.1f%%' % accuracy(predictions, batch_labels))
      print('Validation accuracy: %.1f%%' % accuracy(
        valid_prediction.eval(), valid_labels))
  print('Test accuracy: %.1f%%' % accuracy(test_prediction.eval(), test_labels))

