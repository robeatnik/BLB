from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os,sys
import string,re
import locale

import tensorflow as tf
import numpy as np
from numpy import genfromtxt

FLAGS = None

def input_file(df):
  print('# Importing',df,'...')
  data=genfromtxt(df, delimiter=' ')
  return data[:,0],data[:,1:]

def get_parameter():
  hw=FLAGS.h_width
  hmin=FLAGS.h_min
  hmax=FLAGS.h_max
  nSteps = FLAGS.training_steps
  bSize = FLAGS.batch_size
  nNodes1 = FLAGS.hidden_nodes_1
  fValidate = FLAGS.validate_frequency
  rLearn = FLAGS.learning_rate
  lrDecay = FLAGS.learning_rate_decay
  beta = FLAGS.regularization_weight
  tCycles = FLAGS.training_redundance
  return hmin,hw,hmax,nSteps,bSize,nNodes1,fValidate,rLearn,lrDecay,beta,tCycles

def set_label(key,data,hmin,hc,hmax):
  loc=len(data[0])
  onehot_rep={'smaller_than':[1,0], 'bigger_than':[0,1]}
  selectedData=[]
  label=[]
  for ih in range(len(key)):
    h = key[ih]
    if h >= hmin and h <= hmax:
      selectedData.append(data[ih])
      if h < hc:
        label.append(onehot_rep['smaller_than'])
      elif h > hc:
        label.append(onehot_rep['bigger_than'])
  print('#',len(selectedData),'entries between',hmin,'and',hmax)
  selectedData=np.array(selectedData)
  label=np.array(label)
  index=np.array(range(len(label)))
  np.random.shuffle(index)
  return selectedData[index],label[index]

# Not used in searching, because the data size is dynamical.
def prepare_train_data(data,nPixels,nLabels,size):
# set up to train, validate, and test data
  data_all=data[0]
  data_critical=data[1]

  size_train,size_validate,size_test = size
  if size_train+size_validate+size_test>len(data_all):
    print('Error: Not enough data')
    print(size_train,size_validate,size_test,'...',len(data_all))
    exit(0)

  x_train=np.array([i[:nPixels:] for i in data_all[:size_train]])
  y_train=np.array([i[nPixels::] for i in data_all[:size_train]])
#  print(x_train.shape)

  mark_validate=size_train+size_validate
  x_validate=np.array([i[:nPixels:] for i in data_all[size_train:mark_validate]])
  y_validate=np.array([i[nPixels::] for i in data_all[size_train:mark_validate]])

  mark_test=mark_validate+size_test
  x_test=np.array([i[:nPixels:] for i in data_all[mark_validate:mark_test]])
  y_test=np.array([i[nPixels::] for i in data_all[mark_validate:mark_test]])

  if len(data_critical)==0:
    x_critical=np.array([])
    y_critical=np.array([])
  else:
    x_critical=np.array([i[:nPixels:] for i in data_critical])
    y_critical=np.array([i[nPixels::] for i in data_critical])

  return x_train,y_train,x_validate,y_validate,x_test,y_test,x_critical,y_critical

def train(train_data,train_param):
  hmin,hc,hmax,nSteps,bSize,nNodes1,fValidate,rLearn,lrDecay,beta,tCycles=train_param
  x_train,y_train,x_validate,y_validate,x_test,y_test,x_critical,y_critical=train_data
  nPixels=len(x_train[0])
  nLabels=len(y_train[0])

  # Create the model
  x = tf.placeholder(tf.float32, [None, nPixels])
  W1 = tf.Variable(tf.random_normal([nPixels, nNodes1], stddev=0.01))
  b1 = tf.Variable(tf.zeros([nNodes1]))
  y1 = tf.nn.sigmoid(tf.matmul(x, W1) + b1)
  W2 = tf.Variable(tf.random_normal([nNodes1, nLabels], stddev=0.01))
  b2 = tf.Variable(tf.zeros([nLabels]))
  y = tf.matmul(y1, W2) + b2

  # Define loss and optimizer
  y_ = tf.placeholder(tf.float32, [None, nLabels])

  # The raw formulation of cross-entropy,
  #
  #   tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.softmax(y)),
  #                                 reduction_indices=[1]))
  #
  # can be numerically unstable.
  #
  # So here we use tf.nn.softmax_cross_entropy_with_logits on the raw
  # outputs of 'y', and then average across the batch.
  #
  # L2-regularization
  #
  l2_loss = tf.nn.l2_loss(W1) + tf.nn.l2_loss(b1) + tf.nn.l2_loss(W2) + tf.nn.l2_loss(b2)
  cross_entropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(y, y_))
  regularized_cross_entropy = cross_entropy + beta * l2_loss
  train_step = tf.train.GradientDescentOptimizer(rLearn).minimize(regularized_cross_entropy)

  # Test trained model
  correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

  print("# nstep accuracy cross_entropy L2_loss c.e./L2 beta rLearn")
  with tf.Session() as sess:
    # Train
    tf.initialize_all_variables().run()
    for nstep in range(nSteps*tCycles):
      i = nstep % nSteps
#      print(x_train.shape,nSteps,bSize,nPixels)
      batch_xs = np.reshape(x_train,(nSteps,bSize,nPixels))
      batch_ys = np.reshape(y_train,(nSteps,bSize,nLabels))
      sess.run(train_step, feed_dict={x: batch_xs[i], y_: batch_ys[i]})
      rLearn = rLearn * lrDecay

      # Validate
      if (nstep + 1) % fValidate == 0:
        temp_acc = sess.run(accuracy, feed_dict={x: x_validate,y_: y_validate})
        temp_ce = sess.run(cross_entropy, feed_dict={x: batch_xs[i], y_: batch_ys[i]})
#        temp_l2 = sess.run(l2_loss, feed_dict={x: batch_xs[i], y_: batch_ys[i]})
        temp_l2 = sess.run(l2_loss)
        print(nstep + 1,temp_acc,temp_ce,temp_l2,temp_ce/temp_l2,beta,rLearn)

    # Test
    print('# Train result:', hc, sess.run(accuracy, feed_dict={x: x_train,
                                                         y_: y_train}))
    print('# Test result:', hc, sess.run(accuracy, feed_dict={x: x_test,
                                                        y_: y_test}))

    # Analyze at criticality
    if len(x_critical) != 0:
      print('# Critical result:', hc, sess.run(accuracy, feed_dict={x: x_critical,
                                                         y_: y_critical}))

def main(_):
  # Import data
#  key,data_all = input_file('data_entanglementSpectrum')
  key,data_all = input_file('data_sparse')
  # Normalize the h parameters to units of pi
  key = 2.*key/len(key)

  # Set up parameters
  train_param = get_parameter()
  hmin,hw,hmax=train_param[0:3]
  nSteps,bSize = train_param[3:5]
  size_validate=5000
  size_test=10000

  hInterval=hw
  hLowerBound=hmin+hw
  hUpperBound=hmax-hw
  nInterval=int((hUpperBound-hLowerBound)/hInterval)

  for i in range(nInterval+1):
    hc=hLowerBound+i*hInterval
    hmin=hc-hw
    hmax=hc+hw
#    print(hmin,hw,hmax)

    # Set binary labels
    data,label = set_label(key,data_all,hmin,hc,hmax)
    FLAGS.training_steps=len(data)//bSize
    FLAGS.h_min = hmin
    FLAGS.h_width = hc
    FLAGS.h_max = hmax
    train_param = get_parameter()
    size_train = len(data)//bSize*bSize
#  size = [size_train,size_validate,size_test]

    # Training ...
#    train_data = prepare_train_data(data,nPixels,nLabels,size)
    nullArray = np.array([])
    train_data = (data[:size_train],label[:size_train],
                  data[:size_validate],label[:size_validate],
                  data[:size_test],label[:size_test],
                  nullArray,nullArray)
    train(train_data,train_param)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--h_width', type=float, default=0.1,
                      help='width of h (single side)')
  parser.add_argument('--h_min', type=float, default=0.0,
                      help='lower bound of h')
  parser.add_argument('--h_max', type=float, default=2.0,
                      help='upper bound of h')
  parser.add_argument('--training_steps', type=int, default=5000,
                      help='number of batches in training (default 5000)')
  parser.add_argument('--batch_size', type=int, default=100,
                      help='size of a batch in training (default 100)')
  parser.add_argument('--hidden_nodes_1', type=int, default=100,
                      help='number of nodes in hidden layer 1 (default 100)')
  parser.add_argument('--validate_frequency', type=int, default=100,
                      help='validate frequency in training (default 100)')
  parser.add_argument('--learning_rate', type=float, default=1.0,
                      help='initial learning rate (default 1.0)')
  parser.add_argument('--learning_rate_decay', type=float, default=1.0,
                      help='learning rate decay (default 1.0, no decay)')
  parser.add_argument('--regularization_weight', type=float, default=0.1,
                      help='regularization weight (default 0.1)')
  parser.add_argument('--training_redundance', type=int, default=1,
                      help='Cycles of data in training (default 1)')
  FLAGS = parser.parse_args()
  print('# ',' '.join(sys.argv))
  tf.app.run()
