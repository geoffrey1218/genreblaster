import tensorflow as tf 
import numpy as np 
import matplotlib.pyplot as plt
import os 
import math

def process_tfrecords():
  '''
  Help from http://www.machinelearninguru.com/deep_learning/tensorflow/basics/tfrecord/tfrecord.html
  '''
  # You can call this function in a loop to train the model, 100 images at a time
  def training_step(i, update_test_data, update_train_data):

    # training on batches of 100 images with 100 labels
    batch_X, batch_Y = sess.run([images, labels])

    # compute training values for visualisation
    if update_train_data:
      a, c, im, w, b, l = sess.run([accuracy, cross_entropy, I, allweights, allbiases,lr],
                                feed_dict={X: batch_X, Y_: batch_Y, step: i})
      print(str(i) + ": accuracy:" + str(a) + " loss: " + str(c) + " (lr:" + str(l) + ")")

    # compute test values for visualisation
    if update_test_data:
      a, c, im = sess.run([accuracy, cross_entropy, It],
                          feed_dict={X: mnist.test.images, Y_: mnist.test.labels})
      print(str(i) + ": ********* epoch " + str(i*100//mnist.train.images.shape[0]+1) + " ********* test accuracy:" + str(a) + " test loss: " + str(c))

    # the backpropagation training step
    sess.run(train_step, {X: batch_X, Y_: batch_Y, step: i})
    
  features = {
    'image/encoded': tf.FixedLenFeature([], tf.string),
    'image/format': tf.FixedLenFeature([], tf.string),
    'image/class/label': tf.FixedLenFeature([], tf.int64),
    'image/height': tf.FixedLenFeature([], tf.int64),
    'image/width': tf.FixedLenFeature([], tf.int64),
  }

  # pass filenames to queue
  filename_queue = tf.train.string_input_producer(['dataset_photos/genres_validation_00000-of-00002.tfrecord'], num_epochs=1)

  # read next record
  reader = tf.TFRecordReader()
  _, serialized_example = reader.read(filename_queue)

  # decode record
  record_features = tf.parse_single_example(serialized_example, features=features)

  # convert feature data
  image = tf.image.decode_png(record_features['image/encoded'], 1, tf.uint8)
  label = tf.cast(record_features['image/class/label'], tf.int32)
  height = tf.cast(record_features['image/height'], tf.int32)
  width = tf.cast(record_features['image/width'], tf.int32)

  image = tf.reshape(image, [128, 128, 1])

  # create random batches
  images, labels = tf.train.shuffle_batch(
    [image, label], 
    batch_size=10, 
    capacity=30, 
    num_threads=1, 
    min_after_dequeue=10
  )

  # input X: 128x128 grayscale images, the first dimension (None) will index the images in the mini-batch
  X = tf.placeholder(tf.float32, [None, 128, 128, 1])
  # correct answers will go here
  Y_ = tf.placeholder(tf.float32, [None, 10])
  # step for variable learning rate
  step = tf.placeholder(tf.int32)

  # four convolutional layers with their channel counts, and a
  # fully connected layer (tha last layer has 10 softmax neurons)
  K = 64   # first convolutional layer output depth
  L = 128  # second convolutional layer output depth
  M = 256  # third convolutional layer
  N = 512  # fourth convolutional layer
  P = 1024  # fully connected layer

  W1 = tf.Variable(tf.truncated_normal([2, 2, 1, K], stddev=0.1))  # 2x2 patch, 1 input channel, K output channels
  B1 = tf.Variable(tf.ones([K])/10)
  W2 = tf.Variable(tf.truncated_normal([2, 2, K, L], stddev=0.1))
  B2 = tf.Variable(tf.ones([L])/10)
  W3 = tf.Variable(tf.truncated_normal([2, 2, L, M], stddev=0.1))
  B3 = tf.Variable(tf.ones([M])/10)
  W4 = tf.Variable(tf.truncated_normal([2, 2, M, N], stddev=0.1))
  B4 = tf.Variable(tf.ones([N])/10)    

  W5 = tf.Variable(tf.truncated_normal([16 * 16 * N, P], stddev=0.1))
  B5 = tf.Variable(tf.ones([P])/10)
  W6 = tf.Variable(tf.truncated_normal([P, 10], stddev=0.1))
  B6 = tf.Variable(tf.ones([10])/10)

  # The model
  stride = 2  # output is 28x28
  Y1 = tf.nn.relu(tf.nn.conv2d(X, W1, strides=[1, stride, stride, 1], padding='SAME') + B1)
  stride = 2  # output is 14x14
  Y2 = tf.nn.relu(tf.nn.conv2d(Y1, W2, strides=[1, stride, stride, 1], padding='SAME') + B2)
  stride = 2  # output is 7x7
  Y3 = tf.nn.relu(tf.nn.conv2d(Y2, W3, strides=[1, stride, stride, 1], padding='SAME') + B3)
  stride = 2  # output is 7x7
  Y4 = tf.nn.relu(tf.nn.conv2d(Y3, W4, strides=[1, stride, stride, 1], padding='SAME') + B4)

  # reshape the output from the fourth convolution for the fully connected layer
  YY = tf.reshape(Y4, shape=[-1, 16 * 16 * N])

  Y5 = tf.nn.relu(tf.matmul(YY, W5) + B5)
  Ylogits = tf.matmul(Y5, W6) + B6
  Y = tf.nn.softmax(Ylogits)

  # cross-entropy loss function (= -sum(Y_i * log(Yi)) ), normalised for batches of 100  images
  # TensorFlow provides the softmax_cross_entropy_with_logits function to avoid numerical stability
  # problems with log(0) which is NaN
  cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=Ylogits, labels=Y_)
  cross_entropy = tf.reduce_mean(cross_entropy)*100

  # accuracy of the trained model, between 0 (worst) and 1 (best)
  correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

  lr = 0.0001 +  tf.train.exponential_decay(0.003, step, 2000, 1/math.e)
  train_step = tf.train.AdamOptimizer(lr).minimize(cross_entropy)



  # initialize global and local vars
  init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
  sess = tf.Session()
  sess.run(init_op) 
  
  for i in range(200):
    update_test_data = (200 % 20 == 0)
    training_step(i, update_test_data, True)
  
  sess.close() 




if __name__ == '__main__':
  process_tfrecords()