import tensorflow as tf 
import numpy as np 
import matplotlib.pyplot as plt
import os 

def process_tfrecords():
  '''
  Help from http://www.machinelearninguru.com/deep_learning/tensorflow/basics/tfrecord/tfrecord.html
  '''
  with tf.Session() as sess:
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
    image = tf.decode_raw(record_features['image/encoded'], tf.float32)
    label = tf.cast(record_features['image/class/label'], tf.int32)
    height = tf.cast(record_features['image/height'], tf.int32)
    width = tf.cast(record_features['image/width'], tf.int32)

    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sess.run(init_op) 
    height_val = height.eval() 
    width_val = width.eval()
    sess.close()

    print(height_val, width_val)
    image = tf.reshape(image, [150, 129, 3])

    # create random batches
    images, labels = tf.train.shuffle_batch(
      [image, label], 
      batch_size=10, 
      capacity=30, 
      num_threads=1, 
      min_after_dequeue=10
    )

    # initialize global and local vars
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sess.run(init_op) 

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    for batch_index in range(5):
      img, lbl = sess.run([images, labels])
      img = img.astype(np.uint8)

      for j in range(6):
        plt.subplot(2, 3, j+1)
        plt.imshow(img[j, ...])
        plt.title(lbl[j])
      
      # stop threads
      coord.request_stop()

      # wait for threads to stop
      coord.join(threads)
      sess.close() 


if __name__ == '__main__':
  process_tfrecords()