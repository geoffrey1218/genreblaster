from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports
import numpy as np
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)

def cnn_model_fn(features, labels, mode):
    # Input Layer
    input_layer = tf.reshape(features, [-1, 128, 128, 1])

    # Convolutional Layer #1
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=64,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)

    # Pooling Layer #1
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

    # Convolutional Layer #2 and Pooling Layer #2
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=128,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

    conv3 = tf.layers.conv2d(
        inputs=pool2,
        filters=256,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)
    pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2], strides=2)

    conv4 = tf.layers.conv2d(
        inputs=pool3,
        filters=512,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)
    pool4 = tf.layers.max_pooling2d(inputs=conv4, pool_size=[2, 2], strides=2)

    pool4_flat = tf.reshape(pool4, [-1, 8 * 8 * 512])
    dense = tf.layers.dense(
        inputs=pool4_flat, units=1024, activation=tf.nn.relu)
    dropout = tf.layers.dropout(
        inputs=dense, rate=0.5, training=mode == tf.estimator.ModeKeys.TRAIN)

    # Logits Layer
    logits = tf.layers.dense(inputs=dropout, units=10)

    predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
        "classes": tf.argmax(input=logits, axis=1),
        # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
        # `logging_hook`.
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate Loss (for both TRAIN and EVAL modes)
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
            labels=labels, predictions=predictions["classes"])}
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def decode(sample):
    features = {
        'image/encoded': tf.FixedLenFeature([], tf.string),
        'image/format': tf.FixedLenFeature([], tf.string),
        'image/class/label': tf.FixedLenFeature([], tf.int64),
        'image/height': tf.FixedLenFeature([], tf.int64),
        'image/width': tf.FixedLenFeature([], tf.int64),
    }
    record_features = tf.parse_single_example(sample, features=features)

    # convert feature data
    image = tf.image.decode_png(record_features['image/encoded'], 1, tf.uint8)
    image = tf.cast(image, tf.float32)
    label = tf.cast(record_features['image/class/label'], tf.int32)

    image_resized = tf.reshape(image, [128, 128, 1])
    return image_resized, label


def train_input_fn():
    BATCH_SIZE = 20
    dataset = tf.data.TFRecordDataset(
        [f'dataset_photos/genres_train_0000{i}-of-00002.tfrecord' for i in range(2)])
    dataset = dataset.map(decode)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.repeat()
    iterator = dataset.make_one_shot_iterator()

    features, labels = iterator.get_next()
    return features, labels


def eval_input_fn():
    BATCH_SIZE = 20
    dataset = tf.data.TFRecordDataset(
        [f'dataset_photos/genres_validation_0000{i}-of-00002.tfrecord' for i in range(2)])
    dataset = dataset.map(decode)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.repeat(1)
    iterator = dataset.make_one_shot_iterator()

    features, labels = iterator.get_next()
    return features, labels


def main():
    # Create the Estimator
    genre_classifier = tf.estimator.Estimator(
        model_fn=cnn_model_fn, model_dir="/tmp/genre_convnet_model")

    # Set up logging for predictions
    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=50)
    
    # Train the model
    genre_classifier.train(
        input_fn=train_input_fn,
        max_steps=1500,
        hooks=[logging_hook])

    # Evaluate the model and print results
    eval_results = genre_classifier.evaluate(input_fn=eval_input_fn)
    print(eval_results)


if __name__ == "__main__":
    main()
