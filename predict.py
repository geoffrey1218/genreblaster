from model import cnn_model_fn
from spectrogram import make_single_spectrogram
from slices import slice_image

import tensorflow as tf
import os
from collections import Counter

GENRES = [
    'blues',
    'classical',
    'country',
    'disco',
    'hiphop',
    'jazz',
    'metal',
    'pop',
    'reggae',
    'rock'
]

def _parse_function(filename, label):
    image_string = tf.read_file(filename)
    image = tf.image.decode_png(image_string)
    image = tf.cast(image, tf.float32)
    image_resized = tf.reshape(image, [128, 128, 1])
    return image_resized, label

def get_input_fn(filepaths):
    def predict_input_fn():
        filenames = tf.constant(filepaths)
        labels = tf.constant([-1 for i in range(len(filepaths))])

        dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
        dataset = dataset.map(_parse_function)
        dataset = dataset.batch(1)
        dataset = dataset.repeat(1)
        iterator = dataset.make_one_shot_iterator()

        features, labels = iterator.get_next()
        return features, labels
    return predict_input_fn

def main():
    filename = input('Enter the name of a sound file in the cwd: ')
    print('Creating spectrogram...')
    cwd = os.getcwd()
    filepath = os.path.join(cwd, filename)
    make_single_spectrogram(filepath, '/tmp/spectrogram.png')

    print('Slicing spectrogram...')
    slice_dir = os.path.join(os.sep, 'tmp', 'slices')
    try:
        os.mkdir(slice_dir)
    except FileExistsError:
        pass
    slices = slice_image('/tmp/spectrogram.png', 'spectrogram.png', slice_dir)
    input_fn = get_input_fn(slices)

    print('Running model...')
    genre_classifier = tf.estimator.Estimator(
        model_fn=cnn_model_fn, model_dir="/tmp/genre_convnet_model")
    predictions = genre_classifier.predict(input_fn=input_fn)
    classes = [GENRES[p['classes']] if (p['probabilities'][p['classes']] > 0.5) else 'indeterminate' for p in predictions]
    print(Counter(classes))

if __name__ == '__main__':
    main()