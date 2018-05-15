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

GENRES_AB = {
    'blues': 'B',
    'classical': 'CL',
    'country': 'CO',
    'disco': 'D',
    'hiphop': 'H',
    'jazz': 'J',
    'metal': 'M',
    'pop': 'P',
    'reggae': 'RE',
    'rock': 'RO',
    'indeterminate': 'I'
}

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

def get_predictions_from_slices(slices, tmp_dir='/tmp'):
    input_fn = get_input_fn(slices)
    model_dir = os.path.join(tmp_dir, 'genre_convnet_model')
    genre_classifier = tf.estimator.Estimator(
        model_fn=cnn_model_fn, model_dir=model_dir)
    predictions = genre_classifier.predict(input_fn=input_fn)
    classes = [GENRES[p['classes']] if (p['probabilities'][p['classes']] > 0.5) else 'indeterminate' for p in predictions]
    return classes

def print_predicted_genre_and_breakdown(classes):
    class_counts = Counter(classes)
    most_common_genre, most_common_count  = class_counts.most_common(1)[0]
    num_slices = len(classes)
    if most_common_count > num_slices / 2: # one genre makes up more than 50%
        print('Predicted genre:', most_common_genre.upper())
    else:
        print('Genre indeterminate')

    print('Genre breakdown:')
    for genre, count in class_counts.most_common():
        print('\t{}: {}/{}'.format(genre, count, num_slices))
    
def predict_from_original_dataset(genre_name):
    genre_dir = 'dataset_photos/slices/{}'.format(genre_name)
    for subdir, dirs, files in os.walk(genre_dir):
        slices = [os.path.join(subdir, f) for f in files]
        predictions = get_predictions_from_slices(slices)
        print_predicted_genre_and_breakdown(predictions)

def main():
    tmp_dir = os.path.join(os.sep, 'tmp')
    filename = input('Enter the name of a sound file in the cwd: ')
    print('Creating spectrogram...')
    cwd = os.getcwd()
    filepath = os.path.join(cwd, filename)
    spectrogram_file = os.path.join(tmp_dir, 'spectrogram.png')
    make_single_spectrogram(filepath, spectrogram_file)

    print('Slicing spectrogram...')
    slice_dir = os.path.join(tmp_dir, 'slices')
    try:
        os.mkdir(slice_dir)
    except FileExistsError:
        pass
    slices = slice_image(spectrogram_file, 'spectrogram.png', slice_dir)

    print('Running model...')
    classes = get_predictions_from_slices(slices, tmp_dir)

    print('Cleaning up spectrogram and slices...')
    os.remove(spectrogram_file) 
    for s in slices:
        os.remove(s)

    print_predicted_genre_and_breakdown(classes)
    timeline = ''
    for i, genre in enumerate(classes):
        if i % 10 == 0:
            timeline += '({}s) '.format(i*3)
        timeline += '{} '.format(GENRES_AB[genre])
    print('Timeline:', timeline)

if __name__ == '__main__':
    main()