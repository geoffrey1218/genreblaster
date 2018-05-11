from model import cnn_model_fn
import tensorflow as tf

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

def predict_input_fn():
    genre = 'blues'
    filenames = tf.constant(["./dataset_photos/slices/{}/{}.00000_{}.png".format(genre, genre, i) for i in range(0, 1280, 128)])
    labels = tf.constant([-1 for i in range(10)])

    dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
    dataset = dataset.map(_parse_function)
    dataset = dataset.batch(1)
    dataset = dataset.repeat(1)
    iterator = dataset.make_one_shot_iterator()

    features, labels = iterator.get_next()
    return features, labels

def main():
    genre_classifier = tf.estimator.Estimator(
        model_fn=cnn_model_fn, model_dir="/tmp/genre_convnet_model")
    predictions = genre_classifier.predict(input_fn=predict_input_fn)
    for p in predictions:
        genre_class = p['classes']
        probability = p['probabilities'][genre_class]
        print(GENRES[genre_class], probability)

if __name__ == '__main__':
    main()