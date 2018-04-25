import tensorflow as tf 
import os 

def get_label(filepath):
  labels = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
  filename = str(filepath).split('\\')[-1]
  labelname = filename.split('.')[0]
  return labels.index(labelname)

def read_my_file_format(filename_and_label_tensor):
  """Consumes a single filename and label as a ' '-delimited string.

  Args:
    filename_and_label_tensor: A scalar string tensor.

  Returns:
    Two tensors: the decoded image, and the string label.
  """
  filename, label = tf.decode_csv(filename_and_label_tensor, [[""], [""]], " ")
  file_contents = tf.read_file(filename)
  label = get_label(filename)
  example = tf.image.decode_png(file_contents)
  return example, label

def get_image_dataset():
  filenames = tf.data.Dataset.list_files(f'./slices/country/*.png')
  it = filenames.make_one_shot_iterator()
  files_found = []
  image_list = []
  label_list = []
  with tf.Session() as sess:
    next_element = it.get_next()
    while True:
      try:
        filepath = sess.run(next_element)
        label = get_label(filepath)
        files_found.append(f'{str(filepath)}, {label}')
      except tf.errors.OutOfRangeError:
        break
  print(files_found)
  input_queue = tf.train.string_input_producer(files_found)
  while input_queue:
    image, label = read_my_file_format(input_queue.dequeue())
    image_list.append(image)
    label_list.append(label)
  
  print(image_list)
  print(label_list)


if __name__ == '__main__':
  get_image_dataset()