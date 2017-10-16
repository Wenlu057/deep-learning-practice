
# coding: utf-8

# In[74]:

from __future__ import print_function
import numpy as np
import tensorflow as tf
import os
import sys
from six.moves.urllib.request import urlretrieve
from shutil import copyfile
import IPython.core.debugger
import tarfile
import h5py
import random
from PIL import Image
from six.moves import cPickle as pickle
import time
from datetime import datetime


# In[2]:

dbg = IPython.core.debugger.Pdb()


# In[3]:

url = 'http://ufldl.stanford.edu/housenumbers/'
TRAIN = 'train.tar.gz'
TEST = 'test.tar.gz'
EXTRA = 'extra.tar.gz'
def maybe_download(filename, work_dir):
    """Download the data from source_url, unless it's already here
    
        Args:
            filename: string, name of the file in the directory.
            work_dir: string, path to working directory.
        Returns:
            Path to resulting file.
            
    """
    if not os.path.exists(work_dir):
        os.mkdir(work_dir)
    file_path = os.path.join(work_dir,filename)
    if not os.path.exists(file_path):
        temp_file, _ = urlretrieve(url+filename)
        copyfile(temp_file, file_path)

    print('Found and verified:', filename, os.stat(file_path).st_size, 'bytes')
    return file_path


# In[4]:

work_dir = 'SVHN/'
train_path = maybe_download(TRAIN,work_dir)
test_path = maybe_download(TEST,work_dir)
extra_path = maybe_download(EXTRA, work_dir)


# In[5]:

def maybe_extract(filename):
    root = os.path.splitext(os.path.splitext(filename)[0])[0]
    if os.path.isdir(root):
        print('%s already present - Skipping extraction of %s' %(root, filename))
    else:
        print('Extracting data for %s. This may take a while. Please wait.' %root)
        tar = tarfile.open(filename)
        sys.stdout.flush()
        tar.extractall(work_dir)
        print('Extracting done!')
        tar.close
    return root
train_dir = maybe_extract(train_path)
test_dir = maybe_extract(test_path)


# In[6]:

class ExampleReader(object):
    def __init__(self, path_to_image_files):
        self._path_to_image_files = path_to_image_files
        self._num_examples = len(self._path_to_image_files)
        self._example_pointer = 0

    @staticmethod
    def _get_attrs(digit_struct_mat_file, index):
        """
        Returns a dictionary which contains keys: label, left, top, width and height, each key has multiple values.
        """
        attrs = {}
        f = digit_struct_mat_file
        item = f['digitStruct']['bbox'][index].item()
        for key in ['label', 'left', 'top', 'width', 'height']:
            attr = f[item][key]
            values = [f[attr.value[i].item()].value[0][0]
                      for i in range(len(attr))] if len(attr) > 1 else [attr.value[0][0]]
            attrs[key] = values
        return attrs

    @staticmethod
    def _preprocess(image, bbox_left, bbox_top, bbox_width, bbox_height):
        cropped_left, cropped_top, cropped_width, cropped_height = (int(round(bbox_left - 0.15 * bbox_width)),
                                                                    int(round(bbox_top - 0.15 * bbox_height)),
                                                                    int(round(bbox_width * 1.3)),
                                                                    int(round(bbox_height * 1.3)))
        image = image.crop([cropped_left, cropped_top, cropped_left + cropped_width, cropped_top + cropped_height])
        image = image.resize([64, 64])
        return image

    @staticmethod
    def _int64_feature(value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    @staticmethod
    def _float_feature(value):
        return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

    @staticmethod
    def _bytes_feature(value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def read_and_convert(self, digit_struct_mat_file):
        """
        Read and convert to example, returns None if no data is available.
        """

        if self._example_pointer == self._num_examples:
            return None
        path_to_image_file = self._path_to_image_files[self._example_pointer]
        index = int(path_to_image_file.split('/')[-1].split('.')[0]) - 1
        self._example_pointer += 1

        attrs = ExampleReader._get_attrs(digit_struct_mat_file, index)
        label_of_digits = attrs['label']
        length = len(label_of_digits)
        if length > 5:
            # skip this example
            return self.read_and_convert(digit_struct_mat_file)

        digits = [10, 10, 10, 10, 10]   # digit 10 represents no digit
        for idx, label_of_digit in enumerate(label_of_digits):
            digits[idx] = int(label_of_digit if label_of_digit != 10 else 0)    # label 10 is essentially digit zero

        attrs_left, attrs_top, attrs_width, attrs_height = map(lambda x: [int(i) for i in x], [attrs['left'], attrs['top'], attrs['width'], attrs['height']])
        min_left, min_top, max_right, max_bottom = (min(attrs_left),
                                                    min(attrs_top),
                                                    max(map(lambda x, y: x + y, attrs_left, attrs_width)),
                                                    max(map(lambda x, y: x + y, attrs_top, attrs_height)))
        center_x, center_y, max_side = ((min_left + max_right) / 2.0,
                                        (min_top + max_bottom) / 2.0,
                                        max(max_right - min_left, max_bottom - min_top))
        bbox_left, bbox_top, bbox_width, bbox_height = (center_x - max_side / 2.0,
                                                        center_y - max_side / 2.0,
                                                        max_side,
                                                        max_side)
        image = np.array(ExampleReader._preprocess(Image.open(path_to_image_file), bbox_left, bbox_top, bbox_width, bbox_height)).tobytes()
#         dbg.set_trace()
        example = tf.train.Example(features=tf.train.Features(feature={
            'image': ExampleReader._bytes_feature(image),
            'length': ExampleReader._int64_feature(length),
            'digits': tf.train.Feature(int64_list=tf.train.Int64List(value=digits))
        }))
        return example


# In[7]:

def convert_to_tfrecords(dataset_and_digit_struct, tfrecords, writer_callback):
    num_examples = []
    writers = []
    
    for record in tfrecords:
        num_examples.append(0)
        writers.append(tf.python_io.TFRecordWriter(record))
    for dataset, digit_struct in dataset_and_digit_struct:
        image_files = tf.gfile.Glob(os.path.join(dataset, '*.png'))
        total_files = len(image_files)
        print('%d files found in %s' %(total_files,dataset))
#         dbg.set_trace()
        with h5py.File(digit_struct,'r') as f:
            example_reader = ExampleReader(image_files)
            for index, image_file in enumerate(image_files):
                if index%10 == 0:
                    print('(%d/%d) processing %s' % (index+1, total_files, image_file))
                example = example_reader.read_and_convert(f)
                if example is None:
                    break
                idx = writer_callback(tfrecords)
                writers[idx].write(example.SerializeToString())
                num_examples[idx] += 1
    for writer in writers:
        writer.close()
    return num_examples


# In[8]:

train_struct_mat = os.path.join(train_dir, 'digitStruct.mat')
test_struct_mat = os.path.join(test_dir, 'digitStruct.mat')
train_tfrecords = os.path.join(work_dir, 'train.tfrecords')
valid_tfrecords = os.path.join(work_dir, 'valid.tfrecords')
test_tfrecords = os.path.join(work_dir, 'test.tfrecords')
# First assume the tfrecords is not existed yet.
if not os.path.exists(train_tfrecords):
    print ('Processing training and validation data...')
    [num_train_examples, num_val_examples] = convert_to_tfrecords([(train_dir,train_struct_mat)],[train_tfrecords, valid_tfrecords], lambda paths:0 if random.random()>0.1 else 1)
else: 
    print('The file %s already exists'% train_tfrecords)
    
if not os.path.exists(test_tfrecords):
    print('Processing testing data...')
    [num_test_examples] = convert_to_tfrecords([(test_dir,test_struct_mat)],[test_tfrecords],lambda paths: 0)
else:
    print('The file %s already exists'% test_tfrecords)
    


# In[9]:

metadata = {}
if not os.path.exists('SVHN/metadata.pickle'):
    metadata = {
        'num_train_examples' : num_train_examples,
        'num_val_examples' : num_val_examples,
        'num_test_examples' : num_test_examples
    }
# root = '.'

def maybe_pickle(dataset,dest_dir,filename, force= False):
    """
    pickle the dataset as the pickle file
    
    Args:
         dataset: the dataset need to pickle.
         dest_dir: path where you save the pickle files.
         filename: str represents the name of the dataset
    Return:
         dataset_names: the name of the pickle file
    """

    file_path = os.path.join(dest_dir, filename) + '.pickle'
    if os.path.exists(file_path) and not force:
        print('%s already present - Skipping pickling.' % filename)
    else:
        print('Pickling %s.' % file_path)
        try:
            with open(file_path, 'wb') as f:
                pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            print('Unable to save data to', filename, ':', e)
    return file_path

meta_pickle = maybe_pickle(metadata, work_dir, 'metadata' )


# In[10]:

pickle_file = 'SVHN/metadata.pickle'
if os.path.exists(pickle_file):
    try:
        with open(pickle_file, 'rb') as f:
            save = pickle.load(f)
            num_train_examples = save['num_train_examples']
            num_val_examples = save['num_val_examples']
            num_test_examples = save['num_test_examples']
            del save
            print('num_train_examples', num_train_examples)
            print('num_val_examples', num_val_examples)
            print('num_test_examples', num_test_examples)
    except Exception as e:
        print('Unables to load data from', pickle_file, ':', e)


# In[11]:

class A(object):
    def foo(self,x):
        print("executing foo(%s, %s)"%(self,x)) #self --> __main__.A object at...
    @classmethod
    def class_foo(cls,x): #cls --> __main__.A
        print("executing class_foo(%s,%s)"%(cls,x))
    
    @staticmethod
    def static_foo(x):
        print("executing static_foo(%s)"%x)
a=A()
a.foo(1)
"""With classmethods, the class of the object instance is
   implicityly passed as the first argument instead of self.
   If you define something to be a classmethod it is probably
   because you intend to call it from the class rather than
   from a class instance."""
a.class_foo(1)
A.class_foo(2)
# A.foo(2)
a.static_foo(1)
A.static_foo('hi')
print(a.foo)
print(a.class_foo)
print(A.class_foo)
print(a.static_foo)
print(A.static_foo)

"""classmethod must have a reference to a class object as the first parameter,
   whereas staticmethod can have no parameters at all."""


# In[12]:

with h5py.File('SVHN/train/digitStruct.mat', 'r') as digit_struct_mat_file:
    item = digit_struct_mat_file['digitStruct']['bbox'][0].item()
    label = digit_struct_mat_file[item]['label']
    print(digit_struct_mat_file[label[1].item()][0][0])
#     for i in digit_struct_mat_file[item].__iter__():
#         print(i)


# In[22]:

def build_batch(train_tfrecords, num_examples, batch_size, shuffled):
#     dbg.set_trace()
    assert tf.gfile.Exists(train_tfrecords), '%s not found' % train_tfrecords
    filename_queue = tf.train.string_input_producer([train_tfrecords], num_epochs=None)
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        features = {
            'image': tf.FixedLenFeature([], tf.string),
            'length': tf.FixedLenFeature([],tf.int64),
            'digits': tf.FixedLenFeature([5],tf.int64)
        })
    image_vector = tf.decode_raw(features['image'], tf.uint8)
    image = tf.image.convert_image_dtype(image_vector, dtype = tf.float32)
    image = tf.multiply(tf.subtract(image,0.5), 2)
    image = tf.reshape(image,[64,64,3])
#     image = tf.random_crop(image, [54,54,3])
    length = tf.cast(features['length'], tf.int32)
    digits = tf.cast(features['digits'], tf.int32)
    min_queue_examples = int(0.4*num_examples)
    if shuffled:
        image_batch, length_batch, digits_batch = tf.train.shuffle_batch([image, length,digits],
                                                                        batch_size = batch_size,
                                                                        num_threads = 2,
                                                                        capacity = min_queue_examples + 3*batch_size,
                                                                        min_after_dequeue = min_queue_examples)
    else:
        image_batch, length_batch, digits_batch = tf.train.batch([image, length, digits],
                                                                batch_size = batch_size,
                                                                num_threads = 2,
                                                                capacity = min_queue_examples + 3*batch_size)
    return image_batch, length_batch, digits_batch


# In[14]:

def accuracy(predictions, labels):
    return (100 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))/predictions.shape[0])



# In[47]:

class model(object):
    @staticmethod
    def logits(image_batch, drop_rate):
        patch_size = 5
        num_channels = 3
        depth = 16
        num_hidden = 64
        num_length_label = 7
        num_digit_label = 11
        image_size = 64
        #     variable
        layer1_weights = tf.Variable(tf.truncated_normal([patch_size, patch_size, num_channels, depth], stddev = 0.1))
        layer1_biases = tf.Variable(tf.zeros([depth]))
        layer2_weights = tf.Variable(tf.truncated_normal([patch_size, patch_size, depth, depth], stddev = 0.1))
        layer2_biases = tf.Variable(tf.constant(1.0, shape = [depth]))
        layer3_weights = tf.Variable(tf.truncated_normal([image_size//4 *  image_size//4*depth, num_hidden],stddev = 0.1))
        layer3_biases = tf.Variable(tf.constant(1.0, shape = [num_hidden]))
        length_weights = tf. Variable(tf.truncated_normal([num_hidden, num_length_label], stddev=0.1))
        length_biases = tf.Variable(tf.constant(1.0, shape =[num_length_label]))
        digit1_weights = tf. Variable(tf.truncated_normal([num_hidden, num_digit_label], stddev=0.1))
        digit1_biases = tf.Variable(tf.constant(1.0, shape =[num_digit_label]))
        digit2_weights = tf. Variable(tf.truncated_normal([num_hidden, num_digit_label], stddev=0.1))
        digit2_biases = tf.Variable(tf.constant(1.0, shape =[num_digit_label]))
        digit3_weights = tf. Variable(tf.truncated_normal([num_hidden, num_digit_label], stddev=0.1))
        digit3_biases = tf.Variable(tf.constant(1.0, shape =[num_digit_label]))
        digit4_weights = tf. Variable(tf.truncated_normal([num_hidden, num_digit_label], stddev=0.1))
        digit4_biases = tf.Variable(tf.constant(1.0, shape =[num_digit_label]))
        digit5_weights = tf. Variable(tf.truncated_normal([num_hidden, num_digit_label], stddev=0.1))
        digit5_biases = tf.Variable(tf.constant(1.0, shape =[num_digit_label]))
   
        conv1 = tf.nn.conv2d(image_batch, layer1_weights, [1,1,1,1], padding = 'SAME')
        hidden1 = tf.nn.relu(conv1+layer1_biases)
        pool1 = tf.nn.max_pool(
            hidden1, ksize=[1,2,2,1], strides = [1,2,2,1], padding = 'SAME')
        conv = tf.nn.conv2d(pool1, layer2_weights, [1,1,1,1], padding = 'SAME')
        hidden2 = tf.nn.relu(conv+layer2_biases)
        pool = tf.nn.max_pool(
            hidden2, ksize=[1,2,2,1], strides =[1,2,2,1], padding = 'SAME')
        shape = pool.get_shape().as_list()
        reshape = tf.reshape(pool,[shape[0], shape[1]*shape[2]*shape[3]])
        hidden = tf.nn.relu(tf.matmul(reshape, layer3_weights) + layer3_biases)
        
        length_logits = tf.matmul(hidden, length_weights) + length_biases
        digit1_logits = tf.matmul(hidden, digit1_weights) + digit1_biases
        digit2_logits = tf.matmul(hidden, digit2_weights) + digit2_biases
        digit3_logits = tf.matmul(hidden, digit3_weights) + digit3_biases
        digit4_logits = tf.matmul(hidden, digit4_weights) + digit4_biases
        digit5_logits = tf.matmul(hidden, digit5_weights) + digit5_biases
         
        length_logits, digit_logits = length_logits, tf.stack([digit1_logits, digit2_logits, digit3_logits,
                                                                digit4_logits, digit5_logits], axis=1)    
        
        
        return length_logits, digit_logits
        
    @staticmethod
    def loss(length_batch, length_logits, digit_batch, digists_logits):
        length_cross_entropy = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels=length_batch, logits=length_logits))
        digit1_cross_entropy = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels=digit_batch[:, 0], logits=digits_logits[:, 0, :]))
        digit2_cross_entropy = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels=digit_batch[:, 1], logits=digits_logits[:, 1, :]))
        digit3_cross_entropy = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels=digit_batch[:, 2], logits=digits_logits[:, 2, :]))
        digit4_cross_entropy = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels=digit_batch[:, 3], logits=digits_logits[:, 3, :]))
        digit5_cross_entropy = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels=digit_batch[:, 4], logits=digits_logits[:, 4, :]))
        loss = length_cross_entropy + digit1_cross_entropy + digit2_cross_entropy + digit3_cross_entropy + digit4_cross_entropy + digit5_cross_entropy
        return loss
    


# In[83]:

class Evaluator(object):
#     def __init__(self, path_to_eval_log_dir):
#         self.summary_writer = tf.summary.FileWriter(path_to_eval_log_dir)

    def evaluate(self, path_to_checkpoint, path_to_tfrecords_file, num_examples, global_step):
        batch_size = 128
        num_batches = num_examples // batch_size
        needs_include_length = False

        with tf.Graph().as_default():
            image_batch, length_batch, digits_batch = build_batch(path_to_tfrecords_file,
                                                                  num_examples=num_examples,
                                                                  batch_size=batch_size,
                                                                  shuffled=False)
            length_logits, digits_logits = model.logits(image_batch, drop_rate=0.0)
            length_predictions = tf.argmax(length_logits, axis=1)
            digits_predictions = tf.argmax(digits_logits, axis=2)

            if needs_include_length:
                labels = tf.concat([tf.reshape(length_batch, [-1, 1]), digits_batch], axis=1)
                predictions = tf.concat([tf.reshape(length_predictions, [-1, 1]), digits_predictions], axis=1)
            else:
                labels = digits_batch
                predictions = digits_predictions

            labels_string = tf.reduce_join(tf.as_string(labels), axis=1)
            predictions_string = tf.reduce_join(tf.as_string(predictions), axis=1)

            accuracy, update_accuracy = tf.metrics.accuracy(
                labels=labels_string,
                predictions=predictions_string
            )

            with tf.Session() as sess:
                sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
                coord = tf.train.Coordinator()
                threads = tf.train.start_queue_runners(sess=sess, coord=coord)

                restorer = tf.train.Saver()
                restorer.restore(sess, path_to_checkpoint)

                for _ in range(num_batches):
                    sess.run(update_accuracy)

                accuracy_val = sess.run(accuracy)

                coord.request_stop()
                coord.join(threads)

        return accuracy_val


# In[84]:

def train(path_to_train_tfrecords_file, num_train_examples, path_to_val_tfrecords_file, num_val_examples,
           path_to_train_log_dir, path_to_restore_checkpoint_file):
    batch_size = 32
    initial_patience = 100
    num_steps_to_show_loss = 100
    num_steps_to_check = 1000
    decay_steps = 10000
    decay_rate = 0.9
    learning_rate = 1e-2
    graph = tf.Graph()
    with graph.as_default():
#             dbg.set_trace()
            image_batch, length_batch, digit_batch = build_batch(path_to_train_tfrecords_file,
                                                      num_examples = num_train_examples,
                                                      batch_size = batch_size,
                                                      shuffled = True)
            length_logits, digit_logits = model.logits(image_batch, drop_rate = 0.2)
            length_cross_entropy = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels=length_batch, logits=length_logits))
            digit1_cross_entropy = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels=digit_batch[:, 0], logits=digit_logits[:, 0, :]))
            digit2_cross_entropy = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels=digit_batch[:, 1], logits=digit_logits[:, 1, :]))
            digit3_cross_entropy = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels=digit_batch[:, 2], logits=digit_logits[:, 2, :]))
            digit4_cross_entropy = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels=digit_batch[:, 3], logits=digit_logits[:, 3, :]))
            digit5_cross_entropy = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels=digit_batch[:, 4], logits=digit_logits[:, 4, :]))
            loss = length_cross_entropy + digit1_cross_entropy + digit2_cross_entropy + digit3_cross_entropy + digit4_cross_entropy + digit5_cross_entropy

#             loss = model.loss(length_batch, length_logits, digit_batch, digit_logits)
            #Optimizer
            global_step = tf.Variable(0, trainable=False)
            learning_rate = tf.train.exponential_decay(learning_rate, global_step=global_step,
                                                   decay_steps=decay_steps, decay_rate=decay_rate, staircase=True)
            optimizer = tf.train.GradientDescentOptimizer(learning_rate)
            train_op = optimizer.minimize(loss, global_step=global_step)
            with tf.Session() as sess:
                evaluator = Evaluator()
                sess.run(tf.global_variables_initializer())
                coord = tf.train.Coordinator()
                threads = tf.train.start_queue_runners(sess=sess, coord=coord)
                saver = tf.train.Saver()
                if path_to_restore_checkpoint_file is not None:
                    assert tf.train.checkpoint_exists(path_to_restore_checkpoint_file),                         '%s not found' % path_to_restore_checkpoint_file
                    saver.restore(sess, path_to_restore_checkpoint_file)
                    print ('Model restored from file: %s' % path_to_restore_checkpoint_file)
                print ('Start training')
                patience = initial_patience
                best_accuracy = 0.0
                duration = 0.0
#                 print(sess.run(learning_rate))
                while True:
                    start_time = time.time()
                    _, loss_val, global_step_val, learning_rate_val = sess.run([train_op, loss, global_step, learning_rate])
                    duration += time.time() - start_time

                    if global_step_val % num_steps_to_show_loss == 0:
                        examples_per_sec = batch_size * num_steps_to_show_loss / duration
                        duration = 0.0
                        print ('=> %s: step %d, loss = %f (%.1f examples/sec)' % (
                            datetime.now(), global_step_val, loss_val, examples_per_sec))
                    if global_step_val % num_steps_to_check != 0:
                        continue


                    print ('=> Evaluating on validation dataset...')
                    path_to_latest_checkpoint_file = saver.save(sess, os.path.join(path_to_train_log_dir, 'latest.ckpt'))
                    accuracy = evaluator.evaluate(path_to_latest_checkpoint_file, path_to_val_tfrecords_file,
                                              num_val_examples,
                                              global_step_val)
                    print ('==> accuracy = %f, best accuracy %f' % (accuracy, best_accuracy))

                    if accuracy > best_accuracy:
                        path_to_checkpoint_file = saver.save(sess, os.path.join(path_to_train_log_dir, 'model.ckpt'),
                                                         global_step=global_step_val)
                        print ('=> Model saved to file: %s' % path_to_checkpoint_file)
                        patience = initial_patience
                        best_accuracy = accuracy
                    else:
                        patience -= 1

                    print ('=> patience = %d' % patience)
                    if patience == 0:
                        break

                coord.request_stop()
                coord.join(threads)
                print ('Finished')
            


# In[ ]:

log_dir = 'SVHN/log'
if not os.path.exists(log_dir):
    os.mkdir(log_dir)
train(train_tfrecords,num_train_examples,valid_tfrecords,num_val_examples,log_dir,None)

