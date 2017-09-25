
# coding: utf-8

# In[15]:

from __future__ import print_function
import os
import numpy as np
import gzip
import tensorflow as tf
from six.moves.urllib.request import urlretrieve
from shutil import copyfile


# In[29]:

TRAIN_IMAGES = 'train-images-idx3-ubyte.gz'
TRAIN_LABELS = 'train-labels-idx1-ubyte.gz'
TEST_IMAGES = 't10k-images-idx3-ubyte.gz'
TEST_LABELS = 't10k-labels-idx1-ubyte.gz'

SOURCE_URL = 'https://storage.googleapis.com/cvdf-datasets/mnist/'
def maybe_download(filename, work_directory, source_url):
    """Download the data from source_url, unless it's already here
    
        Args:
            filename: string, name of the file in the directory.
            work_directory: string, path to working directory.
            source_url: url to download from if file doesn't exist.
            
        Returns:
             Path to resulting file.
    """
    if not os.path.exists(work_directory):
        os.mkdir(work_directory)
    filepath = os.path.join(work_directory, filename)
    if not os.path.exists(filepath):
        temp_file_name, _ = urlretrieve(source_url)
        copyfile(temp_file_name, filepath)
    print('Successfully downloaded', filename, os.stat(filepath).st_size, 'bytes')
    return filepath



# In[31]:

def _read32(bytestream):
    dt = numpy.dtype(numpy.uint32).newbyteorder('>')
    return numpy.frombuffer(bytestream.read(4), dtype = dt)[0]


# In[32]:

def extract_images(f):
    """Extract the images into a 4D uint8 numpy array [index, y, x, depth].
    
    Args:
        f: A file object that can be passed into a gzip reader.
    Returns:
        data: A 4D uint8 numpy array [index, y, x, depth].
    Raises:
        ValueError: If the bytestream does not start with 2051
    """
    
    print('Extrating', f.name)
    with gzip.GzipFile(fileobj=f) as bytestream:
        magic = _read32(bytestream)
        if magic != 2051:
            raise ValueError('Invalid magic number %d in MNIST image file: %s' %
                            (magic, f.name))
            num_images = _read32(bytestream)
            rows = _read32(bytestream)
            cols = _read32(bytestream)
            buf = bytestream.read(rows * cols * numimages)
            data = numpy.frombuffer(buf, dtype = numpy.uint8)
            data = data.reshape(num_images, rows, ccols, 1)
            return data


# In[30]:

work_dir = "MNIST_data/"
train_path = maybe_download(TRAIN_IMAGES, work_dir, SOURCE_URL + TRAIN_IMAGES)

