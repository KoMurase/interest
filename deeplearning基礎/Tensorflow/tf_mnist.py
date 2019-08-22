import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

import os
import struct
import numpy as np

def load_mnist(path,kind = 'train'):
    labels_path = os.path.join(path,'%s-labels-idx1-ubyte' % kind)
    image_path = os.path.join(path,'%s-labels-idx3-ubyte' % kind)

    with open(labels_path, 'rb') as lbpath:

        magic, n = struct.unpack('>II',lbpath.read(8))

        labels = np.fromfile(lbpath,dtype=np.uint8)

    with open(image_path,'rb') as imgpath:
        magic,num,rows,cols = struct.unpack('>IIII',imgpath.read(16))
        images = np.fromfle(imgpath,dttype=np.uint8).reshape(len(labels),784)
        images = ((images/255.)-.5) * 2

    return images ,labels


X_data ,y_data = load_mnist('./',kind='train')
