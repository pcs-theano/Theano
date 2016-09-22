import caffe
import lmdb
import time
import numpy as np
import theano
import random

class read_lmdb(object):

    def __init__(self, batch_size, lmdb_file, image_mean_path=None):
        self.image_shape = get_image_shape(lmdb_file)
        if image_mean_path is None:        
            self.image_mean = get_image_mean((104, 117, 123), self.image_shape)
        else:
            self.image_mean = get_image_mean(image_mean_path, self.image_shape)
        self.lmdb_data = lmdb.open(lmdb_file, readonly=True)
        self.in_txn = self.lmdb_data.begin()
        self.total_number = int(self.lmdb_data.stat()['entries'])
        self.cursor = self.in_txn.cursor()
        self.batch_size = batch_size
        self.output_data = np.zeros((self.batch_size,)+self.image_shape, dtype=theano.config.floatX)
        self.output_label = np.zeros((self.batch_size,), dtype='int32')

    def __del__(self):
        self.cursor.close()
        self.lmdb_data.close()

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def next(self):
        for i in xrange(self.batch_size):
    
            if self.cursor.next() is False:
                self.cursor.first()

            (key, value) = self.cursor.item()
            raw_datum = bytes(value)
            datum = caffe.proto.caffe_pb2.Datum()
            datum.ParseFromString(value)
            label = datum.label
            data = caffe.io.datum_to_array(datum)[np.newaxis] - self.image_mean
            self.output_data[i,...] = data
            self.output_label[i] = label

        return (transform(self.output_data, self.image_shape), self.output_label)

    def set_cursor(self, n):
        self.cursor = self.in_txn.cursor()
        if n > 0:
            for i in xrange(n*self.batch_size):
                self.cursor.next()


def get_image_mean(image_mean, image_shape):
    if isinstance(image_mean, tuple):
        image_mean_out = np.zeros(image_shape)
        for i in xrange(len(image_mean)):
            image_mean_out[i,...] = image_mean[i]
        image_mean_out = image_mean_out[np.newaxis]
    else:
        blob = caffe.proto.caffe_pb2.BlobProto()
        data = open(image_mean, 'rb').read()
        blob.ParseFromString(data[0:-1])
        image_mean_out = np.array(caffe.io.blobproto_to_array(blob))
    return image_mean_out

def transform(input, input_shape, crop_size=224, is_mirror=True):
    mirror = input[:, :, :, ::-1]
    crop_x = random.randint(0, (input_shape[-1]-crop_size)-1)
    crop_y = random.randint(0, (input_shape[-2]-crop_size)-1)
    do_mirror = is_mirror and random.randint(0, 1) 
    
    if do_mirror:
        return mirror[:, :, crop_y:crop_y+crop_size, crop_x:crop_x+crop_size]
    else:
        return input[:, :, crop_y:crop_y+crop_size, crop_x:crop_x+crop_size]

def get_image_shape(lmdb_file):
    lmdb_data = lmdb.open(lmdb_file, readonly=True)
    in_txn = lmdb_data.begin()
    cursor = in_txn.cursor()
    cursor.first()
    (key, value) = cursor.item()
    raw_datum = bytes(value)
    datum = caffe.proto.caffe_pb2.Datum()
    datum.ParseFromString(value)
    data = caffe.io.datum_to_array(datum)
    image_height = data.shape[1]
    image_width = data.shape[2]
    image_channels = data.shape[0]
    cursor.close()
    lmdb_data.close()
    return (image_channels, image_height, image_width)
