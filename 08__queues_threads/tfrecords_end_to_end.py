# -*- coding: utf-8 -*-
from __future__ import print_function

"""
Created on Thu Jan 26 00:41:43 2017

@author: tomhope
"""

import os
import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets import mnist

#### WRITE TFRECORDS  # noqa
save_dir = "D:\\mnist"

# Download data to save_Dir
data_sets = mnist.read_data_sets(save_dir,
                                 dtype=tf.uint8,
                                 reshape=False,
                                 validation_size=1000)

data_splits = ["train", "test", "validation"]
for d in range(len(data_splits)):
    print("saving " + data_splits[d])
    data_set = data_sets[d]

    filename = os.path.join(save_dir, data_splits[d] + '.tfrecords')
    writer = tf.python_io.TFRecordWriter(filename)
    for index in range(data_set.images.shape[0]):
        image = data_set.images[index].tostring()
        example = tf.train.Example(features=tf.train.Features(feature={
            'height': tf.train.Feature(int64_list=tf.train.Int64List(
                value=[data_set.images.shape[1]])),
            'width': tf.train.Feature(int64_list=tf.train.Int64List(
                value=[data_set.images.shape[2]])),
            'depth': tf.train.Feature(int64_list=tf.train.Int64List(
                value=[data_set.images.shape[3]])),
            'label': tf.train.Feature(int64_list=tf.train.Int64List(
                value=[int(data_set.labels[index])])),
            'image_raw': tf.train.Feature(bytes_list=tf.train.BytesList(
                value=[image]))}))
        writer.write(example.SerializeToString())
    writer.close()


# READ
NUM_EPOCHS = 10

filename = os.path.join("D:\\mnist", "train.tfrecords")

filename_queue = tf.train.string_input_producer(
    [filename], num_epochs=NUM_EPOCHS)


reader = tf.TFRecordReader()
_, serialized_example = reader.read(filename_queue)
features = tf.parse_single_example(
    serialized_example,
    features={
        'image_raw': tf.FixedLenFeature([], tf.string),
        'label': tf.FixedLenFeature([], tf.int64),
    })

image = tf.decode_raw(features['image_raw'], tf.uint8)
image.set_shape([784])


image = tf.cast(image, tf.float32) * (1. / 255) - 0.5

label = tf.cast(features['label'], tf.int32)


# Shuffle the examples + batch
images_batch, labels_batch = tf.train.shuffle_batch(
    [image, label], batch_size=128,
    capacity=2000,
    min_after_dequeue=1000)


W = tf.get_variable("W", [28*28, 10])
y_pred = tf.matmul(images_batch, W)
loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y_pred, labels=labels_batch)

loss_mean = tf.reduce_mean(loss)

train_op = tf.train.AdamOptimizer().minimize(loss)

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
init = tf.local_variables_initializer()
sess.run(init)

# coordinator
coord = tf.train.Coordinator()
try:
    step = 0
    while not coord.should_stop():
        step += 1
        sess.run([train_op])
        if step % 500 == 0:
            loss_mean_val = sess.run([loss_mean])
            print(step)
            print(loss_mean_val)
except tf.errors.OutOfRangeError:
    print('Done training for %d epochs, %d steps.' % (NUM_EPOCHS, step))
finally:
    # When done, ask the threads to stop.
    coord.request_stop()


# example -- get image,label
img1, lbl1 = sess.run([image, label])

# example - get random batch
labels, images = sess.run([labels_batch, images_batch])
