# -*- coding: utf-8 -*-
"""
Created on Sat Jan 21 23:52:54 2017

@author: tomhope
"""
from __future__ import print_function

import tensorflow as tf

import threading
import time

sess = tf.InteractiveSession()

queue1 = tf.FIFOQueue(capacity=10, dtypes=[tf.string])

enque_op = queue1.enqueue(["F"])
# size is 0 before run
sess.run(queue1.size())
enque_op.run()
sess.run(queue1.size())

enque_op = queue1.enqueue(["I"])
enque_op.run()
enque_op = queue1.enqueue(["F"])
enque_op.run()
enque_op = queue1.enqueue(["O"])
enque_op.run()

sess.run(queue1.size())


x = queue1.dequeue()
x.eval()
x.eval()
x.eval()
x.eval()

# Hangs forever if queue empty if running INTERACTIVELY

# for dequeue many, need to specify shapes in advance...
queue1 = tf.FIFOQueue(capacity=10, dtypes=[tf.string], shapes=[()])
# ....
inputs = queue1.dequeue_many(4)

inputs.eval()


# single queue, but execute sess.run calls in parallel...
gen_random_normal = tf.random_normal(shape=())
queue = tf.FIFOQueue(capacity=100, dtypes=[tf.float32], shapes=())
enque = queue.enqueue(gen_random_normal)


def add():
    for i in range(10):
        sess.run(enque)


# Create 10 threads that run add()
threads = [threading.Thread(target=add, args=()) for i in range(10)]
for t in threads:
    t.start()
print(sess.run(queue.size()))
time.sleep(0.01)
print(sess.run(queue.size()))
time.sleep(0.01)
print(sess.run(queue.size()))


x = queue.dequeue_many(10)
print(x.eval())
sess.run(queue.size())


# A coordinator for threads.
# a simple mechanism to coordinate the termination of a set of threads

gen_random_normal = tf.random_normal(shape=())
queue = tf.FIFOQueue(capacity=100, dtypes=[tf.float32], shapes=())
enque = queue.enqueue(gen_random_normal)


def add(coord, i):
    while not coord.should_stop():
        sess.run(enque)
        if i == 1:
            coord.request_stop()


coord = tf.train.Coordinator()
threads = [threading.Thread(target=add, args=(coord, i)) for i in range(10)]
coord.join(threads)

for t in threads:
    t.start()

print(sess.run(queue.size()))
time.sleep(0.01)
print(sess.run(queue.size()))
time.sleep(0.01)
print(sess.run(queue.size()))


gen_random_normal = tf.random_normal(shape=())
queue = tf.RandomShuffleQueue(capacity=100, dtypes=[tf.float32],
                              min_after_dequeue=1)
enqueue_op = queue.enqueue(gen_random_normal)

qr = tf.train.QueueRunner(queue, [enqueue_op] * 4)
coord = tf.train.Coordinator()
enqueue_threads = qr.create_threads(sess, coord=coord, start=True)
coord.request_stop()
coord.join(enqueue_threads)

print(sess.run(queue.size()))
