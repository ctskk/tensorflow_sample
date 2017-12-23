#!/usr/bin/python
# -*- coding: utf-8 -*-
# pylint: disable=C0103, C0111, C0326, C0301, E1129, C0325, W0613, R0914

#tensorboard --logdir=./data でport6006が開く

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import numpy as np

import tensorflow as tf

import model as model
from reader import Cifar10Reader

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('epoch', 30, "訓練するEpoch数")
tf.app.flags.DEFINE_string('data_dir', './data/', "訓練データのディレクトリ")
tf.app.flags.DEFINE_string('checkpoint_dir', './checkpoints/', "チェックポイントを保存するディレクトリ")

def _loss(logits, label):
    labels = tf.cast(label, tf.int64)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels, name='cross_entropy_per_example')
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    return cross_entropy_mean

def _train(total_loss, global_step):
    opt = tf.train.GradientDescentOptimizer(learning_rate=0.001)
    grads = opt.compute_gradients(total_loss)
    train_op = opt.apply_gradients(grads, global_step=global_step)
    return train_op

filenames = [
    os.path.join(FLAGS.data_dir, 'data_batch_%d.bin' % i) for i in range(1, 6)
]

def main(argv=None):
    global_step = tf.Variable(0, trainable=False)

    train_placeholder = tf.placeholder(tf.float32, shape=[32, 32, 3], name='input_image')
    label_placeholder = tf.placeholder(tf.int32, shape=[1], name='label')

    #(height, width, depth) -> (batch, height, width, depth)
    #処理inferenceにはplaceholderを与えて画像を32x32x3の次元に展開させたものを与える。
    image_node = tf.expand_dims(train_placeholder, 0)

    logits = model.inference(image_node)
    total_loss = _loss(logits, label_placeholder)
    train_op = _train(total_loss, global_step)

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())

        total_duration = 0

        #30世代回す
        for epoch in range(1, FLAGS.epoch + 1):
            start_time = time.time()

            #data_batch_1-5.bin を処理する
            for file_index in range(5):
                print('Epoch %d: %s' % (epoch, filenames[file_index]))
                reader = Cifar10Reader(filenames[file_index])

                #1ファイルあたり10000画像入っているので全部処理する
                for index in range(10000):
                    image = reader.read(index)

                    #sess.runにはplaceholderをキーにして、流し込むバッファを指定すると上手いこと型が合うように渡される
                    _, loss_value, logits_value = sess.run(
                        [train_op, total_loss, logits],
                        feed_dict={
                            train_placeholder: image.byte_array,
                            label_placeholder: image.label
                        })

                    assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

                    if index % 1000 == 0:
                        print('[%d]: %r' % (image.label, logits_value))

                reader.close()

            duration = time.time() - start_time
            total_duration += duration

            #1世代ごとに時間とLogを出力する
            print('epoch %d duration = %d src' % (epoch, duration))
            tf.summary.FileWriter(FLAGS.checkpoint_dir, sess.graph)

        print('Total duration = %d src' % total_duration)

if __name__ == '__main__':
    tf.app.run()
