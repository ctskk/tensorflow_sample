#!/usr/bin/python
# -*- coding: utf-8 -*-
# pylint: disable=C0103, C0111, C0326, C0301, E1129, W0613

#tensorboard --logdir=./data でport6006が開く

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time

import tensorflow as tf

import model as model
from reader import Cifar10Reader

#FLAGSのプロパティを設定していく
#ソース中の利用方法を見るに単なる直値なので単純変数でも良さそうだが、Graph生成上の都合と思われる
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('epoch', 30, "訓練するEpoch数")
tf.app.flags.DEFINE_string('data_dir', './data/', "訓練データのディレクトリ")
tf.app.flags.DEFINE_string('checkpoint_dir', './checkpoints/', "チェックポイントを保存するディレクトリ")

#読み出し対象となるファイル data_batch_N.bin を配列にしておく
filenames = [
	os.path.join(FLAGS.data_dir, 'data_batch_%d.bin' % i) for i in range(1, 6)
]

def main(argv=None):
	train_placeholder = tf.placeholder(tf.float32, shape=[32, 32, 3], name='input_image')

	#(height, width, depth) -> (batch, height, width, depth)
	#処理inferenceにはplaceholderを与えて画像を32x32x3の次元に展開させたものを与える。
	image_node = tf.expand_dims(train_placeholder, 0)   #指定列の直前を1列挿入する
	logits = model.inference(image_node)

	with tf.Session() as sess:
		sess.run(tf.initialize_all_variables())

		total_duration = 0

		#30世代回す
		for epoch in range(1, FLAGS.epoch + 1):
			start_time = time.time()

			#data_batch_1-5.bin を処理する
			for file_index in range(5):
				print('Epoch %d: %s' % (epoch, filenames[file_index]))

				#試験データを 高, 幅, rgb の行列で読み込むためのクラスを初期化
				reader = Cifar10Reader(filenames[file_index])

				#1ファイルあたり10000画像入っているので全部処理する
				for index in range(10000):
					#画像データを label + w, h, rgb で読み込む
					image = reader.read(index)

					#sess.runにはplaceholderをキーにして、流し込むバッファを指定すると上手いこと型が合うように渡される
					logits_value = sess.run([logits], feed_dict={train_placeholder: image.byte_array,})

					if index % 1000 == 0:
						print('[%d]: %r' % (image.label, logits_value))

				reader.close()

			# 時間計測
			duration = time.time() - start_time
			total_duration += duration

			#1世代ごとに時間とLogを出力する
			print('epoch %d duration = %d src' % (epoch, duration))
			tf.summary.FileWriter(FLAGS.checkpoint_dir, sess.graph)

		print('Total duration = %d sec' % total_duration)

if __name__ == '__main__':
	tf.app.run()
