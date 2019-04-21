#!/usr/bin/python
# -*- coding: utf-8 -*-
# pylint: disable=C0103, C0111, W0201, E1101

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np

class Cifar10Record(object):

	width = 32
	height = 32
	depth = 3

	def set_label(self, label_byte):
		'''label_byteで与えられたサイズのバイト列を行列にして保持する'''
		self.label = np.frombuffer(label_byte, dtype=np.uint8)

	def set_image(self, image_bytes):
		'''label_byteで与えられたサイズのバイト列をdepth x height x width の行列にして保持する'''
		#まず1次元のndarrayにする
		byte_buffer = np.frombuffer(image_bytes, dtype=np.int8)
		# 3 x 32 x 32 の行列に変換する(rgbが縦に入っていて、それが幅分続くフォーマット)
		reshaped_array = np.reshape(byte_buffer, [self.depth, self.height, self.width])
		# 高, 幅, rgb の行列に変更する(rgb値が横方向に並ぶ一般系になる)
		self.byte_array = np.transpose(reshaped_array, [1, 2, 0])
		# rgb の各要素を float32 にキャストする
		self.byte_array = self.byte_array.astype(np.float32)

class Cifar10Reader(object):

	def __init__(self, filename):
		if not os.path.exists(filename):
			print(filename + ' is not exist')
			return

		self.bytestream = open(filename, mode="rb")

	def close(self):
		if not self.bytestream:
			self.bytestream.close()

	def read(self, index):
		result = Cifar10Record()

		#元画像データにはlabel + h * w * rgb が入っている
		label_bytes = 1
		image_bytes = result.height * result.width * result.depth
		record_bytes = label_bytes + image_bytes

		self.bytestream.seek(record_bytes * index, 0)

		#label と 画像部を recordに保存する
		result.set_label(self.bytestream.read(label_bytes))
		result.set_image(self.bytestream.read(image_bytes))

		return result
