# Use a trained DenseFuse Net to generate fused images

import tensorflow as tf
import numpy as np
from scipy.misc import imread, imsave
from datetime import datetime
from os import listdir, mkdir, sep
from os.path import join, exists, splitext
from mask_net import mask_net
import time
from skimage import color, measure, morphology, io
import matplotlib.pyplot as plt

def generate(path1, path2, model_path, index, output_path = None):
	img1 = imread(path1) / 255.0
	img2 = imread(path2) / 255.0
	dimension = img1.shape
	h = dimension[0]
	w = dimension[1]
	c = dimension[2]

	if c == 3:
		img1_gray = color.rgb2gray(img1)
		img2_gray = color.rgb2gray(img2)
	else:
		img1_gray = img1
		img2_gray = img2

	img1_gray = img1_gray.reshape([1, h, w, 1])
	img2_gray = img2_gray.reshape([1, h, w, 1])


	with tf.Graph().as_default(), tf.Session() as sess:
		shape = [1, h, w, 1]
		SOURCE1 = tf.placeholder(tf.float32, shape = shape, name = 'SOURCE1')
		SOURCE2 = tf.placeholder(tf.float32, shape = shape, name = 'SOURCE2')

		# source_field = tf.placeholder(tf.float32, shape = source_shape, name = 'source_imgs')

		MN = mask_net('mask_net')
		MAP, MASK = MN.transform(img1 = SOURCE1, img2 = SOURCE2)

		G1 = grad(SOURCE1)
		G2 = grad(SOURCE2)
		GRAD_MAX = tf.cast(tf.abs(G1) > tf.abs(G2), dtype = tf.float32)

		# restore the trained model
		saver = tf.train.Saver()
		saver.restore(sess, model_path)

		output_mask = sess.run(MASK, feed_dict = {SOURCE1: img1_gray, SOURCE2: img2_gray})
		output_mask = output_mask[0, :, :, 0]

		grad_max, g1, g2 = sess.run([GRAD_MAX, tf.abs(G1), tf.abs(G2)], feed_dict = {SOURCE1: img1_gray, SOURCE2: img2_gray})
		grad_max = grad_max[0, :, :, 0]
		g1 = g1[0, :, :, 0]
		g2 = g2[0, :, :, 0]

		# subsequent process
		mask_bool1 = (1 - output_mask) > 0
		dst1 = morphology.remove_small_objects(mask_bool1, min_size = h * w / 60, connectivity = 1, in_place = False)
		dst1 = dst1.astype(np.float32)
		mask_bool2 = (1 - dst1) > 0
		dst2 = morphology.remove_small_objects(mask_bool2, min_size = h * w / 60, connectivity = 1, in_place = False)
		dst2 = dst2.astype(np.float32)

		# fig = plt.figure()
		# mask0 = fig.add_subplot(311)
		# mask1 = fig.add_subplot(312)
		# mask2 = fig.add_subplot(313)
		# mask0.imshow(output_mask, cmap = 'gray')
		# mask1.imshow(dst1, cmap = 'gray')
		# mask2.imshow(dst2, cmap = 'gray')
		# plt.show()

		output_img = np.zeros_like(img1)
		for i in range(c):
			output_img[:, :, i] = np.multiply(dst2, img1[:, :, i]) + np.multiply((1-dst2),
			                                                                            img2[:, :, i])

		imsave(output_path + str(index) + '.jpg', output_img)
		# imsave(output_path + str(index) + '_mask0.jpg', output_mask)
		# imsave(output_path + str(index) + '_mask1.jpg', dst1)
		imsave(output_path + str(index) + '_mask2.jpg', dst2)
		# imsave(output_path + str(index) + '_gradmax.jpg', grad_max)
		# imsave(output_path + str(index) + '_grad1.jpg', g1)
		# imsave(output_path + str(index) + '_grad2.jpg', g2)

def grad(img):
	kernel = tf.constant([[-1 / 8, -1 / 8, -1 / 8], [-1 / 8, 1, -1 / 8], [-1 / 8, -1 / 8, -1 / 8]])
	kernel = tf.expand_dims(kernel, axis = -1)
	kernel = tf.expand_dims(kernel, axis = -1)
	g = tf.nn.conv2d(img, kernel, strides = [1, 1, 1, 1], padding = 'SAME')
	return g