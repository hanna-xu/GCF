# Train the DenseFuse Net

from __future__ import print_function

import scipy.io as scio
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import time
from scipy.misc import imsave
import scipy.ndimage
from mask_net import mask_net
from boundary_net import boundary_net
from connected_regions import connected_regions
from skimage import morphology

patch_size = 106
# TRAINING_IMAGE_SHAPE = (patch_size, patch_size, 2)  # (height, width, color_channels)

LEARNING_RATE = 0.003
EPSILON = 1e-5
DECAY_RATE = 0.75
eps = 1e-8


def train_mask_net(source_imgs, save_path, EPOCHES_set, BATCH_SIZE, logging_period = 1):
	from datetime import datetime
	start_time = datetime.now()
	EPOCHS = EPOCHES_set
	print('Epoches: %d, Batch_size: %d' % (EPOCHS, BATCH_SIZE))

	MODEL_SAVE_PATH = save_path + 'temporary.ckpt'
	num_imgs = source_imgs.shape[0]
	mod = num_imgs % BATCH_SIZE
	n_batches = int(num_imgs // BATCH_SIZE)
	print('Train images number %d, Batches: %d.\n' % (num_imgs, n_batches))

	if mod > 0:
		print('Train set has been trimmed %d samples...\n' % mod)
		source_imgs = source_imgs[:-mod]

	# create the graph
	with tf.Graph().as_default(), tf.Session() as sess:
		SOURCE1 = tf.placeholder(tf.float32, shape = (BATCH_SIZE, patch_size, patch_size, 1), name = 's1')
		SOURCE2 = tf.placeholder(tf.float32, shape = (BATCH_SIZE, patch_size, patch_size, 1), name = 's2')
		print('source img shape:', SOURCE1.shape)

		# upsampling vis and ir images
		MN = mask_net('mask_net')
		MAP, MASK= MN.transform(img1 = SOURCE1, img2 = SOURCE2)

		g1 = grad(SOURCE1)
		g2 = grad(SOURCE2)
		gmax_mask = tf.cast(tf.abs(g1) > tf.abs(g2), dtype = tf.float32)
		# gmax_mask = fil(gmax_mask0)
		LOSS1 = tf.reduce_mean(tf.square(MAP - gmax_mask))

		# LOSS1 = 1-0.5 * SSIM_LOSS(generated_img, SOURCE1) - 0.5 * SSIM_LOSS(generated_img, SOURCE2)
		#LOSS1 = tf.reduce_mean(tf.reduce_sum(tf.square(MAP-gmax_mask), axis = [1, 2, 3])/(patch_size*patch_size))
		# LOSS1=-tf.reduce_mean(tf.square(gf))
		# LOSS1=tf.reduce_mean(tf.norm(tf.square(MAP-gmax_mask), ord = 1)/(patch_size * patch_size))

		regions_numbers1, diffs1, labels1, min_area1 = connected_regions(MASK)
		regions_numbers2, _, _, min_area2 = connected_regions(1 - MASK)
		LOSS2 = tf.reduce_mean(regions_numbers1 + 0.8 * regions_numbers2)
		# LOSS3 = - tf.reduce_mean(min_area1 + min_area2) / (patch_size * patch_size)

		w2 = 5e-3 # with 3:3.2e-4
		LOSS = LOSS1 + w2*LOSS2


		current_iter = tf.Variable(0)
		learning_rate = tf.train.exponential_decay(learning_rate = LEARNING_RATE, global_step = current_iter,
		                                           decay_steps = int(n_batches), decay_rate = DECAY_RATE,
		                                           staircase = False)
		theta = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = 'mask_net')

		solver = tf.train.RMSPropOptimizer(learning_rate).minimize(LOSS, global_step = current_iter,
		                                                              var_list = theta)
		clip = [p.assign(tf.clip_by_value(p, -30, 30)) for p in theta]

		sess.run(tf.global_variables_initializer())
		saver = tf.train.Saver(max_to_keep = 100)

		tf.summary.scalar('Loss', LOSS)
		tf.summary.scalar('Loss1', LOSS1)
		tf.summary.scalar('Loss2', LOSS2)
		tf.summary.scalar('Learning rate', learning_rate)

		tf.summary.image('source1', SOURCE1, max_outputs = 5)
		tf.summary.image('source2', SOURCE2, max_outputs = 5)
		# tf.summary.image('map', MAP, max_outputs = 5)
		tf.summary.image('mask', MASK, max_outputs = 5)
		tf.summary.image('gmax_mask', gmax_mask, max_outputs = 5)

		merged = tf.summary.merge_all()
		writer = tf.summary.FileWriter("logs_mn/", sess.graph)

		# Start Training
		step = 0
		num_imgs = source_imgs.shape[0]

		for epoch in range(EPOCHS):
			np.random.shuffle(source_imgs)
			for batch in range(n_batches):
				step += 1
				current_iter = step
				index = np.random.choice([0, 1], 1)
				batch1 = source_imgs[batch * BATCH_SIZE:(batch * BATCH_SIZE + BATCH_SIZE), :, :, index[0]]
				batch1 = np.expand_dims(batch1, -1)
				batch2 = source_imgs[batch * BATCH_SIZE:(batch * BATCH_SIZE + BATCH_SIZE), :, :, 1-index[0]]
				batch2 = np.expand_dims(batch2, -1)
				FEED_DICT = {SOURCE1: batch1, SOURCE2: batch2}

				sess.run([solver, clip], feed_dict = FEED_DICT)
				loss, loss1, loss2 = sess.run([LOSS, LOSS1, LOSS2], feed_dict = FEED_DICT)

				print("Epoch: [%3s], Batch: [%4s]" % (epoch+1, batch+1))
				if batch % 10 == 0:
					elapsed_time = datetime.now() - start_time
					lr = sess.run(learning_rate)
					print("loss: [%.6s], lr: [%.6s]" % (loss, lr))
					print("elapsed_time: %s\n" % (elapsed_time))

				result = sess.run(merged, feed_dict = FEED_DICT)
				writer.add_summary(result, step)
				if step % logging_period == 0:
					saver.save(sess, save_path + str(step) + '/' + str(step) + '.ckpt')

	writer.close()
	saver.save(sess, save_path + str(epoch) + '/' + str(epoch) + '.ckpt')



# def train_boundary_net(source_imgs, save_path, mask_model_path, mask_model_num, EPOCHES, BATCH_SIZE, logging_period = 50):
# 	from datetime import datetime
# 	start_time = datetime.now()
# 	print('Epoches: %d, Batch_size: %d' % (EPOCHES, BATCH_SIZE))
# 	num_imgs = source_imgs.shape[0]
# 	mod = num_imgs % BATCH_SIZE
# 	n_batches = int(num_imgs // BATCH_SIZE)
# 	print('Train images number %d, Batches: %d.\n' % (num_imgs, n_batches))
#
# 	if mod > 0:
# 		print('Train set has been trimmed %d samples...\n' % mod)
# 		source_imgs = source_imgs[:-mod]
#
#
# 	# create the graph
# 	with tf.Graph().as_default(), tf.Session() as sess:
# 		ORI_MASK = tf.placeholder(tf.float32, shape = (BATCH_SIZE, patch_size, patch_size, 1), name = 'om')
# 		S1 = tf.placeholder(tf.float32, shape = (BATCH_SIZE, patch_size, patch_size, 1), name = '1')
# 		S2 = tf.placeholder(tf.float32, shape = (BATCH_SIZE, patch_size, patch_size, 1), name = '2')
# 		SP = tf.placeholder(tf.float32, shape = (BATCH_SIZE, patch_size, patch_size, 1), name = 'sp')
# 		BN = boundary_net('boundary_net')
# 		MASK = BN.transform(i=SP, origin_mask = ORI_MASK)
# 		MASK2=tf.multiply(MASK, ORI_MASK)
#
# 		grad_mask = grad(MASK)
# 		grad_mask = binary(grad_mask - 0.35) / 2 + 0.5
#
# 		# both_ones_ratio = tf.reduce_sum(tf.multiply(grad_mask, SP), axis = [1, 2]) / tf.reduce_sum(grad_mask + eps,
# 		#                                                                                            axis = [1, 2])
# 		both_ones_ratio = - tf.reduce_mean(tf.reduce_mean(tf.multiply(grad_mask, SP), axis = [1, 2]))
# 		# / tf.reduce_sum(SP + eps, axis = [1, 2])
# 		both_zeros_ratio = -tf.reduce_mean(tf.reduce_mean(tf.multiply(1 - grad_mask, 1 - SP), axis = [1, 2]))
# 		#  / tf.reduce_sum(1 - SP + eps, axis = [1, 2])
# 		LOSS1 = 0*both_ones_ratio + 1 * both_zeros_ratio
# 		LOSS2= tf.reduce_mean(tf.square(MASK-ORI_MASK))
# 		w2 = 0.1 #0.6
# 		LOSS = LOSS1 + w2*LOSS2
#
# 		current_iter = tf.Variable(0)
# 		learning_rate = tf.train.exponential_decay(learning_rate = 0.001, global_step = current_iter,
# 		                                           decay_steps = int(n_batches), decay_rate = DECAY_RATE,
# 		                                           staircase = False)
# 		theta = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = 'boundary_net')
# 		solver = tf.train.RMSPropOptimizer(learning_rate).minimize(LOSS, global_step = current_iter,
# 		                                                              var_list = theta)
# 		clip = [p.assign(tf.clip_by_value(p, -30, 30)) for p in theta]
#
# 		sess.run(tf.global_variables_initializer())
# 		saver = tf.train.Saver(max_to_keep = 50)
#
# 		tf.summary.scalar('Loss', LOSS)
# 		tf.summary.scalar('Loss1', LOSS1)
# 		tf.summary.scalar('Loss2', LOSS2)
# 		tf.summary.scalar('both_zeros', both_zeros_ratio)
# 		tf.summary.scalar('both_ones', both_ones_ratio)
# 		tf.summary.scalar('Learning rate', learning_rate)
# 		tf.summary.image('origin', ORI_MASK, max_outputs = 4)
# 		tf.summary.image('result', MASK, max_outputs = 4)
# 		tf.summary.image('result2', MASK2, max_outputs = 4)
# 		tf.summary.image('super_pixels', SP, max_outputs=4)
# 		tf.summary.image('grad_mask', grad_mask, max_outputs=4)
# 		tf.summary.image('s1', S1, max_outputs = 4)
# 		tf.summary.image('s2', S2, max_outputs = 4)
#
# 		merged = tf.summary.merge_all()
# 		writer = tf.summary.FileWriter("logs_bn/", sess.graph)
#
# 		# ** Start Training **
# 		step = 0
# 		num_imgs = source_imgs.shape[0]
#
# 		MN = mask_net('mask_net')
# 		SOURCE1 = tf.placeholder(tf.float32, shape = (BATCH_SIZE, patch_size, patch_size, 1), name = 's1')
# 		SOURCE2 = tf.placeholder(tf.float32, shape = (BATCH_SIZE, patch_size, patch_size, 1), name = 's2')
# 		_, MASK0 = MN.transform(img1 = SOURCE1, img2 = SOURCE2)
#
#
# 		# restore the trained model
# 		vars = [var for var in tf.trainable_variables() if var.name.startswith("mask_net")]
# 		# print(vars)
# 		saver0 = tf.train.Saver(var_list= vars)
# 		saver0.restore(sess, mask_model_path + str(mask_model_num) + '/' + str(mask_model_num) + '.ckpt')
#
#
# 		for epoch in range(EPOCHES):
# 			np.random.shuffle(source_imgs)
# 			for batch in range(n_batches):
# 				step += 1
# 				current_iter = step
# 				index = np.random.choice([0, 1], 1)
# 				batch1 = source_imgs[batch * BATCH_SIZE:(batch * BATCH_SIZE + BATCH_SIZE), :, :, index[0]]
# 				batch1 = np.expand_dims(batch1, -1)
# 				batch2 = source_imgs[batch * BATCH_SIZE:(batch * BATCH_SIZE + BATCH_SIZE), :, :, 1-index[0]]
# 				batch2 = np.expand_dims(batch2, -1)
# 				batchsp = source_imgs[batch * BATCH_SIZE:(batch * BATCH_SIZE + BATCH_SIZE), :, :, 2]
# 				batchsp = np.expand_dims(batchsp, -1)
#
# 				ori_mask = sess.run(MASK0, feed_dict = {SOURCE1: batch1, SOURCE2: batch2})
# 				ori_mask2 = ori_mask[:, :, :, 0]
# 				# remove small regions
# 				for c in range(BATCH_SIZE):
# 					mask_bool1 = (1 - ori_mask2[c, :, :]) > 0
# 					dst1 = morphology.remove_small_objects(mask_bool1, min_size = patch_size * patch_size / 60, connectivity = 1,
# 					                                       in_place = False)
# 					dst1 = dst1.astype(np.float32)
# 					mask_bool2 = (1 - dst1) > 0
# 					dst2 = morphology.remove_small_objects(mask_bool2, min_size = patch_size * patch_size / 60, connectivity = 1,
# 					                                       in_place = False)
# 					dst2 = dst2.astype(np.float32)
# 					dst2 = dst2.reshape([1, patch_size, patch_size, 1])
# 					if c == 0:
# 						ori_mask_remove = dst2
# 					else:
# 						ori_mask_remove = np.concatenate((ori_mask_remove, dst2), axis = 0)
#
#
# 				FEED_DICT = {ORI_MASK: ori_mask_remove, SP: batchsp, S1: batch1, S2: batch2}
#
# 				sess.run([solver, clip], feed_dict = FEED_DICT)
# 				loss, loss1, loss2 = sess.run([LOSS, LOSS1, LOSS2], feed_dict = FEED_DICT)
#
# 				print("Epoch: [%3s], Batch: [%4s]" % (epoch+1, batch+1))
# 				if batch % 10 == 0:
# 					elapsed_time = datetime.now() - start_time
# 					lr = sess.run(learning_rate)
# 					print('loss: %s, batch: %s/%s' % (loss, batch, n_batches))
# 					print('loss1: %s, loss2: %s, w_loss2: %s' % (loss1, loss2, w2*loss2))
# 					print("lr: %s, elapsed_time: %s" % (lr, elapsed_time))
#
# 				result = sess.run(merged, feed_dict = FEED_DICT)
# 				writer.add_summary(result, step)
# 				if step % logging_period == 0:
# 					saver.save(sess, save_path + str(step) + '/' + str(step) + '.ckpt')
#
# 	writer.close()
# 	saver.save(sess, save_path + str(epoch) + '/' + str(epoch) + '.ckpt')

def grad(img):
	kernel = tf.constant([[-1 / 8, -1 / 8, -1 / 8], [-1 / 8, 1, -1 / 8], [-1 / 8, -1 / 8, -1 / 8]])
	kernel = tf.expand_dims(kernel, axis = -1)
	kernel = tf.expand_dims(kernel, axis = -1)
	g = tf.nn.conv2d(img, kernel, strides = [1, 1, 1, 1], padding = 'SAME')
	return g


def fil(img):
	kernel = tf.constant([[-1, -1, -1], [-1, 8.0, -1], [-1, -1, -1]])
	kernel = tf.expand_dims(kernel, axis = -1)
	kernel = tf.expand_dims(kernel, axis = -1)
	filter_img = tf.abs(tf.nn.conv2d(img, kernel, strides = [1, 1, 1, 1], padding = 'SAME'))
	diff_or_same = binary(filter_img - 4.5) / 2 + 0.5
	img_res = diff_or_same * (1 - img) + (1 - diff_or_same) * img
	return img_res


def binary(input):
	x = input
	with tf.get_default_graph().gradient_override_map({"Sign": 'QuantizeGrad'}):
		x = tf.sign(x)
	return x

# def connected_region_number(imgs):
# 	numbers = []
# 	areas=[]
# 	for i in range(int(imgs.shape[0])):
# 		area=[]
# 		img = imgs[i, :, :, 0]
# 		img_down = img[0:-1, :]
# 		img_down = tf.concat([tf.expand_dims(1 - img[0, :], axis = 0), img_down], axis = 0)
#
# 		img_right = img[:, 0:-1]
# 		img_right = tf.concat([tf.expand_dims(1 - img[:, 0], axis = 1), img_right], axis = 1)
#
# 		img_rightdown = img[0:-1, 0:-1]
# 		col0 = tf.expand_dims(1 - img[0:-1, 0], axis = 1)
# 		img_rightdown = tf.concat([col0, img_rightdown], axis = 1)
# 		row0 = tf.expand_dims(tf.concat([tf.expand_dims(1 - img[0, 0], axis = 0), 1 - img[0, 0:-1]], axis = 0),
# 		                      axis = 0)
# 		img_rightdown = tf.concat([row0, img_rightdown], axis = 0)
#
# 		img_rightup = img[1:, 0:-1]
# 		col00 = tf.expand_dims(1 - img[1:, 0], axis = 1)
# 		img_rightup = tf.concat([col00, img_rightup], axis = 1)
# 		row00 = tf.concat(
# 			[tf.expand_dims(1 - img[int(img.shape[0]) - 1, 0], axis = 0), 1 - img[int(img.shape[0]) - 1, 0:-1]],
# 			axis = 0)
# 		row00 = tf.expand_dims(row00, axis = 0)
# 		img_rightup = tf.concat([img_rightup, row00], axis = 0)
#
# 		img_h_diff1 = img - img_down
# 		img_w_diff1 = img - img_right
# 		img_rightdown_diff = img - img_rightdown
# 		img_rightup_diff = img - img_rightup
#
# 		diff = (tf.abs(img_h_diff1) + tf.abs(img_w_diff1)+ tf.abs(img_rightdown_diff) + tf.abs(img_rightup_diff))/4
#
# 		diffs = binary(diff-0.9)
# 		diffs = diffs / 2 + 0.5
# 		batch_labels = tf.reduce_sum(diffs, axis = [0, 1])
# 		numbers.append(batch_labels)
#
# 		# diffs = tf.expand_dims(diffs, axis = 0)
# 		# diffs = tf.expand_dims(diffs, axis = -1)
# 		# if i == 0:
# 		# 	D = diffs
# 		# else:
# 		# 	D = tf.concat([D, diffs], axis = 0)
# 	return numbers




# def SSIM_LOSS(img1, img2, size = 11, sigma = 1.5):
# 	window = _tf_fspecial_gauss(size, sigma)  # window shape [size, size]
# 	k1 = 0.01
# 	k2 = 0.03
# 	L = 1  # depth of image (255 in case the image has a different scale)
# 	c1 = (k1 * L) ** 2
# 	c2 = (k2 * L) ** 2
# 	mu1 = tf.nn.conv2d(img1, window, strides = [1, 1, 1, 1], padding = 'VALID')
# 	mu2 = tf.nn.conv2d(img2, window, strides = [1, 1, 1, 1], padding = 'VALID')
# 	mu1_sq = mu1 * mu1
# 	mu2_sq = mu2 * mu2
# 	mu1_mu2 = mu1 * mu2
# 	sigma1_sq = tf.nn.conv2d(img1 * img1, window, strides = [1, 1, 1, 1], padding = 'VALID') - mu1_sq
# 	sigma2_sq = tf.nn.conv2d(img2 * img2, window, strides = [1, 1, 1, 1], padding = 'VALID') - mu2_sq
# 	sigma1_2 = tf.nn.conv2d(img1 * img2, window, strides = [1, 1, 1, 1], padding = 'VALID') - mu1_mu2
#
# 	# value = (2.0 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)
# 	ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma1_2 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
# 	value = tf.reduce_mean(ssim_map)
# 	return value
#
# def _tf_fspecial_gauss(size, sigma):
# 	"""Function to mimic the 'fspecial' gaussian MATLAB function
# 	"""
# 	x_data, y_data = np.mgrid[-size // 2 + 1:size // 2 + 1, -size // 2 + 1:size // 2 + 1]
#
# 	x_data = np.expand_dims(x_data, axis = -1)
# 	x_data = np.expand_dims(x_data, axis = -1)
#
# 	y_data = np.expand_dims(y_data, axis = -1)
# 	y_data = np.expand_dims(y_data, axis = -1)
#
# 	x = tf.constant(x_data, dtype = tf.float32)
# 	y = tf.constant(y_data, dtype = tf.float32)
#
# 	g = tf.exp(-((x ** 2 + y ** 2) / (2.0 * sigma ** 2)))
# 	return g / tf.reduce_sum(g)
