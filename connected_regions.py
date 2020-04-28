import tensorflow as tf


def connected_regions(mask):
	kernels = []
	kernels2 = []
	kernels.append(
		tf.expand_dims(tf.expand_dims(tf.constant([[-1, 0, 0], [0, 1.0, 0], [0, 0, 0]]), axis = -1), axis = -1))
	kernels.append(
		tf.expand_dims(tf.expand_dims(tf.constant([[0, -1, 0], [0, 1.0, 0], [0, 0, 0]]), axis = -1), axis = -1))
	kernels.append(
		tf.expand_dims(tf.expand_dims(tf.constant([[0, 0, -1], [0, 1.0, 0], [0, 0, 0]]), axis = -1), axis = -1))
	kernels.append(
		tf.expand_dims(tf.expand_dims(tf.constant([[0, 0, 0], [-1, 1.0, 0], [0, 0, 0]]), axis = -1), axis = -1))

	diffs = tf.zeros_like(mask)
	for i in range(4):
		diffs = diffs + tf.abs(tf.nn.conv2d(mask, kernels[i], strides = [1, 1, 1, 1], padding = 'SAME'))
	diffs = tf.tanh(diffs/4)

	diffs = binary(diffs - 0.9) / 2 + 0.5

	numbers = tf.reduce_sum(diffs, axis = [1, 2, 3])

	B, H, W, _ = diffs.get_shape().as_list()
	for i in range(B):
		diff = diffs[i, :, :, 0]
		diff2 = tf.reshape(tf.cast(diff, dtype = tf.float32), [H * W])
		diff3 = tf.cumsum(diff2, axis = 0)
		diff4 = tf.reshape(diff3, [H, W])
		diff4_4c = tf.expand_dims(diff4, axis = -1)
		diff4_4c = tf.expand_dims(diff4_4c, axis = 0)
		if i == 0:
			Diff2 = tf.multiply(diff4_4c, tf.expand_dims(diffs[i, :, :, :], axis = 0))
		else:
			Diff2 = tf.concat([Diff2, tf.multiply(diff4_4c, tf.expand_dims(diffs[i, :, :, :], axis = 0))], axis = 0)

	kernels2.append(
		tf.expand_dims(tf.expand_dims(tf.constant([[1.0, 0, 0], [0, 0, 0], [0, 0, 0]]), axis = -1), axis = -1))
	kernels2.append(
		tf.expand_dims(tf.expand_dims(tf.constant([[0, 1.0, 0], [0, 0, 0], [0, 0, 0]]), axis = -1), axis = -1))
	kernels2.append(
		tf.expand_dims(tf.expand_dims(tf.constant([[0, 0, 1.0], [0, 0, 0], [0, 0, 0]]), axis = -1), axis = -1))
	kernels2.append(
		tf.expand_dims(tf.expand_dims(tf.constant([[0, 0, 0], [1.0, 0, 0], [0, 0, 0]]), axis = -1), axis = -1))
	kernels2.append(
		tf.expand_dims(tf.expand_dims(tf.constant([[0, 0, 0], [0, 0, 1.0], [0, 0, 0]]), axis = -1), axis = -1))
	kernels2.append(
		tf.expand_dims(tf.expand_dims(tf.constant([[0, 0, 0], [0, 0, 0], [1.0, 0, 0]]), axis = -1), axis = -1))
	kernels2.append(
		tf.expand_dims(tf.expand_dims(tf.constant([[0, 0, 0], [0, 0, 0], [0, 1.0, 0]]), axis = -1), axis = -1))
	kernels2.append(
		tf.expand_dims(tf.expand_dims(tf.constant([[0, 0, 0], [0, 0, 0], [0, 0, 1.0]]), axis = -1), axis = -1))

	for k in range(200):
		if k == 0:
			labels = neigh_max(Diff2, mask, kernels2)
		else:
			labels = neigh_max(labels, mask, kernels2)

	# classes = tf.cast(tf.reduce_max(Diff2, axis = [1, 2, 3]), tf.int32)
	# print("classes:", classes)
	class_max = 250
	# area = tf.Variable(tf.zeros([B, class_max], dtype=tf.float32), trainable=False)
	for k in range(class_max):
		sub_region = tf.multiply(tf.sign(labels - (k - 0.1)) / 2 + 0.5, -tf.sign(labels - (k + 0.1)) / 2 + 0.5)
		sub_area = tf.expand_dims(tf.reduce_sum(sub_region, axis = [1, 2, 3]), axis=1)
		if k == 0:
			area = sub_area
		else:
			area = tf.concat([area, sub_area], axis = 1)

	area_min = []
	for i in range(B):
		area_addto0 = - (tf.sign(area[i, :] - 0.1) - 1) * 10000
		area2 = area[i, :] + area_addto0
		area_top_indices = tf.nn.top_k(-area2).indices
		area_min.append(area2[area_top_indices[0]])

	return numbers, Diff2, labels, area_min


def neigh_max(i, mask, kernels):
	for c in range(8):
		if c == 0:
			neighs = tf.nn.conv2d(i, kernels[c], strides = [1, 1, 1, 1], padding = 'SAME')
			neighs_mask = tf.nn.conv2d(mask, kernels[c], strides = [1, 1, 1, 1], padding = 'SAME')
			neighs_with_mask = tf.multiply(neighs, neighs_mask)
		else:
			neighs = tf.nn.conv2d(i, kernels[c], strides = [1, 1, 1, 1], padding = 'SAME')
			neighs_mask = tf.nn.conv2d(mask, kernels[c], strides = [1, 1, 1, 1], padding = 'SAME')
			neighs_with_mask = tf.concat([neighs_with_mask, tf.multiply(neighs, neighs_mask)], axis = -1)

	neighs_max = tf.expand_dims(tf.reduce_max(neighs_with_mask, axis = -1), axis = -1)
	labels = tf.multiply(neighs_max, mask)
	return labels


def binary(input):
	x = input
	with tf.get_default_graph().gradient_override_map({"Sign": 'QuantizeGrad'}):
		x = tf.sign(x)
	return x
