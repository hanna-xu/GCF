import tensorflow as tf
from tensorflow.python import pywrap_tensorflow
import numpy as np

WEIGHT_INIT_STDDEV = 0.1


class mask_net(object):

	def __init__(self, sco):
		self.vars=[]
		self.encoder = Encoder(sco)
		self.decoder = Decoder(sco)

	def transform(self, img1, img2):
		img = tf.concat([img1, img2], 3)
		code = self.encoder.encode(img)
		self.target_features = code
		map, mask = self.decoder.decode(self.target_features)
		return (map, mask)


class Encoder(object):
	def __init__(self, scope_name):
		self.scope = scope_name
		self.weight_vars = []
		with tf.variable_scope(self.scope):
			with tf.variable_scope('encoder'):
				self.weight_vars.append(self._create_variables(2, 48, 3, scope = 'conv1_1'))
				self.weight_vars.append(self._create_variables(48, 48, 3, scope = 'dense_block_conv1'))
				self.weight_vars.append(self._create_variables(96, 48, 3, scope = 'dense_block_conv2'))
				self.weight_vars.append(self._create_variables(144, 48, 3, scope = 'dense_block_conv3'))

				# self.weight_vars.append(self._create_variables(192, 48, 3, scope = 'dense_block_conv4'))
				# self.weight_vars.append(self._create_variables(240, 48, 3, scope = 'dense_block_conv5'))


	def _create_variables(self, input_filters, output_filters, kernel_size, scope):
		shape = [kernel_size, kernel_size, input_filters, output_filters]
		with tf.variable_scope(scope):
			kernel = tf.Variable(tf.truncated_normal(shape, stddev = WEIGHT_INIT_STDDEV), name = 'kernel')
			bias = tf.Variable(tf.zeros([output_filters]), name = 'bias')
		return (kernel, bias)

	def encode(self, image):
		dense_indices = [1, 2, 3, 4, 5]
		out = image
		for i in range(len(self.weight_vars)):
			kernel, bias = self.weight_vars[i]

			if i in dense_indices:
				out = conv2d(out, kernel, bias, dense = True, use_lrelu = True, Scope = self.scope + '/encoder/b' + str(i))
			else:
				out = conv2d(out, kernel, bias, dense = False, use_lrelu = True, Scope = self.scope + '/encoder/b' + str(i))
		return out


class Decoder(object):
	def __init__(self, scope_name):
		self.weight_vars = []
		self.scope = scope_name
		with tf.name_scope(scope_name):
			with tf.variable_scope('decoder'):
				self.weight_vars.append(self._create_variables(192, 192, 3, scope = 'conv2_1'))
				self.weight_vars.append(self._create_variables(192, 128, 3, scope = 'conv2_2'))
				self.weight_vars.append(self._create_variables(128, 64, 3, scope = 'conv2_3'))
				self.weight_vars.append(self._create_variables(64, 1, 3, scope = 'conv2_4'))
				self.weight_vars.append(self._create_variables(1, 1, 3, scope = 'conv2_5'))
				# self.weight_vars.append(tf.Variable(tf.ones([1]), name='thre'))
				# self.weight_vars.append(self._create_variables(16, 1, 3, scope = 'conv2_6'))
				# self.weight_vars.append(self._create_variables(32, 1, 3, scope = 'conv2_4'))

	# self.weight_vars.append(self._create_variables(16, 1, 3, scope = 'conv2_5'))

	def _create_variables(self, input_filters, output_filters, kernel_size, scope):
		with tf.variable_scope(scope):
			shape = [kernel_size, kernel_size, input_filters, output_filters]
			kernel = tf.Variable(tf.truncated_normal(shape, stddev = WEIGHT_INIT_STDDEV), name = 'kernel')
			bias = tf.Variable(tf.zeros([output_filters]), name = 'bias')
		return (kernel, bias)

	def decode(self, image):
		final_layer_idx = len(self.weight_vars) - 1

		out = image
		for i in range(len(self.weight_vars)):
			kernel, bias = self.weight_vars[i]
			# if i == 0:
			# 	out = conv2d(out, kernel, bias, dense = False, use_lrelu = True,
			# 	             Scope = self.scope + '/decoder/b' + str(i), BN = False)
			# if i == final_layer_idx - 1:
			# 			# 	out = conv2d(out, kernel, bias, dense = False, use_lrelu= True, BN = True,
			# 			# 	             Scope = self.scope + '/decoder/b' + str(i))
			# 			# 	print("i,", i)
			if i == final_layer_idx:
				out = conv2d(out, kernel, bias, dense = False, use_lrelu = False,
				             Scope = self.scope + '/decoder/b' + str(i), BN = True)
				out = tf.nn.tanh(out) / 2 + 0.5
				# threshold = tf.zeros_like(out)
				# condition=tf.less(out, threshold)
				# ones=tf.ones_like(out)
				# zeros = tf.zeros_like(out)
				# out = tf.where(condition, zeros, ones)
				# th=self.weight_vars[len(self.weight_vars)-1]

				mask = binary(out-0.5)
				mask = mask / 2 + 0.5
			else:
				out = conv2d(out, kernel, bias, dense = False, use_lrelu = True, BN = True,
				             Scope = self.scope + '/decoder/b' + str(i))
		return out, mask


def conv2d(x, kernel, bias, dense = False, use_lrelu = True, Scope = None, BN = True):
	# padding image with reflection mode
	x_padded = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], mode = 'REFLECT')
	# conv and add bias
	out = tf.nn.conv2d(x_padded, kernel, strides = [1, 1, 1, 1], padding = 'VALID')
	out = tf.nn.bias_add(out, bias)
	if BN:
		with tf.variable_scope(Scope):
			out = tf.layers.batch_normalization(out, training = True)
	if use_lrelu:
		out = tf.maximum(out, 0.2 * out)#tf.nn.relu(out)
	if dense:
		out = tf.concat([out, x], 3)
	return out


@tf.RegisterGradient("QuantizeGrad")
def sign_grad(op, grad):
	input = op.inputs[0]
	cond = (input >= -1) & (input <= 1)
	zeros = tf.zeros_like(grad)
	return tf.where(cond, grad, zeros)

def binary(input):
	x = input
	with tf.get_default_graph().gradient_override_map({"Sign": 'QuantizeGrad'}):
		x = tf.sign(x)
	return x