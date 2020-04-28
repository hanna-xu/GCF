import tensorflow as tf
from tensorflow.python import pywrap_tensorflow
import numpy as np

WEIGHT_INIT_STDDEV = 0.1


class boundary_net(object):
	def __init__(self, sco):
		self.net = Net(sco)

	def transform(self, i, origin_mask):
		input = tf.concat([i, origin_mask], 3)
		output = self.net.process(input)
		return output


class Net(object):
	def __init__(self, scope_name):
		self.weight_vars = []
		self.scope = scope_name
		with tf.name_scope(scope_name):
			with tf.variable_scope('decoder'):
				self.weight_vars.append(self._create_variables(2, 32, 3, scope = 'conv2_1'))
				self.weight_vars.append(self._create_variables(32, 64, 3, scope = 'conv2_2'))
				self.weight_vars.append(self._create_variables(64, 128, 3, scope = 'conv2_3'))
				self.weight_vars.append(self._create_variables(128, 64, 3, scope = 'conv2_4'))
				self.weight_vars.append(self._create_variables(64, 1, 3, scope = 'conv2_5'))


	def _create_variables(self, input_filters, output_filters, kernel_size, scope):
		with tf.variable_scope(scope):
			shape = [kernel_size, kernel_size, input_filters, output_filters]
			kernel = tf.Variable(tf.truncated_normal(shape, stddev = WEIGHT_INIT_STDDEV), name = 'kernel')
			bias = tf.Variable(tf.zeros([output_filters]), name = 'bias')
		return (kernel, bias)

	def process(self, input):
		final_layer_idx = len(self.weight_vars) - 1

		out = input
		for i in range(len(self.weight_vars)):
			kernel, bias = self.weight_vars[i]
			if i == final_layer_idx:
				out = conv2d(out, kernel, bias, dense = False, use_lrelu = False,
				             Scope = self.scope + '/net/b' + str(i), BN = True)
				out = tf.nn.tanh(out) / 2 + 0.5
				mask = binary(out-0.5)
				mask = mask / 2 + 0.5
			else:
				out = conv2d(out, kernel, bias, dense = False, use_lrelu = True, BN = True,
				             Scope = self.scope + '/net/b' + str(i))
		return mask


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
		out = tf.maximum(out, 0.2 * out) #tf.nn.relu(out)
	if dense:
		out = tf.concat([out, x], 3)
	return out


# @tf.RegisterGradient("QuantizeGrad")
# def sign_grad(op, grad):
# 	input = op.inputs[0]
# 	cond = (input >= -1) & (input <= 1)
# 	zeros = tf.zeros_like(grad)
# 	return tf.where(cond, grad, zeros)
#
def binary(input):
	x = input
	with tf.get_default_graph().gradient_override_map({"Sign": 'QuantizeGrad'}):
		x = tf.sign(x)
	return x