#!/usr/bin/env python
# -*- coding:utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

#import input_data for extracting MNIST_data
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf


flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_boolean('fake_data', False, 'If true, uses fake data '
                     'for unit testing.')
flags.DEFINE_integer('max_steps', 101, 'Number of steps to run trainer.')
flags.DEFINE_float('learning_rate', 0.001, 'Initial learning rate.')
flags.DEFINE_float('dropout', 0.9, 'Keep probability for training dropout.')
flags.DEFINE_string('data_dir', '/tmp/data', 'Directory for storing data')
flags.DEFINE_string('summaries_dir', '/tmp/mnist_logs', 'Summaries directory')

#initialize weight
#add small noize for avoiding vanishing gradient
def weight_variable(shape):
	initial = tf.truncated_normal(shape,stddev=0.1)
	return tf.Variable(initial)

#initialize bias to constant 0.1
def bias_variable(shape):
	initial = tf.constant(0.1, shape=shape)
	return tf.Variable(initial)

#define convolution layer
#stride=2,padding=SAME
def conv2d(x, W):
	return tf.nn.conv2d(x,W,strides=[1,2,2,1], padding='SAME')

#define activation function h(x)=min(1,|x|) for quantization
def cabs(x):
	return tf.minimum(1.0, tf.abs(x), name='cabs')

#quantizaion activations
def activate(x, fa):
	return fa(cabs(x))


def variable_summaries(var, name):
	"""Attach a lot of summaries to a Tensor."""
	with tf.name_scope('summaries'):
		mean = tf.reduce_mean(var)
		tf.scalar_summary('mean/' + name, mean)
		with tf.name_scope('stddev'):
			stddev = tf.sqrt(tf.reduce_sum(tf.square(var - mean)))
		tf.scalar_summary('sttdev/' + name, stddev)
		tf.scalar_summary('max/' + name, tf.reduce_max(var))
		tf.scalar_summary('min/' + name, tf.reduce_min(var))
		tf.histogram_summary(name, var)

#define quantization of weight, activation and grdient
def get_dorefa(bitW, bitA, bitG):
	""" return the three quantization functions fw, fa, fg, for weights,
	activations and gradients respectively"""

	G = tf.get_default_graph()

	def quantize(x, k):
		n = float(2**k-1)
		with G.gradient_override_map({"Floor": "Identity"}):
			return tf.round(x*n)/n

	def fw(x):
		if bitW == 32:
			return x
		if bitW == 1:
			with G.gradient_override_map({"Sign": "Identity"}):
				E = tf.stop_gradient(tf.reduce_mean(tf.abs(x)))
				return tf.sign(x/E)*E
		x = tf.tanh(x)
		x = x/tf.reduce_max(tf.abs(x))*0.5+0.5
		return 2*quantize(x, bitW) - 1

	def fa(x):
		if bitA == 32:
			return x
		return quantize(x, bitA)

	global GRAD_DEFINED

	#for setting the grad_fg node at only one time
	if not GRAD_DEFINED:
		@tf.RegisterGradient("FGGrad")

		#in the back-propagate, quantization gradient
		def grad_fg(op, x):
			rank = x.get_shape().ndims
			assert rank is not None
			maxx = tf.reduce_max(tf.abs(x), list(range(1,rank)), keep_dims=True)
			x = x / maxx
			n = float(2**bitG-1)
			x = x * 0.5 + 0.5 + tf.random_uniform(
					tf.shape(x), minval=-0.5/n, maxval=0.5/n)
			x = tf.clip_by_value(x, 0.0, 1.0)
			x = quantize(x, bitG) - 0.5
			return x * maxx * 2
	GRAD_DEFINED = True

	def fg(x):
		if bitG == 32:
			return x
		with G.gradient_override_map({"Identity": "FGGrad"}):
			return tf.identity(x)
	return fw, fa, fg



#extracting MNIST_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

sess = tf.InteractiveSession()

#set each bitwidth(BITW:weight, BITA:activation, BITG:gradient)
BITW = 1
BITA = 2
BITG = 6

#for setting the grad_fg node at only one time
GRAD_DEFINED = False

fw, fa, fg = get_dorefa(BITW, BITA, BITG)

with tf.name_scope("input"):
	x = tf.placeholder("float", shape=[None, 784], name="x_input")
	y_ = tf.placeholder("float", shape=[None, 10], name="y_input")

"""first layer"""
#kernel size=4, input_depth=1, output_depth=5
with tf.name_scope('W_conv1'):
	W_conv1 = weight_variable([4,4,1,5])
	variable_summaries(W_conv1, '/W_conv1')
with tf.name_scope('b_conv1'):
	b_conv1 = bias_variable([5])
	variable_summaries(b_conv1, '/b_conv1')

#change image to 28*28 Monochrome image
with tf.name_scope("input_reshape"):
	x_image = tf.reshape(x, [-1,28,28,1])
	tf.image_summary("input", x_image, 10)

#apply convolution, activation function and quantization activation
with tf.name_scope('h_conv1_act_bi'):
	h_conv1_act_bi = activate(conv2d(x_image, W_conv1) + b_conv1, fa)
	variable_summaries(h_conv1_act_bi, '/h_conv1_act_bi')

"""second layer"""
#kernel size=3, input_depth=5, output_depth=16
with tf.name_scope('W_conv2'):
	W_conv2 = weight_variable([3,3,5,16])
	variable_summaries(W_conv2, '/W_conv2')

#apply quantization weight
with tf.name_scope('W_conv2_w_bi'):
	W_conv2_w_bi = fw(W_conv2)
	variable_summaries(W_conv2_w_bi, '/W_conv2_w_bi')

#apply quantization gradient
with tf.name_scope('W_conv2_grad_w_bi'):
	W_conv2_w_grad_bi = fg(W_conv2_w_bi)
	variable_summaries(W_conv2_w_grad_bi, '/W_conv2_grad_w_bi')

with tf.name_scope('b_conv2'):
	b_conv2 = bias_variable([16])
	variable_summaries(b_conv2, '/b_conv2')

#apply convolution, activation function and quantization activation
with tf.name_scope('h_conv2_act_bi'):
	h_conv2_act_bi = activate(conv2d(h_conv1_act_bi, W_conv2_w_grad_bi) + b_conv2, fa)
	variable_summaries(h_conv2_act_bi, '/h_conv2_act_bi')

"""first FC-layer"""
#node=200
with tf.name_scope('W_fc1'):
	W_fc1 = weight_variable([7*7*16, 200])
	variable_summaries(W_fc1, '/W_fc1')

#apply quantization gradient
with tf.name_scope('W_fc1_grad_bi'):
	W_fc1_grad_bi = fg(W_fc1)
	variable_summaries(W_fc1_grad_bi, '/W_fc1_grad_bi')

with tf.name_scope('b_fc1'):
	b_fc1 = bias_variable([200])
	variable_summaries(b_fc1, '/b_fc1')

#flatten matrix for matmul caluculation
with tf.name_scope('h_conv_flat'):
	h_conv_flat = tf.reshape(h_conv2_act_bi, [-1, 7*7*16])
	variable_summaries(h_conv_flat, '/h_conv_flat')

#apply relu activation function
with tf.name_scope('h_fc1'):
	h_fc1 = tf.nn.relu(tf.matmul(h_conv_flat, W_fc1_grad_bi) + b_fc1)
	variable_summaries(h_fc1, '/h_fc1')

"""second FC-layer"""
#node=10
with tf.name_scope('W_fc2'):
	W_fc2 = weight_variable([200, 10])
	variable_summaries(W_fc2, '/W_fc2')

#apply quantization gradient
with tf.name_scope('W_fc2_grad_bi'):
	W_fc2_grad_bi = fg(W_fc2)
	variable_summaries(W_fc2_grad_bi, '/W_fc2_grad_bi')

with tf.name_scope('b_fc2'):
	b_fc2 = bias_variable([10])
	variable_summaries(b_fc2, '/b_fc2')

"""softmax"""
with tf.name_scope("Wx_b") as scope:
	y = tf.nn.softmax(tf.matmul(h_fc1, W_fc2_grad_bi) + b_fc2)
	variable_summaries(y, '/Wx_b')

with tf.name_scope('cross_entropy'):
	cross_entropy = -tf.reduce_sum(y_ * tf.log(y))
	tf.scalar_summary('cross entropy', cross_entropy)

#optimize objective function by minimizing cross_entropy(lr=0.001)
with tf.name_scope("train") as scope:
	train_step = tf.train.GradientDescentOptimizer(0.001).minimize(cross_entropy)

#caluculation of
with tf.name_scope('accuracy'):
	with tf.name_scope('correct_prediction'):
		correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
	with tf.name_scope('accuracy'):
		accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
	tf.scalar_summary('accuracy', accuracy)

merged = tf.merge_all_summaries()
writer = tf.train.SummaryWriter("/tmp/mnist_logs", graph=sess.graph_def)

tf.initialize_all_variables().run()

for i in range(1000):
	batch_xs, batch_ys = mnist.train.next_batch(100)
	if i % 100 == 0:
		train_accuracy = accuracy.eval(feed_dict={x:batch_xs, y_:batch_ys})
		writer.add_summary(train_accuracy, i)
		print "step %d train_accuracy %g" % (i, train_accuracy)
	train_step.run(feed_dict={x: batch_xs, y_:batch_ys})

print "test_accuracy %g" % accuracy.eval(feed_dict={x:mnist.test.images, y_:mnist.test.labels})


