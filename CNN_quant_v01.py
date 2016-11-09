#!/usr/bin/env python
# -*- coding:utf-8 -*-

#import input_data for extracting MNIST_data
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import time

#initialize weight
#add small noize for avoiding vanishing gradient
def weight_variable(shape, name):
	initial = tf.truncated_normal(shape,stddev=0.1)
	return tf.Variable(initial, name=name)

#initialize bias to constant 0.1
def bias_variable(shape, name):
	initial = tf.constant(0.1, shape=shape)
	return tf.Variable(initial, name=name)

#define convolution layer
#stride=2,padding=SAME
def conv2d(x, W):
	return tf.nn.conv2d(x,W,strides=[1,2,2,1], padding='VALID')

#define activation function h(x)=min(1,|x|) for quantization
def cabs(x):
	return tf.minimum(1.0, tf.abs(x), name='cabs')

#quantizaion activations
def activate(x, fa):
	return fa(cabs(x))

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


start = time.time()

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

x = tf.placeholder("float", shape=[None, 784])
y_ = tf.placeholder("float", shape=[None, 10])

"""first layer"""
#kernel size=4, input_depth=1, output_depth=5
W_conv1 = weight_variable([4,4,1,5], "W_conv1")
b_conv1 = bias_variable([5], "b_conv1")

#change image to 28*28 Monochrome image
x_image = tf.reshape(x, [-1,28,28,1])

#apply convolution, activation function and quantization activation
h_conv1_act_bi = activate(conv2d(x_image, W_conv1) + b_conv1, fa)

"""second layer"""
#kernel size=3, input_depth=5, output_depth=16
W_conv2 = weight_variable([3,3,5,16], "W_conv2")

#apply quantization weight
W_conv2_w_bi = fw(W_conv2)

#apply quantization gradient
W_conv2_w_grad_bi = fg(W_conv2_w_bi)
b_conv2 = bias_variable([16], "b_conv2")

#apply convolution, activation function and quantization activation
h_conv2_act_bi = activate(conv2d(h_conv1_act_bi, W_conv2_w_grad_bi) + b_conv2, fa)

"""first FC-layer"""
#node=200
W_fc1 = weight_variable([6*6*16, 200], "W_fc1")

#apply quantization gradient
W_fc1_grad_bi = fg(W_fc1)
b_fc1 = bias_variable([200], "b_fc1")

#flatten matrix for matmul caluculation
h_conv_flat = tf.reshape(h_conv2_act_bi, [-1, 6*6*16])

#apply cabs activation function
h_fc1 = cabs(tf.matmul(h_conv_flat, W_fc1_grad_bi) + b_fc1)

"""second FC-layer"""
#node=10
W_fc2 = weight_variable([200, 10], "W_fc2")

#apply quantization gradient
W_fc2_grad_bi = fg(W_fc2)
b_fc2 = bias_variable([10], "b_fc2")

"""softmax"""
y = tf.nn.softmax(tf.matmul(h_fc1, W_fc2_grad_bi) + b_fc2)

cross_entropy = -tf.reduce_sum(y_ * tf.log(y))

#optimize objective function by minimizing cross_entropy(lr=0.001)
train_step = tf.train.GradientDescentOptimizer(0.001).minimize(cross_entropy)

#caluculation of accuracy
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

tf.initialize_all_variables().run()
saver = tf.train.Saver()

for i in range(10000):
	batch_xs, batch_ys = mnist.train.next_batch(100)
	if i % 100 == 0:
		train_accuracy = accuracy.eval(feed_dict={x:batch_xs, y_:batch_ys})
		print "step %d train_accuracy %g" % (i, train_accuracy)
	train_step.run(feed_dict={x: batch_xs, y_:batch_ys})
	#if train_accuracy == 1:
		#break

print "test_accuracy %g" % accuracy.eval(feed_dict={x:mnist.test.images, y_:mnist.test.labels})

elapsed_time = time.time() - start
print "elapsed_time: %f sec" % elapsed_time

saver.save(sess, "model.npy")
