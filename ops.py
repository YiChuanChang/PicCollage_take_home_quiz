import tensorflow as tf
import tensorflow.contrib as tf_contrib
import tensorflow.contrib.slim as slim

weight_init = tf_contrib.layers.variance_scaling_initializer() # kaming init for encoder / decoder
weight_regularizer = tf_contrib.layers.l2_regularizer(scale=0.0001)

def batch_norm(x, scope=None):
	return tf.contrib.layers.batch_norm(x, scope=scope)

##################################################################################
# Activation function
##################################################################################
def relu(x):
	return tf.nn.relu(x)

def tanh(x):
	return tf.tanh(x)

##################################################################################
# Layer
##################################################################################
def conv(x, channels, kernel=5, stride=2, use_bias=True, scope='conv'):
	with tf.variable_scope(scope):
		weight_init = tf_contrib.layers.variance_scaling_initializer()

		x = tf.layers.conv2d(inputs=x, filters=channels,
								kernel_size=kernel, kernel_initializer=weight_init,
								kernel_regularizer=weight_regularizer,
								strides=stride, use_bias=use_bias)

		return x

def linear(x, units, use_bias=True, scope='linear'):
    with tf.variable_scope(scope):
        x = flatten(x)
        x = tf.layers.dense(x, units=units, kernel_initializer=weight_init, kernel_regularizer=weight_regularizer, use_bias=use_bias)

        return x

def flatten(x) :
    return tf.layers.flatten(x)

