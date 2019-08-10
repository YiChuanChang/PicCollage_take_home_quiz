import csv
import tensorflow as tf
import tensorflow.contrib.slim as slim
import random
import os
from ops import *

batch_size = 64
total_epoch = 51
checkpoint_dir = 'checkpoint'

def load_image(image_path, label):
	image_buffer = tf.read_file(image_path)
	image = tf.image.decode_png(image_buffer, channels=3)
	image = tf.image.resize_images(image, [128, 128])
	image = tf.cast(image, tf.float32)
	return image, label, image_path

def network(image, scope='network', reuse=None):

	with tf.variable_scope(scope, reuse=reuse):
		x = conv(image, channels=8, kernel=5, stride=2, scope='conv_0')
		x = batch_norm(x, scope='batch_norm_0')
		x = relu(x)
		print(x)

		x = conv(x, channels=16, kernel=5, stride=2, scope='conv_1')
		x = batch_norm(x, scope='batch_norm_1')
		x = relu(x)
		print(x)

		x = conv(x, channels=32, kernel=5, stride=2, scope='conv_2')
		x = batch_norm(x, scope='batch_norm_2')
		x = relu(x)
		print(x)

		x = conv(x, channels=64, kernel=5, stride=2, scope='conv_3')
		x = batch_norm(x, scope='batch_norm_3')
		x = relu(x)
		print(x)

		x = conv(x, channels=128, kernel=5, stride=2, scope='conv_4')
		x = batch_norm(x, scope='batch_norm_4')
		x = relu(x)
		print(x)

		x = linear(x, units=256, use_bias=True, scope='linear_0')
		x = batch_norm(x, scope='batch_norm_5')
		x = relu(x)
		print(x)

		x = linear(x, units=1, use_bias=True, scope='linear_1')
		x = tanh(x)
		print(x)

	return x


graph = tf.Graph()
with graph.as_default():
	file_names = tf.placeholder(dtype = tf.string, shape=(None,))
	labels = tf.placeholder(dtype = tf.float32, shape=(None,))
	# prepocessing data and set them into batches
	sliced_data = tf.data.Dataset.from_tensor_slices((file_names, labels))
	data = sliced_data.map(lambda file_name, label: load_image(file_name, label))
	# data = data.shuffle(buffer_size=100000)
	batched_data = data.batch(batch_size)
	iterator = tf.data.Iterator.from_structure(batched_data.output_types,
                                                       batched_data.output_shapes)
	batch_images, batch_labels, batch_paths = iterator.get_next()
	dataset_initialize = iterator.make_initializer(batched_data)
	
	batch_predictions = network(batch_images)
	batch_predictions = tf.reshape(batch_predictions, [-1])
	tf.losses.mean_squared_error(labels=batch_labels, predictions=batch_predictions)
	loss = tf.losses.get_total_loss()
	train_loss_s = tf.summary.scalar("train_loss", loss)
	test_loss_s = tf.summary.scalar("test_loss", loss)
	train_loss_merge = tf.summary.merge([train_loss_s])
	test_loss_merge = tf.summary.merge([test_loss_s])

	# using ADAM optimizer
	optimizer_1 = tf.train.AdamOptimizer(learning_rate=0.001)
	optimizer_2 = tf.train.AdamOptimizer(learning_rate=0.0001)
	train_op_1 = optimizer_1.minimize(loss)
	train_op_2 = optimizer_2.minimize(loss)

	init = tf.global_variables_initializer()

	
def train(sess, train_set, train_label, writer, epoch):
	sess.run(dataset_initialize, feed_dict={file_names: train_set,
											labels: train_label})
	print('############## Training ############## ')
	i = 0
	average_loss = 0
	while True:
		try:
			i = i+1
			# training
			if(epoch<=14):
				loss_, train_loss_merge_, _ = sess.run([loss, train_loss_merge, train_op_1])  
			else:
				loss_, train_loss_merge_, _ = sess.run([loss, train_loss_merge, train_op_2])  

			writer.add_summary(train_loss_merge_, epoch*(int(105000/batch_size)+1)+i)
			average_loss += loss_  
			if(i%100==0):	
				print('data_batch: ' + str(i))
				print(loss_)
		except tf.errors.OutOfRangeError:
			break
	average_loss = average_loss/i
	with open("train_log.txt", "a") as train_log:
		train_log.write('Epoch '+str(epoch)+' ,loss: '+ str(average_loss)+'\n')

def test(sess, test_set, test_label, writer, epoch):
	sess.run(dataset_initialize, feed_dict={file_names: test_set,
											labels: test_label})

	print('############## Testing ############## ')
	i = 0
	average_loss = 0
	while True:
		try:
			i = i+1
			loss_, test_loss_merge_, labels_, paths_, prediction_ = sess.run([loss, test_loss_merge, batch_labels, batch_paths, batch_predictions])   
			writer.add_summary(test_loss_merge_, (epoch/5)*(int(45000/batch_size)+1)+i)
			average_loss += loss_
			if(i%100==0):
				print('data_batch: ' + str(i))
				print(loss_)
				with open("test_prediction.txt", "a") as test_prediction:
					test_prediction.write('\n\n')
					test_prediction.write('Epoch '+str(epoch)+'##################\n')
					for j in range(len(labels_)):
						test_prediction.write(str(paths_[j])+', gt:'+str(labels_[j])+', prediction:'+str(prediction_[j])+'\n')
					# print(str(paths_[i])+', gt:'+str(labels_[i])+', prediction:'+str(prediction_[i]))
		except tf.errors.OutOfRangeError:
			break
	average_loss = average_loss/i
	with open("test_log.txt", "a") as test_log:
		test_log.write('Epoch '+str(epoch)+' ,loss: '+ str(average_loss)+'\n')

def shuffle(path, label):
	tmp = list(zip(path, label))
	random.shuffle(tmp)
	path, label = zip(*tmp)

	return path, label

def define_train_n_test_set(response_csv):
	train_set = []
	train_label = []
	test_set = []
	test_label = []
	for i in range(len(response_csv)):
		if(i%10<3):
			test_set.append(response_csv[i][0])
			test_label.append(response_csv[i][1])
		else:
			train_set.append(response_csv[i][0])
			train_label.append(response_csv[i][1])
	train_set, train_label = shuffle(train_set, train_label)
	test_set, test_label = shuffle(test_set, test_label)

	print('Training data number: '+str(len(train_set)))
	print('Testing data number: '+str(len(test_set)))
	return train_set, train_label, test_set, test_label

def read_csv_file(filename):
	with open(filename, newline='') as csvfile:
		reponses_csv = csv.reader(csvfile, delimiter=',')
		reponses_csv = list(reponses_csv)
		reponses_csv = reponses_csv[1:]
		reponses_csv = sorted(reponses_csv, key=lambda row: float(row[1]), reverse=True)

		for row in reponses_csv:
			row[0] = 'train_imgs/'+row[0]+'.png'
			row[1] = float(row[1])
	return reponses_csv


def main():
	# read in csv file
	reponses_csv = read_csv_file('train_responses.csv')

	# 30% of data for testing, 70% of data for training
	train_set, train_label, test_set, test_label = define_train_n_test_set(reponses_csv)

	with tf.Session(graph=graph) as sess:
		# initialze variables
		sess.run(init)
		saver = tf.train.Saver() # saver to save model
		writer = tf.summary.FileWriter("TensorBoard/", graph = sess.graph) # for loss visualization

		# count variable numbers
		total_parameters = 0
		for variable in tf.trainable_variables(): # iterate all variables
			local_parameters=1
			shape = variable.get_shape()  #getting shape of a variable
			print(shape)
			print(variable)
			for i in shape:
				local_parameters*=i.value  #mutiplying dimension values
			total_parameters+=local_parameters
		print('Total Variable Numbers:'+ str(total_parameters))

		test(sess, test_set, test_label, writer, epoch=0)
		# Train the network
		for epoch in range(total_epoch):
			print('Now in epoch '+str(epoch))
			train(sess, train_set, train_label, writer, epoch)
			if(epoch%5==0 and epoch!=0):
				test(sess, test_set, test_label, writer, epoch)
				saver.save(sess, os.path.join(checkpoint_dir, 'correlation.model'), global_step=epoch)


if __name__ == '__main__':
	main()

