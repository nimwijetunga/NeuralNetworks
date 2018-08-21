import tensorflow as tf 
from image_preprocessing import get_data_sets_and_labels, preprocess_image
import numpy as np
from PIL import Image
import os
import sys

train_x, train_y, test_x, test_y = get_data_sets_and_labels()
print(len(train_x[0]))

#hidden layers in NN
n_nodes_hl1 = 500
n_nodes_hl2 = 500
n_nodes_hl3 = 500


n_classes = 3
batch_size = 30

#height * width 
x = tf.placeholder('float', [None, len(train_x[0])]);  #our data (28px x 28px x 3(RGB) = 1728px) -> we feed the NN pixel by pixel
y = tf.placeholder('float', [None, 3]); #label for data (the expected value for handwriting i.e 0-9 in one hot)

hidden_1_layer = {'weights': tf.Variable(tf.random_normal([len(train_x[0]), n_nodes_hl1])),
'biases':tf.Variable(tf.random_normal([n_nodes_hl1]))}

hidden_2_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
'biases':tf.Variable(tf.random_normal([n_nodes_hl2]))}

hidden_3_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
'biases':tf.Variable(tf.random_normal([n_nodes_hl3]))}

output_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])),
'biases':tf.Variable(tf.random_normal([n_classes]))}

def neural_network_model(data):
	#layers
	# (input_data * weights) + biases 
	#if all input_data = 0 then neurons would not fire so add bias

	l1 = tf.add(tf.matmul(data, hidden_1_layer['weights']), hidden_1_layer['biases'])
	l1 = tf.nn.relu(l1) #activation func relu (sigmoid fn)

	l2 = tf.add(tf.matmul(l1, hidden_2_layer['weights']), hidden_2_layer['biases'])
	l2 = tf.nn.relu(l2)

	l3 = tf.add(tf.matmul(l2, hidden_3_layer['weights']), hidden_3_layer['biases'])
	l3 = tf.nn.relu(l3)

	#output layer
	output = tf.add(tf.matmul(l3, output_layer['weights']), output_layer['biases'])

	return output


def train_neural_network(x):
	prediction = neural_network_model(x)
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction,labels=y))
	optimizer = tf.train.AdamOptimizer().minimize(cost);

	hm_epochs = 10 #how many iterations (feed forward -> backprop)

	saver=tf.train.Saver()
	with tf.Session() as sess:
		sess.run(tf.initialize_all_variables())

		for epoch in range(hm_epochs):
			epoch_loss = 0

			i = 0
			while i < len(train_x):
				start = i
				end = i+batch_size

				batch_x = np.array(train_x[start:end])
				batch_y = np.array(train_y[start:end])

				_, c = sess.run([optimizer, cost], feed_dict = {x: batch_x, y: batch_y}) #run the session which allows for modifications of weights until we get a good model
				epoch_loss += c
				i += batch_size

			print('Epoch', epoch, 'completed out of', hm_epochs, 'loss:', epoch_loss)

		correct = tf.equal(tf.argmax(prediction,1), tf.argmax(y,1)) #tells us wether prediction = actual
		accuracy = tf.reduce_mean(tf.cast(correct,'float'))
		#everyhting above is trainning data
		#accuracy depends on how well the model can predict values for test data
		print('Accuracy:', accuracy.eval({x:test_x, y:test_y}))
		#save trained model
		checkpoint_path = os.path.join('trained_model', 'model.ckpt')
		saver.save(sess, checkpoint_path)

def resize_img(path):
	testDataDir = os.listdir( path )
	for file in testDataDir:
		if os.path.isfile(path+'/'+file):
			im = Image.open(path+'/'+file).convert('RGB')
			head, tail = os.path.split(path+'/'+file)
			imResize = im.resize((24,24), Image.ANTIALIAS)
			imResize.save(head+'_resized/imageset/'+tail, 'JPEG', quality=90)

def get_input_data(image_dir, num_images):	
	resize_img(image_dir)
	if image_dir.endswith('/'):
		image_dir = image_dir[:-1]
	new_path = image_dir + '_resized/imageset/'
	data = preprocess_image(new_path, num_images)
	return np.array(data)

def use_neural_network(input_data):
	prediction = neural_network_model(x)

	saver = tf.train.Saver()

	with tf.Session() as sess:
		sess.run(tf.initialize_all_variables())
		saver.restore(sess,"trained_model/model.ckpt")
		# print(sess.run(prediction, {x:input_data})) gives us an array of 3 numbers [pear,orange,apple]
		output = sess.run(tf.argmax(prediction, 1), {x:input_data}) #fineds the number that is the largest from all the above
		if(output[0] == 0):
			print 'pear'
		elif(output[0] == 1):
			print 'orange'
		elif(output[0] == 2):
			print 'apple'

# train_neural_network(x)

if __name__ == '__main__':
	num_images = sys.argv[1] 
	image_dir = sys.argv[2]  
	data = get_input_data(image_dir, num_images)
	use_neural_network(data)