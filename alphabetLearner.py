
from __future__ import print_function

import numpy as np
import tensorflow as tf



training_text_features="abcdefghijklmnopqrstuvwxyz"
training_text_labels="bcdefghijklmnopqrstuvwxyza"

len_of_features=len(training_text_features)

def ret_one_hot(x):
	val=np.zeros([len_of_features])
	val[ord(x)-97]=1.0
	return(val)

training_one_hot_features=[]
training_one_hot_labels=[]

#Converting features to one hot

for i in range(len_of_features):
	training_one_hot_features.append(ret_one_hot(training_text_features[i]))
training_one_hot_features=np.array(training_one_hot_features)

#training_one_hot_features=np.flip(training_one_hot_features,1)

for i in range(len_of_features):
	training_one_hot_labels.append(ret_one_hot(training_text_labels[i]))
training_one_hot_labels=np.array(training_one_hot_labels)

#training_one_hot_labels=np.flip(training_one_hot_labels,1)

print(training_one_hot_features.shape)
print(training_one_hot_labels.shape)

lr = 0.001
training_iters = 100000
batch_size = 1

n_inputs = 26   
n_steps = 1    
n_hidden_units = 1024
n_classes = 26

x = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
y = tf.placeholder(tf.float32, [None, n_classes])

weights = {
    'in': tf.Variable(tf.random_normal([n_inputs, n_hidden_units])),
    'out': tf.Variable(tf.random_normal([n_hidden_units, n_classes]))}
biases = {
    'in': tf.Variable(tf.constant(0.1, shape=[n_hidden_units, ])),
    'out': tf.Variable(tf.constant(0.1, shape=[n_classes, ]))}    

def RNN(X, weights, biases):
    
    X = tf.reshape(X, [-1, n_inputs])

    
    X_in = tf.matmul(X, weights['in']) + biases['in']
    
    X_in = tf.reshape(X_in, [-1, n_steps, n_hidden_units])

   

    
    if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
        cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden_units, forget_bias=1.0, state_is_tuple=True)
    else:
        cell = tf.contrib.rnn.BasicLSTMCell(n_hidden_units)
    
    outputs, final_state = tf.nn.dynamic_rnn(cell, X_in,time_major=False,dtype=tf.float32)

    
    if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
        outputs = tf.unpack(tf.transpose(outputs, [1, 0, 2]))    # states is the last outputs
    else:
        outputs = tf.unstack(tf.transpose(outputs, [1,0,2]))
    results = tf.matmul(outputs[-1], weights['out']) + biases['out']    # shape = (128, 10)

    return results



pred = RNN(x, weights, biases)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
train_op = tf.train.AdamOptimizer().minimize(cost)

correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
next_char = tf.argmax(pred,1)
test_features=[]
for i in range(1):
	test_features.append(ret_one_hot('a'))
test_features=np.array(test_features)

pred_char='a'
with tf.Session() as sess:
    
    if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
        init = tf.initialize_all_variables()
    else:
        init = tf.global_variables_initializer()
    sess.run(init)
    hm_epochs=10
    step = 0
    for epoch in range(hm_epochs):
    	epoch_loss=0
    	global epoch_x
    	global epoch_y
    	epoch_x=training_one_hot_features;epoch_y=training_one_hot_labels;epoch_x=epoch_x.reshape([-1,n_steps,n_inputs])
    	_,c = sess.run([train_op, cost], feed_dict={x: epoch_x, y: epoch_y})
    	epoch_loss += c

        print('Epoch', epoch, 'completed out of',hm_epochs,'loss:',epoch_loss)
    for i in range(100):
    	test_features=[]
    	for i in range(1):
    		test_features.append(ret_one_hot(pred_char))
    	test_features=np.array(test_features)
    	
    	pred_val=sess.run([next_char],feed_dict={x:test_features.reshape([-1,n_steps,n_inputs])})
    	num_val=pred_val[0][0]
    	#print(a[0][0])
    	print(chr(97+num_val),end=',')
    	pred_char=chr(97+num_val)
    	

   

print()


