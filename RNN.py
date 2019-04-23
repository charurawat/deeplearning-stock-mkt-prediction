# SYS6016 - Project Final
# Charu Rawat (cr4zy), Elena Gillis (emg3sc)

# Implementing RNN on NASDAQ Wayfair stock data for price movement prediction

# importing packages
import pandas as pd
import tensorflow as tf

import os
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#%% reading in the data
data_in = pd.read_csv('data/data_transformed.csv')

#%% processing the data
# fill all NAs with the next available value
data_in = data_in.fillna(method='bfill')

# labels are between [0,3). So converting -1 to a label of 2
data_in['label'] = np.where(data_in['label'] == -1, 2, data_in['label'])
data_in['shift_label'] = data_in['label'].shift(-1)

#%% define a function to iterate through batches
def next_batch(data_in, dependent, target, offset, n_steps, batch_size):
    
    X_data = data_in[dependent][(offset * n_steps):((offset+1) * n_steps)].values
    X_data = X_data.reshape([batch_size, -1, len(dependent)])
    
    Y_data = data_in[target][(offset * n_steps):((offset+1) * n_steps)].values[-1]
    Y_data = Y_data.reshape([-1])
    
    return(X_data, Y_data)

#%% parameters for the RNN
n_inputs = 60
n_neurons = 10
n_outputs = 3
n_steps = 10
learning_rate = 0.001
n_iterations = 1000
batch_size = 1

#%% set up placeholders for input data
with tf.name_scope("Input") as rnn_scope:
    X = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
with tf.name_scope("Output") as rnn_scope:
    y = tf.placeholder(tf.int32, [None])

#%%  RNN model

# set up RNN using OutputProjectWrapper
with tf.name_scope("RNN") as rnn_scope:
    cell = tf.contrib.rnn.OutputProjectionWrapper(
        tf.nn.rnn_cell.BasicRNNCell(num_units=n_neurons, activation=tf.nn.relu),
        output_size= n_outputs)

with tf.name_scope("Output") as op_scope:    
    outputs, states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)

# Logits 
logits = tf.layers.dense(states, n_outputs)

# xentropy and loss
cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
loss = tf.reduce_mean(cross_entropy)

# optimizer
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

# prediction
prediction = tf.nn.in_top_k(logits, y, 1)
accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))

#%% run the session, train RNN on the data

# initailize the writer
writer = tf.summary.FileWriter('./graphs/RNN', tf.get_default_graph())

dependent = [col for col in data_in.columns if 'lag' in col]
target = 'shift_label'

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    init.run()
    
    counter = 0
    epoch = 0
    acc = 0
    total_loss = 0
    
    for iteration in range(n_iterations):
        X_batch, y_batch = next_batch(data_in, dependent, target, counter, n_steps, batch_size)
         
        if( X_batch.size < (n_steps * len(dependent))):
            continue
        
        _,  outs = sess.run([optimizer,outputs], feed_dict={X: X_batch, y: y_batch})
        total_loss += loss.eval(feed_dict={X: X_batch, y: y_batch})
        acc += accuracy.eval(feed_dict={X: X_batch, y: y_batch})
            
        if iteration % 100 == 0:
            print(counter)
            print(iteration, "\tLoss: ", (total_loss/ iteration))
            print(iteration, "\tAccuracy: ", (acc/ iteration))    
        
        counter += 1
    
writer.close()    
