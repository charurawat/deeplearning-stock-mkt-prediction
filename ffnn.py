# FFNN
# importing packages
import pandas as pd
import tensorflow as tf
import time
import matplotlib.pyplot as plt
import numpy as np

#%% reading in the data
data_in = pd.read_csv('C:/Users/Elena/Desktop/Spring 19/ML/Project/Phase2/data_transformed.csv')

#%% FFNN

# one-hot-encoding
y_vals = pd.DataFrame({'y_11':[0 for i in range(data_in.shape[0])], 'y_0':[0 for i in range(data_in.shape[0])], 'y_1':[0 for i in range(data_in.shape[0])]})
y_vals.index = range(min(data_in.index),max(data_in.index)+1)

for i in list(data_in.index):
    label = data_in.loc[i]['label']
    if label == -1:
        y_vals.loc[i]['y_11'] = 1
    elif label == 0:
        y_vals.loc[i]['y_0'] = 1
    else:
        y_vals.loc[i]['y_1'] = 1

# tf data structure
data_in = data_in.fillna(0)
n_test = 10000

#Train-Test split. Test has the last n_test rows.
train_data = data_in.iloc[:-n_test]
test_data = data_in.iloc[-n_test:]

X_train = train_data.iloc[:, 0:data_in.shape[1]-1]
Y_train = y_vals.iloc[:-n_test]

X_test = test_data.iloc[:, 0:data_in.shape[1]-1]
Y_test = y_vals.iloc[-n_test:]

### Train and Test
train = (X_train.values, Y_train.values)
valid = (X_test.values, Y_test.values)

#%% setting parameters
batch_size = 100
learning_rate = 0.01
n_epochs = 10
hidden_layer_units = 15

#%% training and test data
# create training, testing data and batch it
train_data = tf.data.Dataset.from_tensor_slices(train)
train_data = train_data.batch(batch_size)

test_data = tf.data.Dataset.from_tensor_slices(valid)
test_data = test_data.batch(batch_size)

# create one iterator and initialize it with different datasets
iterator = tf.data.Iterator.from_structure(train_data.output_types, train_data.output_shapes)
stock_input, label = iterator.get_next()

train_init = iterator.make_initializer(train_data)  # initializer for train_data
test_init = iterator.make_initializer(test_data)  # initializer for train_data

#%% first hidden layer
with tf.name_scope("Hidden_layer"):

    hidden_layer_units = 15
    
    with tf.name_scope('weights'):
        W = tf.get_variable(initializer = tf.random_normal(shape=[stock_input.shape[1].value, hidden_layer_units]), name="Weight_1")
        #tf.summary.scalar('W',W)
        
    with tf.name_scope('biases'):
        B = tf.cast(tf.get_variable(initializer = tf.zeros(shape = [hidden_layer_units]), name = 'Bias_1'), tf.float32)
        #tf.summary.scalar('B', B)

    stock_input = tf.cast(stock_input,tf.float32)
    
    with tf.name_scope('Wx_plus_b'):
        preact = tf.add((tf.matmul(stock_input,W)),B)
        #tf.summary.histogram('pre_activations', preact)
    
    hidden_layer_1 = tf.nn.relu(preact)
    #tf.summary.histogram('activations', hidden_layer_1)
    
with tf.name_scope("Output_layer"):

    Output_units = y_vals.shape[1]

    W2 = tf.get_variable(initializer = tf.random_normal(shape=[W.shape[1].value, Output_units]), name="Weight_2")
    #tf.summary.scalar('W2', W2)
    B2 = tf.cast(tf.get_variable(initializer = tf.zeros(shape = [Output_units]), name = 'Bias_2'), tf.float32)
    #tf.summary.scalar('B2', B2)
    
    logits = tf.add(tf.matmul(hidden_layer_1,W2),B2)
    #tf.summary.histogram('output', logits)
    
    # use cross entropy of softmax of logits as the loss function
    with tf.name_scope('total'):
        xentropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=label, logits=logits)
        #tf.summary.scalar('xentropy', xentropy)
        
    with tf.name_scope('loss'):
        loss = tf.reduce_mean(xentropy)
        #tf.summary.scalar('loss', loss)

with tf.name_scope("Optimizer") as sigmoid_scope:
    optimize = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
  
preds = tf.nn.softmax(logits)
with tf.name_scope('accuracy'):
    with tf.name_scope('correct_predictions'):
        correct_preds = tf.equal(tf.argmax(preds, 1), tf.argmax(label, 1))
    with tf.name_scope('accuracy'):
        accuracy = tf.reduce_sum(tf.cast(correct_preds, tf.float32))
#tf.summary.scalar('accuracy', accuracy)

# merge and write all the summaries
#merged = tf.summary.merge_all()

#%% run session

l_train = []
l_test = []

with tf.Session() as sess:
    start_time = time.time()
    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter('./graphs', tf.get_default_graph())
    
    # train the model n_epochs times
    
    
    for i in range(n_epochs):
        sess.run(train_init)  # drawing samples from train_data
        total_loss = 0
        n_batches = 0
        
        try:
            while True:
                _, batch_loss = sess.run([optimize, loss])
                total_loss += batch_loss
                n_batches += 1
                
        except tf.errors.OutOfRangeError:
            pass
        
        l_train.append(total_loss / n_batches)           
        #print('Average loss epoch {0}: {1}'.format(i, total_loss / n_batches))
    print('Total time: {0} seconds'.format(time.time() - start_time))
    

    sess.run(test_init)  # drawing samples from test_data
    total_correct_preds = 0
    try:
        while True:
            accuracy_batch = sess.run(accuracy)
            total_correct_preds += accuracy_batch
    except tf.errors.OutOfRangeError:
        pass
   
    a = total_correct_preds / n_test
    print('Accuracy {0}'.format(a))
    
writer.close()

#%% visualize

# Create count of the number of epochs
epoch_count = range(1, len(l_train) + 1)

# Visualize loss history
plt.plot(epoch_count, l_train, 'r--')
#plt.plot(epoch_count, l_test, 'b-')
#plt.legend(['Training Loss', 'Test Loss'])
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show();
