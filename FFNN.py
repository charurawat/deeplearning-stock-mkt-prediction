# SYS6016 - Project Phase II
# Charu Rawat (cr4zy), Elena Gillis (emg3sc)

# Implementing FFNN on NASDAQ Wayfair stock data for price movement prediction

# importing packages
import pandas as pd
import tensorflow as tf
import time
import matplotlib.pyplot as plt

#%% reading in the data
data_in = pd.read_csv('data/data_transformed.csv')

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

# train and test size
train_size = len(train_data)
test_size = len(test_data)

#%% setting parameters
batch_size = 100
learning_rate = 0.01
n_epochs = 1000
hidden_layer1_units = 40
hidden_layer2_units = 20

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

#%% FFNN layers

# First hidden layer
with tf.name_scope("Hidden_layer1") as sigmoid_scope:
    W = tf.get_variable(initializer = tf.random_normal(shape=[stock_input.shape[1].value, hidden_layer1_units]), name="Weight_1")
    B = tf.cast(tf.get_variable(initializer = tf.zeros(shape = [hidden_layer1_units]), name = 'Bias_1'), tf.float32)
    stock_input = tf.cast(stock_input,tf.float32)
    preact1 = tf.add((tf.matmul(stock_input,W)),B)
    hidden_layer_1 = tf.nn.relu(preact1)
    
# Second hidden layer
with tf.name_scope("Hidden_layer2") as sigmoid_scope:
    W2 = tf.get_variable(initializer = tf.random_normal(shape=[hidden_layer_1.shape[1].value, hidden_layer2_units]), name="Weight_2")
    B2 = tf.cast(tf.get_variable(initializer = tf.zeros(shape = [hidden_layer2_units]), name = 'Bias_2'), tf.float32)
    preact2 = tf.add((tf.matmul(stock_input,W)),B)
    hidden_layer_2 = tf.nn.relu(preact2)

# Output layer    
with tf.name_scope("Output_layer") as sigmoid_scope:
    Output_units = y_vals.shape[1]
    W3 = tf.get_variable(initializer = tf.random_normal(shape=[hidden_layer_2.shape[1].value, Output_units]), name="Weight_3")
    B3 = tf.cast(tf.get_variable(initializer = tf.zeros(shape = [Output_units]), name = 'Bias_3'), tf.float32)
    logits = tf.add(tf.matmul(hidden_layer_2,W3),B3)
    xentropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=label, logits=logits)
    loss = tf.reduce_mean(xentropy)

# Adam Optimizer
with tf.name_scope("Optimizer"):
    optimize = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)
 
# Predictions
preds = tf.nn.softmax(logits)
correct_preds = tf.equal(tf.argmax(preds, 1), tf.argmax(label, 1))
accuracy = tf.reduce_sum(tf.cast(correct_preds, tf.float32))

#%% Run the trainging session

# initialize arrays to store valus
l_train = []
l_test = []
acc_train = []
acc_test = []        

# start writer
writer = tf.summary.FileWriter('./graphs', tf.get_default_graph())

# train the model and record train loss and accuracy
with tf.Session() as sess:
    start_time = time.time()
    sess.run(tf.global_variables_initializer())
    for i in range(n_epochs): 
        sess.run(train_init)
        total_loss = 0
        total_a = 0
        n_batches = 0
        try:
            while True:
                _, l, a = sess.run([optimize, loss, accuracy])
                total_loss += l
                total_a += a
                n_batches += 1
        except tf.errors.OutOfRangeError:
            pass
        
        # to print loss and accuracy for each epoch
#        print('\nEpoch {0}, Loss: {1}, Accuracy: {2}'
#              .format(i, total_loss/n_batches, total_a/train_size))
        l_train.append(total_loss/n_batches)
        acc_train.append(total_a/train_size)
        
        sess.run(test_init)
        total_v_loss = 0
        total_correct_preds = 0
        try:
            while True:
                loss_batch, accuracy_batch = sess.run([loss, accuracy])
                total_v_loss += loss_batch
                total_correct_preds += accuracy_batch
        except tf.errors.OutOfRangeError:
            pass
        l_test.append(total_v_loss/batch_size)
        acc_test.append(total_correct_preds/n_test)

    
# final test accuracy
    sess.run(test_init)  # drawing samples from test_data
    correct_preds = 0
    try:
        while True:
            accuracy_batch = sess.run(accuracy)
            correct_preds += accuracy_batch
    except tf.errors.OutOfRangeError:
        pass
   
    a = correct_preds / n_test
    print('Number of units in hidden layer 1: {0}'.format(hidden_layer1_units))
    print('Number of units in hidden layer 2: {0}'.format(hidden_layer2_units))
    print('{0} epochs'.format(n_epochs))
    print('Learning Rate: {0}'.format(learning_rate))
    print('{0} batches'.format(batch_size))
    print('Total time: {0} seconds'.format(time.time() - start_time))
    print('Train Loss: {0}'.format(l_train[-1]))
    print('Train Accuracy: {0}'.format(acc_train[-1]))
    print('Test Loss: {0}'.format(l_test[-1]))
    print('Test Accuracy: {0}'.format(acc_test[-1]))
    #print('Accuracy {0}'.format(a))  

# close writer
writer.close()    

#%% Visualize loss and accuracy for train and test

# Create count of the number of epochs
epoch_count = range(10, len(l_train))

# Loss
plt.plot(epoch_count, l_train[10:], 'r-')
plt.plot(epoch_count, l_test[10:], 'b-')
plt.legend(['Train', 'Test'])
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss over epochs')
plt.show();

# Accuracy
plt.plot(epoch_count, acc_train[10:], 'r-')
plt.plot(epoch_count, acc_test[10:], 'b-')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy over epochs')
plt.legend(['Train', 'Test'])
plt.show();
