# SYS6016 - Project Final
# Charu Rawat (cr4zy), Elena Gillis (emg3sc)

# Implementing CNN on NASDAQ Wayfair stock data for price movement prediction

# importing packages
import pandas as pd
import tensorflow as tf
import time
import matplotlib.pyplot as plt

#%% reading in the data
data_in = pd.read_csv('data/data_transformed.csv')

#%% one-hot-encoding
y_vals = pd.DataFrame({'y_11':[0 for i in range(data_in.shape[0])], 
                               'y_0':[0 for i in range(data_in.shape[0])], 
                               'y_1':[0 for i in range(data_in.shape[0])]})
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
data_in = data_in.fillna(method = 'bfill')
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
n_epochs = 15

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
    
#%% CNN layers

# Input Layer
input_layer = tf.reshape(tf.cast(stock_input,tf.float32), [-1, 60, 1])

# Convolutional Layer #1
conv1 = tf.layers.conv1d(
      inputs=input_layer,
      filters=64,
      kernel_size=3,
      padding="same",
      activation=tf.nn.relu)

train_size

# Pooling Layer #1
pool1 = tf.layers.max_pooling1d(
        inputs=conv1, 
        pool_size=2, 
        strides=1)

# Convolutional Layer #2 and Pooling Layer #2
conv2 = tf.layers.conv1d(
      inputs=pool1,
      filters=32,
      kernel_size=3,
      padding="same",
      activation=tf.nn.relu)

pool2 = tf.layers.max_pooling1d(
        inputs=conv2, 
        pool_size=2, 
        strides=1)

# Dense Layer
pool2_flat = tf.layers.flatten(pool2)

dense = tf.layers.dense(
        inputs=pool2_flat, 
        units=1, 
        activation=tf.nn.relu)

dropout = tf.layers.dropout(
        inputs=dense, 
        rate=0.4)

# Logits Layer
logits = tf.layers.dense(inputs=dropout, units=3)

xentropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=label, logits=logits)
loss = tf.reduce_mean(xentropy)

# Optimizer
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
writer = tf.summary.FileWriter('./graphs/CNN', tf.get_default_graph())

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
                #print(1)
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
    print('{0} epochs'.format(n_epochs))
    print('Learning Rate: {0}'.format(learning_rate))
    print('{0} batches'.format(batch_size))
    print('Total time: {0} seconds'.format(time.time() - start_time))
    print('Train Loss: {0}'.format(l_train[-1]))
    print('Train Accuracy: {0}'.format(acc_train[-1]))
    print('Test Loss: {0}'.format(l_test[-1]))
    print('Test Accuracy: {0}'.format(acc_test[-1])) 

# close writer
writer.close()

#%% Visualize loss and accuracy for train and test

# Create count of the number of epochs
epoch_count = range(0, len(l_train))

# Loss
plt.plot(epoch_count, l_train, 'r-')
plt.plot(epoch_count, l_test, 'b-')
plt.legend(['Train', 'Test'])
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss over epochs')
plt.show();

# Accuracy
plt.plot(epoch_count, acc_train, 'r-')
plt.plot(epoch_count, acc_test, 'b-')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy over epochs')
plt.legend(['Train', 'Test'])
plt.show();