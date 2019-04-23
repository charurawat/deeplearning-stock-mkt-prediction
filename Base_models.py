# SYS6016 - Project Stage II - Charu Rawat (cr4zy) and Elena Gillis (emg3sc)

import pandas as pd

# packages for baseline models
from sklearn import datasets 
from sklearn.metrics import confusion_matrix 
from sklearn.model_selection import train_test_split 
# for kNN
from sklearn.neighbors import KNeighborsClassifier
# for SVM
from sklearn.svm import SVC 

#%% Read in transformed data
data = pd.read_csv('data/data_transformed.csv')
print(data.shape)

#%% prepare the data
# extract X -> features, y -> label 
X = data.iloc[:,:-1] 
y =  data.iloc[:,-1:] 

# dividing X, y into train and test data 
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0)

#%% kNN classifier

# training a KNN classifier 
knn = KNeighborsClassifier(n_neighbors = 7).fit(X_train, y_train) 
  
# accuracy on X_test 
knn_accuracy = knn.score(X_test, y_test) 
print('kNN Accuracy: {0}'.format(knn_accuracy)) 
  
# creating a confusion matrix 
knn_predictions = knn.predict(X_test)  
cm_knn = confusion_matrix(y_test, knn_predictions) 
print(cm_knn)

#%% SVM classifier
  
# training a linear SVM classifier 
svm_model_linear = SVC(kernel = 'linear', C = 1).fit(X_train, y_train) 
svm_predictions = svm_model_linear.predict(X_test) 
  
# model accuracy for X_test   
svm_accuracy = svm_model_linear.score(X_test, y_test) 
print('SVM Accuracy: {0}'.format(svm_accuracy))
  
# creating a confusion matrix 
cm_svm = confusion_matrix(y_test, svm_predictions) 
print(cm_svm)