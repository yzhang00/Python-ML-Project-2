from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
import numpy as np
from numpy import mean
from numpy import std

# initializing variables
train_X = np.loadtxt("Data/waveform_train_data_1.asc")
train_y = np.loadtxt("Data/waveform_train_labels_1.asc")
test_X = np.loadtxt("Data/waveform_test_data_1.asc")
test_y = np.loadtxt("Data/waveform_test_labels_1.asc")
test_acc = 0
train_acc = 0
test_neurons = 0
train_neurons = 0

# looping through number of neurons in hidden layer to find optimal value
for i in range(1, 400):
    # initializing model with i neurons in hidden layer
    clf = MLPClassifier(hidden_layer_sizes=(i), activation='logistic')
    # calculating accuracy for training data
    clf.fit(train_X, train_y).predict(train_X)
    if clf.score(train_X, train_y) > train_acc:
        train_acc = clf.score(train_X, train_y)
        train_neurons = i
    # calculating accuracy for test data
    clf.fit(train_X, train_y).predict(test_X)
    if clf.score(test_X, test_y) > test_acc:
        test_acc = clf.score(test_X, test_y)
        test_neurons = i

print("THREE LAYER NEURAL NETWORK")
# printing optimal number of neurons for training data
print("Optimal Num Neurons for Training Data: %d" % train_neurons)
clf = MLPClassifier(hidden_layer_sizes=(train_neurons), activation='logistic')
clf.fit(train_X, train_y).predict(train_X)
print("Train Accuracy:", clf.score(train_X, train_y))
clf.fit(train_X, train_y).predict(test_X)
print("Test Accuracy:", clf.score(test_X, test_y))

# printing optimal number of neurons for test data
print("Optimal Num Neurons for Test Data: %d" % test_neurons)
clf = MLPClassifier(hidden_layer_sizes=(test_neurons), activation='logistic')
clf.fit(train_X, train_y).predict(train_X)
print("Train Accuracy:", clf.score(train_X, train_y))
clf.fit(train_X, train_y).predict(test_X)
print("Test Accuracy:", clf.score(test_X, test_y))
