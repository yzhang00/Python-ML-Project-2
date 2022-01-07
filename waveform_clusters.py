from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
import matplotlib.pyplot as plt
import numpy as np
from numpy import mean
from numpy import std
import math

# initializing variables
train_X = np.loadtxt("Data/waveform_train_data_1.asc")
train_y = np.loadtxt("Data/waveform_train_labels_1.asc")
test_X = np.loadtxt("Data/waveform_test_data_1.asc")
test_y = np.loadtxt("Data/waveform_test_labels_1.asc")

# calculating affinity matrix manually based on simularity function


def calc_affinity(x1, x2):
    sqr_dist = 0
    for i in range(len(x1)):
        sqr_dist += (x1[i] - x2[i]) ** 2
    return math.exp(-0.1 * sqr_dist)


affinity = [[0 for _ in range(len(train_X))] for _ in range(len(train_X))]

for i in range(len(train_X)):
    for j in range(len(train_X)):
        affinity[i][j] = calc_affinity(train_X[i], train_X[j])

# initializing models and titles
models = (
    KMeans(n_clusters=2),
    SpectralClustering(n_clusters=2, affinity='rbf')

)

titles = (
    "KMeans Clusters n = 2",
    "Spectral Clusters n = 2"
)

# helper function for calculating score of prediction


def calc_score(prediction, labels):
    sum = 0
    # print('calc score')
    # print(len(prediction))
    # print(len(labels))
    # print(prediction[1:10])
    # print(labels[1:10])
    for i in range(len(prediction)):
        p = prediction[i]
        if p == 0:
            p -= 1
        if p == int(labels[i]):
            sum += 1
    return sum/len(prediction)


# printing output
print("WAVEFORM CLUSTERS")
for clf, title in zip(models, titles):
    print(title)
    label = clf.fit_predict(train_X)
    print("Train Accuracy: ", calc_score(label, train_y))
    label = clf.fit_predict(test_X)
    print("Test Accuracy: ", calc_score(label, test_y))
