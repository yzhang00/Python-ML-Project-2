from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
import matplotlib.pyplot as plt
import numpy as np
from numpy import mean
from numpy import std
import math

# initializing variables
train_X = np.loadtxt("Data/crescent_and_the_full_moon.asc")

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
    KMeans(n_clusters=2).fit_predict(train_X),
    KMeans(n_clusters=4).fit_predict(train_X),
    SpectralClustering(
        n_clusters=2, affinity='rbf').fit_predict(train_X),
    SpectralClustering(
        n_clusters=4, affinity='rbf').fit_predict(train_X)

)

titles = (
    "KMeans Clusters n = 2",
    "KMeans Clusters n = 4",
    "Spectral Clusters n = 2",
    "Spectral Clusters n = 4"
)

for label, title in zip(models, titles):
    plt.figure()
    u_labels = np.unique(label)
    for i in u_labels:
        plt.scatter(train_X[label == i, 0], train_X[label == i, 1], label=i)
    plt.legend()
    plt.title(title)
    plt.show()
