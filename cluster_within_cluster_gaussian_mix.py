from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
import numpy as np
from numpy import mean
from numpy import std
import math

# initializing variables
train_X = np.loadtxt("Data/cluster_within_cluster.asc")

clf = GaussianMixture(n_components=2)
gm = clf.fit(train_X)
label = clf.fit_predict(train_X)
plt.figure()
u_labels = np.unique(label)
for i in u_labels:
    plt.scatter(train_X[label == i, 0], train_X[label == i, 1], label=i)
plt.legend()
plt.title("Cluster Within Cluster Gaussian Mixture n = 2")
plt.show()

# printing params as output
print("GAUSSIAN MIXTURE")
print("Weights:")
print(gm.weights_)
print("Means:")
print(gm.means_)
print("Covariances:")
print(gm.covariances_)
