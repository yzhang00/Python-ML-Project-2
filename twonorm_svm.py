from sklearn import svm, datasets
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import numpy as np
from numpy import mean
from numpy import std

# initializing variables
train_X = np.loadtxt("Data/twonorm_train_data_1.asc")
train_y = np.loadtxt("Data/twonorm_train_labels_1.asc")
test_X = np.loadtxt("Data/twonorm_test_data_1.asc")
test_y = np.loadtxt("Data/twonorm_test_labels_1.asc")

# linear svm
print("LINEAR SVM")
clf = svm.SVC(kernel='linear')
clf.fit(train_X, train_y).predict(train_X)
# printing num vectors, training, and test accuracy
print("Num Support Vectors: %d" % len(clf.support_vectors_))
print("Train Accuracy:", clf.score(train_X, train_y))
clf.fit(train_X, train_y).predict(test_X)
print("Test Accuracy:", clf.score(test_X, test_y))

# rbf svm with varying sigma
for i in range(1, 9):
    print("RBF SVM WITH SIGMA = 2^-%d" % i)
    sig = 2**(-1 * i)
    gam = 1/(2*(sig**2))
    clf = svm.SVC(kernel='rbf', gamma=sig)
    clf.fit(train_X, train_y).predict(train_X)
    # printing num vectors, training, and test accuracy
    print("Num Support Vectors: %d" % len(clf.support_vectors_))
    print("Train Accuracy:", clf.score(train_X, train_y))
    clf.fit(train_X, train_y).predict(test_X)
    print("Test Accuracy:", clf.score(test_X, test_y))
    # printing cross-validation results
    cv = KFold(n_splits=5, random_state=1, shuffle=True)
    scores = cross_val_score(clf, train_X, train_y,
                             scoring='accuracy', cv=cv, n_jobs=-1)
    print('Cross-Validation Accuracy (Mean/Std): %.3f (%.3f)' %
          (mean(scores), std(scores)))
