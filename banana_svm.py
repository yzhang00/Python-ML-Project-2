from sklearn import svm, datasets
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import numpy as np
from numpy import mean
from numpy import std

# initializing variables
train_X = np.loadtxt("Data/banana_train_data_1.asc")
train_y = np.loadtxt("Data/banana_train_labels_1.asc")
test_X = np.loadtxt("Data/banana_test_data_1.asc")
test_y = np.loadtxt("Data/banana_test_labels_1.asc")

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
    clf = svm.SVC(kernel='rbf', gamma=gam)
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

# plotting banana svm classifiers
# referencing code from sklearn website


def make_meshgrid(x, y, h=0.02):
    """Create a mesh of points to plot in

    Parameters
    ----------
    x: data to base x-axis meshgrid on
    y: data to base y-axis meshgrid on
    h: stepsize for meshgrid, optional

    Returns
    -------
    xx, yy : ndarray
    """
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return xx, yy


def plot_contours(ax, clf, xx, yy, **params):
    """Plot the decision boundaries for a classifier.

    Parameters
    ----------
    ax: matplotlib axes object
    clf: a classifier
    xx: meshgrid ndarray
    yy: meshgrid ndarray
    params: dictionary of params to pass to contourf, optional
    """
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out


sig = 2**(-1)
gam = 1/(2*(sig**2))
models = (
    svm.SVC(kernel="linear"),
    svm.SVC(kernel="rbf", gamma=gam)
)
models = (clf.fit(train_X, train_y) for clf in models)

titles = (
    "SVM with linear kernel",
    "SVM with RBF kernel"
)

# Set-up 2x1 grid for plotting.
fig, sub = plt.subplots(2, 1)
plt.subplots_adjust(wspace=0.4, hspace=0.4)

X0, X1 = train_X[:, 0], train_X[:, 1]
xx, yy = make_meshgrid(X0, X1)

for clf, title, ax in zip(models, titles, sub.flatten()):
    plot_contours(ax, clf, xx, yy, cmap=plt.cm.coolwarm, alpha=0.8)
    ax.scatter(X0, X1, c=train_y, cmap=plt.cm.coolwarm, s=20, edgecolors="k")
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xlabel("Attribute 1")
    ax.set_ylabel("Attribute 2")
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_title(title)

plt.show()
