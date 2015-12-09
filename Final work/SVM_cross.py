import scipy.io as sio
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.multiclass import OneVsRestClassifier

from sklearn.svm import LinearSVC
from sklearn import svm
from sklearn import tree


from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVC
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.decomposition import RandomizedPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_selection import SelectKBest
from sklearn.multiclass import OneVsRestClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor

from sklearn.datasets import make_gaussian_quantiles
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel

from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report


from time import time

import random
import sys 
print('Start')

'''
Data = sio.loadmat('X_75.mat')
X_train = Data['X_75']
# Builiding the input matrix

Data = sio.loadmat('X_94247_1_5195.mat')
X_5195 = Data['X_94247_1'];
X_train = np.hstack([X_train,X_5195])

Data = sio.loadmat('X_94247_2_5195.mat')
X_5195 = Data['X_94247_2'];
X_train = np.hstack([X_train,X_5195])

Data = sio.loadmat('X_94247_3_5195.mat')
X_5195 = Data['X_94247_3'];
X_train = np.hstack([X_train,X_5195])

Data = sio.loadmat('x1.mat')
X_5000 = Data['X1'];
X_train = np.hstack([X_train,X_5000])

Data = sio.loadmat('x2.mat')
X_5000 = Data['X2'];
X_train = np.hstack([X_train,X_5000])

Data = sio.loadmat('x3.mat')
X_5000 = Data['X3'];
X_train = np.hstack([X_train,X_5000])

np.save("X_ALL_10270.npy",X_train)
print(X_train.shape)


Data = sio.loadmat('Y_new.mat')
 = Data['Y_new']
np.save("Y_ALL.npy",Y)
print(Y.shape)

Data = sio.loadmat('order_for_94247.mat')
index = Data['order']
index=np.add(index,-1)

np.save("index.npy",index)
print(index.shape)
'''
'''
X = np.load("X_ALL_10270.npy")
Y = np.load("Y_ALL.npy")
index = np.load("index.npy")
print(X.shape)
print(Y.shape)
print(index.shape)



N = 30000;


X_train = X[index[:,0:N].ravel(),:];
X_val = X[index[:,N:60000].ravel(),:];
Y_train = Y[index[:,0:N].ravel(),:];
Y_val = Y[index[:,N:60000].ravel(),:];
print(X_train.shape)
print(X_val.shape)

np.save("X_train_30000.npy",X_train)
np.save("X_val_30000.npy",X_val)
np.save("Y_train_30000.npy",Y_train)
np.save("Y_val_30000.npy",Y_val)


X_train = np.load("X_train_30000.npy")
X_val = np.load("X_val_30000.npy")
Y_train = np.load("Y_train_30000.npy")
Y_val = np.load("Y_val_30000.npy")

X = np.load("X_ALL_10270.npy")
Y = np.load("Y_ALL.npy")
index = np.load("index.npy")
X = X[index[:,0:60000].ravel(),:];
Y = Y[index[:,0:60000].ravel(),:];
np.save("X_60000.npy",X);
np.save("Y_60000.npy",Y);
print(X.shape)
print(Y.shape)
'''
X = np.load("X_60000.npy");
Y = np.load("Y_60000.npy");

X = X[0:5000,:];
Y = Y[0:5000,:];
###############################################################################
# Compute a PCA (eigenfaces) on the face dataset (treated as unlabeled
# dataset): unsupervised feature extraction / dimensionality reduction
nc = 200

print("Extracting the top %d eigenfaces from %d faces"
      % (nc, X.shape[0]))
t0 = time()
pca = RandomizedPCA(n_components=nc).fit(X)
print("done in %0.3fs" % (time() - t0))

print("Projecting the input data on the eigenfaces orthonormal basis")
t0 = time()
X_pca = pca.transform(X)

print("done in %0.3fs" % (time() - t0))



print(X_pca.shape)
print(Y.shape)

# Split the dataset in two equal parts
X_train, X_test, y_train, y_test = train_test_split(
    X_pca, Y.ravel(), test_size=0.5, random_state=0)

# Set the parameters by cross-validation
tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                     'C': [1, 10, 100, 1000]},
                ]
scores = ['precision', 'recall']

for score in scores:
    print("# Tuning hyper-parameters for %s" % score)
    print()

    clf = GridSearchCV(SVC(C=1), tuned_parameters, cv=5,
                       scoring='%s_weighted' % score)
    clf.fit(X_train, y_train)

    print("Best parameters set found on development set:")
    print()
    print(clf.best_params_)
    print()
    print("Grid scores on development set:")
    print()
    for params, mean_score, scores in clf.grid_scores_:
        print("%0.3f (+/-%0.03f) for %r"
              % (mean_score, scores.std() * 2, params))
    print()

    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()
    y_true, y_pred = y_test, clf.predict(X_test)
    print(classification_report(y_true, y_pred))
    print()

