

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
from sklearn.decomposition import TruncatedSVD
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
from sklearn.metrics.pairwise import pairwise_distances
from sklearn import manifold

from time import time
from sklearn import ensemble

from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
from sklearn import preprocessing

import random
import sys
import scipy.io as scio 
print('Start')

Nt = 30000;
Nv = 20000;

X_Train_in = np.load('X_Train_in_30000.npy')
X_Test_in = np.load('X_Test_in_30000.npy')
X_val_in = np.load('X_Val_in_30000.npy')
#X = np.load("X_ALL_1400_OLD.npy")
Y = np.load("Y_ALL.npy")
index = np.load("index.npy")
X_Test = np.load("X_Test_14000.npy")


#X_Train  = X [index[:,0:Nt].ravel(),:];
Y_Train = Y [index[:,0:Nt].ravel(),:];
#X_val = X[index[:,-Nv:].ravel(),:]
Y_val = Y[index[:,-Nv:].ravel(),:]
print(' Input Finish')

print(X_Train_in.shape)
print(X_Test_in.shape)
print(X_val_in.shape)

t0 = time()
clf = RandomForestClassifier(n_estimators= 400)
clf.fit(X_Train_in , Y_Train.ravel())
pre =clf.predict(X_Test_in)
scio.savemat('Prediction1.1', {'Pre':pre})
pre_val =clf.predict(X_val_in)
score = clf.score(X_val_in, Y_val.ravel())
print("done in %0.3fs" % (time() - t0))
print(score)
