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
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_selection import SelectKBest
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor

from sklearn.datasets import make_gaussian_quantiles
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from time import time

import random
import sys 
print('Start')

# the index of all 6000 samples for training 
Data = sio.loadmat('order_for_60000.mat')
index = Data['order2']
N = 5000;
index = range(2*N)
random.shuffle(index)


Data = sio.loadmat('Y_75_train.mat')
Y = Data['Y_75_train']
Y_train = Y[index[0:N],:]
Y_test = Y[index[-N:],:]


X_train_fin  = np.load('X_train_pca_rd.npy')
X_test_fin = np.load('X_test_pca_rd.npy')
X_train_fin = X_train_fin[index[0:N],:]
X_test_fin = X_test_fin[index[0:N],:]
print(X_train_fin.shape)
print(X_test_fin.shape)

t0 = time()
clf = RandomForestClassifier( n_estimators=300);
clf.fit(X_train_fin, Y_train.ravel())
pre = clf.predict(X_test_fin)
A = accuracy_score(pre, Y_test.ravel())
print(A)
print("done in %0.3fs" % (time() - t0))
print('Finish')
