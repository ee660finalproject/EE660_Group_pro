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

import random
import sys
from time import time
print('Start')

# the index of all 6000 samples for training 
Data = sio.loadmat('order_for_60000.mat')
index = Data['order2']
N = 30000;
index = range(2*N)
random.shuffle(index)


Data = sio.loadmat('Y_75_train.mat')
Y = Data['Y_75_train']
Y_train = Y[index[0:N],:]
Y_test = Y[index[-N:],:]


X_train_fin  = np.load('X_train_pca_200.npy')
X_test_fin = np.load('X_test_pca_200.npy')

print(X_train_fin.shape)
print(X_test_fin.shape)

t0 = time()

 
bdt_real = AdaBoostClassifier(
    DecisionTreeClassifier(max_depth=2),
    n_estimators=600,
    learning_rate=1,
    algorithm="SAMME.R")

bdt_real.fit(X_train_fin, Y_train.ravel())
P = bdt_real.predict(X_test_fin)
A = accuracy_score(P, Y_test.ravel())

print(A)
print("done in %0.3fs" % (time() - t0))
print('Finish')
