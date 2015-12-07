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



from time import time

import random
import sys 
print('Start')


Data = sio.loadmat('X_7_test.mat')
X_train = Data['X_7']
# Builiding the input matrix

Data = sio.loadmat('X_68_test.mat')
X_5195 = Data['X_68'];
X_train = np.hstack([X_train,X_5195])

Data = sio.loadmat('X_5195_1_test.mat')
X_5195 = Data['X_5195_1'];
X_train = np.hstack([X_train,X_5195])

Data = sio.loadmat('X_5195_2_test.mat')
X_5195 = Data['X_5195_2'];
X_train = np.hstack([X_train,X_5195])

Data = sio.loadmat('X_5195_3_test.mat')
X_5195 = Data['X_5195_3'];
X_train = np.hstack([X_train,X_5195])

Data = sio.loadmat('X_5000_1_test.mat')
X_5195 = Data['X_5000_1'];
X_train = np.hstack([X_train,X_5195])

Data = sio.loadmat('X_5000_2_test.mat')
X_5195 = Data['X_5000_2'];
X_train = np.hstack([X_train,X_5195])

Data = sio.loadmat('X_5000_3_test.mat')
X_5195 = Data['X_5000_3'];
X_train = np.hstack([X_train,X_5195])



np.save("X_Test_10270.npy",X_train)
print(X_train.shape)

