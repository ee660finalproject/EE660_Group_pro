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

import random
import sys  

print('Start')

Data = sio.loadmat('X_75_train.mat')
X_train = Data['X_75_train']

# the index of all 6000 samples for training 
Data = sio.loadmat('order_for_60000.mat')
index = Data['order2']

# Builiding the input matrix

Data = sio.loadmat('X_94247_1_5195.mat')
X_num = Data['X_94247_1'];
X_new = X_num[index.ravel(),:]
X_train = np.hstack([X_train,X_new])

Data = sio.loadmat('X_94247_2_5195.mat')
X_num = Data['X_94247_2'];
X_new = X_num[index.ravel(),:]
X_train = np.hstack([X_train,X_new])

Data = sio.loadmat('X_94247_3_5195.mat')
X_num = Data['X_94247_3'];
X_new = X_num[index.ravel(),:]
X_train = np.hstack([X_train,X_new])


#print(np.shape(X_train ))

#Output the training sample;
np.save("X_train_5270.npy",X_train)
print('Finish')
sys.exit(0)
