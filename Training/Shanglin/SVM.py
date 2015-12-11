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
'''

'''
X_train = np.load("X_train_30000.npy")
X_val = np.load("X_val_30000.npy")
Y_train = np.load("Y_train_30000.npy")
Y_val = np.load("Y_val_30000.npy")



###############################################################################
# Compute a PCA (eigenfaces) on the face dataset (treated as unlabeled
# dataset): unsupervised feature extraction / dimensionality reduction
nc = 500

print("Extracting the top %d eigenfaces from %d faces"
      % (nc, X_train.shape[0]))
t0 = time()
pca = RandomizedPCA(n_components=nc).fit(X_train)
print("done in %0.3fs" % (time() - t0))

print("Projecting the input data on the eigenfaces orthonormal basis")
t0 = time()
X_train_pca = pca.transform(X_train)
X_val_pca = pca.transform(X_val)
print(X_train_pca.shape)
print(X_val_pca.shape)

np.save('X_train_pca_rd',X_train_pca)
np.save('X_val_pca_rd',X_val_pca)
print("done in %0.3fs" % (time() - t0))

X_train_pca = np.load('X_train_pca_rd.npy')
X_val_pca = np.load('X_val_pca_rd.npy')


n_c = 30000
X_train_fin = X_train_pca[0:n_c,:]
X_val_fin = X_val_pca[0:n_c,:]
Y_train = Y_train[0:n_c,:]
Y_val  = Y_val[0:n_c,:]


t0 = time()
classif = OneVsRestClassifier(SVC(kernel='rbf',C=1000))
classif.fit(X_train_fin, Y_train.ravel())

# Now predict the value of the digit on the second half:
score = classif.score(X_val_fin, Y_val.ravel())
print(score)
'''

X = np.load("X_ALL_10270.npy")
Y = np.load("Y_ALL.npy")
index = np.load("index.npy")

X_train = X[index[:,0:60000].ravel(),:];
X_test = X[index[:,60000:].ravel(),:];
Y_train = Y[index[:,0:60000].ravel(),:];
Y_test = Y[index[:,60000:].ravel(),:];

###############################################################################
# Compute a PCA (eigenfaces) on the face dataset (treated as unlabeled
# dataset): unsupervised feature extraction / dimensionality reduction
nc = 500

print("Extracting the top %d eigenfaces from %d faces"
      % (nc, X_train.shape[0]))
t0 = time()
pca = RandomizedPCA(n_components=nc).fit(X_train)
print("done in %0.3fs" % (time() - t0))

print("Projecting the input data on the eigenfaces orthonormal basis")
t0 = time()
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)
print(X_train_pca.shape)
print(X_test_pca.shape)

t0 = time()
classif = OneVsRestClassifier(SVC(kernel='rbf',C=1000))
classif.fit(X_train_pca, Y_train.ravel())

# Now predict the value of the digit on the second half:
score = classif.score(X_test_pca, Y_test.ravel())
print(score)
print("done in %0.3fs" % (time() - t0))
