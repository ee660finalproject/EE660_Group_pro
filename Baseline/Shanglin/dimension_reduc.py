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
from sklearn.decomposition import PCA, KernelPCA
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
from sklearn.decomposition import TruncatedSVD


import random
import sys
from time import time
from sklearn.decomposition import RandomizedPCA

print('Start')

# the index of all 6000 samples for training 
Data = sio.loadmat('order_for_60000.mat')
index = Data['order2']

X_train_LDA = np.load("X_train_LDA.npy")
X_ori = np.load( "X_train_5270.npy" )

X_base = X_ori ;

Data = sio.loadmat('x1.mat')
X_num = Data['X1'];
X_new = X_num[index.ravel(),:]
X_train_com = np.hstack([X_base,X_new])

Data = sio.loadmat('x2.mat')
X_num = Data['X2'];
X_new = X_num[index.ravel(),:]
X_train_com = np.hstack([X_train_com,X_new])

Data = sio.loadmat('x3.mat')
X_num = Data['X3'];
X_new = X_num[index.ravel(),:]
X_train_com = np.hstack([X_train_com,X_new])

Data = sio.loadmat('Y_75_train.mat')
Y = Data['Y_75_train']

N = 30000;
index = range(2*N)
random.shuffle(index)
X_train_fin = X_train_com[index[0:N], :]
X_test_fin= X_train_com[index[-N:],:]

print(X_base.shape)
print(X_train_com.shape)

###############################################################################
# Compute a PCA (eigenfaces) on the face dataset (treated as unlabeled
# dataset): unsupervised feature extraction / dimensionality reduction
n_components = 200

print("Extracting the top %d eigenfaces from %d faces"
      % (n_components, X_train_fin.shape[0]))
t0 = time()
#pca = RandomizedPCA(n_components=n_components).fit(X_train_fin)
#print(pca.explained_variance_ratio_) 

#kpca = KernelPCA(kernel="rbf", fit_inverse_transform=True, gamma=10)
#svd = TruncatedSVD(n_components=5, random_state=42)
pca  = PCA(n_components=n_components);

print("done in %0.3fs" % (time() - t0))



print("Projecting the input data on the eigenfaces orthonormal basis")
t0 = time()
X_train_pca = pca.fit_transform(X_train_fin)
X_test_pca = pca.fit_transform(X_test_fin)
np.save('X_train_pca_pca',X_train_pca)
np.save('X_test_pca_pca',X_test_pca)
print("done in %0.3fs" % (time() - t0))


