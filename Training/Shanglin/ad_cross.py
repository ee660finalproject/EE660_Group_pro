# -*- coding: cp936 -*-
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
print('Start')

Nt = 1000;
Nv = 1000;
X = np.load("X_ALL_10270.npy")
t0 = time()

print("done in %0.3fs" % (time() - t0))

Y = np.load("Y_ALL.npy")
index = np.load("index.npy")

X_t = X[index[:,0:Nt].ravel(),5270:10269];
X_v = X[index[:,60000-Nv:60000].ravel(),5270:10269];
Y_t = Y[index[:,0:Nt].ravel(),:];
Y_v = Y[index[:,60000-Nv:60000].ravel(),:];
print(X_t.shape)
print(X_v.shape)



nc = 600;


print("Extracting the top %d eigenfaces from %d faces"
      % (nc, X_t.shape[0]))
t0 = time()



#lda = LinearDiscriminantAnalysis(n_components=nc).fit(X_t,Y_t.ravel());
#X_train_lda= lda.transform(X_t)
#X_val_lda = lda.transform(X_v)




rdpca = RandomizedPCA(n_components=nc).fit(X_t)
X_train_rpca = rdpca.transform(X_t)
X_val_rpca = rdpca.transform(X_v)


X_train_week = X[index[:,0:Nt].ravel(),0:7];
X_val_week = X[index[:,60000-Nv:60000].ravel(),0:7];

X_train_des = X[index[:,0:Nt].ravel(),7:75];
X_val_des = X[index[:,60000-Nv:60000].ravel(),7:75];
rdpca = RandomizedPCA(n_components=67).fit(X_train_des)
X_train_des_rpca = rdpca.transform(X_train_des)
X_val_des_rpca = rdpca.transform(X_val_des)


X_train = np.hstack([X_val_des_rpca ])

print("done in %0.3fs" % (time() - t0))
print(X_train.shape )




'''
t0 = time()
classif = OneVsRestClassifier(SVC(kernel='rbf',C=1000))
classif.fit(X_train_lda, Y_train.ravel())

# Now predict the value of the digit on the second half:
score = classif.score(X_val_lda     , Y_val.ravel())
print("done in %0.3fs" % (time() - t0))
print(score)
'''

t0 = time()
#clf = ensemble.GradientBoostingClassifier(learning_rate=1, subsample= 0.5,n_estimators= 500, max_leaf_nodes= 4, max_depth= 3)
#clf = RandomForestClassifier(n_estimators= 500)
clf = AdaBoostClassifier(n_estimators= 500)
clf.fit(X_train_des_rpca , Y_t.ravel())

pre =clf.predict(X_val_des_rpca)
score = clf.score(X_val_des_rpca, Y_v.ravel())
print("done in %0.3fs" % (time() - t0))
print(score)    
        
'''
# Split the dataset in two equal parts
X_train, X_test, y_train, y_test = train_test_split(
    X_train, Y_v.ravel(), test_size=0.2, random_state=0)



# Set the parameters by cross-validation
tuned_parameters = [{ 'max_features':['sqrt','log2'],
                     'max_depth' :[6,8,12,16,20],'n_estimators': [500 ]},
                ]

tuned_parameters = [{ 
                    'learning_rate': [0.1], 'subsample': [0.5],'n_estimators': [500], 'max_leaf_nodes': [4], 'max_depth': [3], 'random_state': [2],
                   'min_samples_split': [5]},
                ]

scores = ['precision', 'recall']

for score in scores:
    print("# Tuning hyper-parameters for %s" % score)
    print()

    clf = GridSearchCV(ensemble.GradientBoostingClassifier(), tuned_parameters, 
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

'''

'''
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

'''
'''
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
'''
