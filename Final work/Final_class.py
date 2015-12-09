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

Nt = 60000;
Nv = 20000;
nc = 68;

X = np.load("X_ALL_1400_OLD.npy")
Y = np.load("Y_ALL.npy")
index = np.load("index.npy")
X_Test = np.load("X_Test_14000.npy")


X_Train  = X [index[:,0:Nt].ravel(),:];
Y_Train = Y [index[:,0:Nt].ravel(),:];
X_val = X[index[:,-Nv:].ravel(),:]
Y_val = Y[index[:,-Nv:].ravel(),:]
print(' Input Finish')


###############################  Feature Processing   ##############################
X_Train_week = X_Train[:,0:7];
X_Train_Des = X_Train[:,7:75];
X_Train_new = X_Train[:,75:75+20];
X_Train_Fin = X_Train[:,75+20:75+20+5195];
X_Train_UPC = X_Train[:,75+20+5195:];

X_Test_week = X_Test[:,0:7];
X_Test_Des = X_Test[:,7:75];
X_Test_new = X_Test[:,75:75+20];
X_Test_Fin = X_Test[:,75+20:75+20+5195];
X_Test_UPC = X_Test[:,75+20+5195:];

X_val_week = X_val[:,0:7];
X_val_Des = X_val[:,7:75];
X_val_new = X_val[:,75:75+20];
X_val_Fin = X_val[:,75+20:75+20+5195];
X_val_UPC = X_val[:,75+20+5195:];

lda = LinearDiscriminantAnalysis(n_components=nc).fit(X_Train_Des,Y_Train.ravel());
X_Train_Des_lda= lda.transform(X_Train_Des)
X_Test_Des_lda = lda.transform(X_Test_Des)
X_val_Des_lda = lda.transform(X_val_Des)

'''
rdpca = RandomizedPCA(n_components=nc).fit(X_Train_Des)
X_Train_Des_rpca = rdpca.transform(X_Train_Des)
X_Test_Des_rpca = rdpca.transform(X_Test_Des)
X_val_Des_rpca = rdpca.transform(X_val_Des)
'''

lda = LinearDiscriminantAnalysis(n_components=nc).fit(X_Train_Des,Y_Train.ravel());
X_Train_Des_lda= lda.transform(X_Train_Des)
X_Test_Des_lda = lda.transform(X_Test_Des)
X_val_Des_lda = lda.transform(X_val_Des)
print(' 75 Finish')


lda = LinearDiscriminantAnalysis(n_components=nc).fit(X_Train_Fin,Y_Train.ravel());
X_Train_Fin_lda= lda.transform(X_Train_Fin)
X_Test_Fin_lda = lda.transform(X_Test_Fin)
X_val_Fin_lda = lda.transform(X_val_Fin)
print(' Fininumber Finish')
nc =nc*2;


lda = LinearDiscriminantAnalysis(n_components=nc).fit(X_Train_UPC,Y_Train.ravel());
X_Train_UPC_lda= lda.transform(X_Train_UPC)
X_Test_UPC_lda = lda.transform(X_Test_UPC)
X_val_UPC_lda = lda.transform(X_val_UPC)

'''
rdpca = RandomizedPCA(n_components=nc).fit(X_Train_UPC)
X_Train_UPC_rpca = rdpca.transform(X_Train_UPC)
X_Test_UPC_rpca = rdpca.transform(X_Test_UPC)
X_val_UPC_rpca = rdpca.transform(X_val_UPC)
'''
print(' All withinin Finish')

X_Train_in = np.hstack([X_Train_Des_lda,X_Train_new,X_Train_Fin_lda,X_Train_UPC_lda])
X_Test_in = np.hstack([X_Test_Des_lda,X_Test_new,X_Test_Fin_lda,X_Test_UPC_lda])
X_val_in = np.hstack([X_val_Des_lda,X_val_new,X_val_Fin_lda,X_val_UPC_lda])


print(X_Train_in.shape)
print(X_Test_in.shape)
print(X_val_in.shape)

t0 = time()
clf = RandomForestClassifier(n_estimators= 500)
clf.fit(X_Train_in , Y_Train.ravel())
pre =clf.predict(X_Test_in)
scio.savemat('Prediction1.3', {'Pre':pre})
pre_val =clf.predict(X_val_in)
score = clf.score(X_val_in, Y_val.ravel())
print("done in %0.3fs" % (time() - t0))
print(score)
