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
print('Start')

# the index of all 6000 samples for training 
Data = sio.loadmat('order_for_60000.mat')
index = Data['order2']

X_train_LDA = np.load("X_train_LDA.npy")
X_ori = np.load( "X_train.npy" )

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
'''
X_train_cho = X_ori;
clf = ExtraTreesClassifier()
clf = clf.fit(X_train_cho, Y.ravel())
model = SelectFromModel(clf, prefit=True)
X_new = model.transform(X_train_cho)
'''
print(X_base.shape)
print(X_train_com.shape)


lda = LinearDiscriminantAnalysis(n_components=100)
X_train_LDA = lda.fit(X_base, Y.ravel()).transform(X_ori)
print(np.shape(X_train_LDA  ))

lda = LinearDiscriminantAnalysis(n_components=100)
X_com_LDA = lda.fit(X_train_com, Y.ravel()).transform(X_train_com)
print(np.shape(X_com_LDA  ))


print(X_new.shape) 
X_train = X_train_LDA[index[0:N], :]
X_train_add = X_com_LDA[index[0:N], :]
Y_train = Y[index[0:N],:]
X_test = X_train_LDA[index[-N:],:]
X_test_add = X_com_LDA[index[-N:],:]
Y_test = Y[index[-N:],:]



clf = RandomForestClassifier(n_estimators=200);
clf.fit(X_train, Y_train.ravel())
pre = clf.predict(X_test)
A = accuracy_score(pre, Y_test.ravel())
print(A)

clf = RandomForestClassifier(n_estimators=200);
clf.fit(X_train_add, Y_train.ravel())
pre = clf.predict(X_test_add)
A = accuracy_score(pre, Y_test.ravel())
print(A)
'''
#P = OneVsRestClassifier(LinearSVC(random_state=0)).fit(X_train, Y_train).predict(X_test)
#A = accuracy_score(Y_test, P)
#print(A)

'''
clf = svm.SVC(decision_function_shape='ovo')
clf.fit(X_train, Y_train.ravel()) 
score = clf.score(X_test, Y_test.ravel())

clf = svm.SVC(decision_function_shape='ovo')
clf.fit(X_train_add, Y_train.ravel()) 
score = clf.score(X_test_add, Y_test.ravel())
'''
dt = DecisionTreeClassifier(max_depth=9, min_samples_leaf=1)
bdt_discrete = AdaBoostClassifier(
     base_estimator=dt,
    n_estimators=600,
    learning_rate=1,
    algorithm="SAMME"
    )
bdt_discrete.fit(X_train, Y_train.ravel())
P = bdt_discrete.predict(X_test)
accuracy_score(P, Y_test.ravel())
'''
print('Finish')
sys.exit(0)
