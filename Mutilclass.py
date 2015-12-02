import scipy.io as sio
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn import svm
from sklearn import tree


SAMX = sio.loadmat('X_75.mat')
X = SAMX['X_75']
SAMY = sio.loadmat('Y.mat')
Y = SAMY['Y']
N = 40000;
X_train = X[1:N,:]
Y_train = Y[1:N,:]
X_test = X[-N:,:]
Y_test = Y[-N:,:]
#P = OneVsRestClassifier(LinearSVC(random_state=0)).fit(X_train, Y_train).predict(X_test)
#A = accuracy_score(Y_test, P)
#print(A)

#clf = svm.SVC(decision_function_shape='ovo')
#clf.fit(X_train, Y_train.ravel()) 
#score = clf.score(X_test, Y_test.ravel())

clf = tree.DecisionTreeClassifier()
clf = clf.fit(X_train, Y_train.ravel())
P = clf.predict(X_test)
score = accuracy_score(Y_test, P)
print(score )
