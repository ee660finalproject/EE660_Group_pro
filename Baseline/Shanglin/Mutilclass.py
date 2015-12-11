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


'''
Data = sio.loadmat('X_75_train.mat')
X_train = Data['X_75_train']
Data = sio.loadmat('X_75_test.mat')
X_test = Data['X_75_test']

Data = sio.loadmat('Y_75_train.mat')
Y_train = Data['Y_75_train']
Data = sio.loadmat('Y_75_test.mat')
Y_test = Data['Y_75_test']

Data = sio.loadmat('order_for_60000.mat')
index = Data['order2']


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

'''
'''
#SAMY = sio.loadmat('Y.mat')

X_ori = np.load( "X_train.npy" )
Data = sio.loadmat('Y_75_train.mat')
Y = Data['Y_75_train']

# This dataset is way to high-dimensional. Better do PCA:
pca = PCA(n_components=200)

# Maybe some original features where good, too?
selection = SelectKBest(k=1)

# Use combined features to transform dataset:
# This dataset is way to high-dimensional. Better do PCA:

#pca = PCA(n_components=200)
#X_train_PCA = pca.fit(X_ori).transform(X_ori)
lda = LinearDiscriminantAnalysis(n_components=200)
X_train_LDA = lda.fit(X_ori, Y.ravel()).transform(X_ori)


#print(pca.explained_variance_ratio_) 
#print(np.shape(X_train_PCA  ))
'''
#X_train_PCA = np.load("X_train_PCA.npy")

X_train_LDA = np.load("X_train_LDA.npy")
Data = sio.loadmat('Y_75_train.mat')
Y = Data['Y_75_train']

N = 30000;
index = range(2*N)
random.shuffle(index)

X_train = X_train_LDA[index[1:N], :]
Y_train = Y[index[1:N],:]
X_test = X_train_LDA[index[-N]:,:]
Y_test = Y[index[-N]:,:]
#P = OneVsRestClassifier(LinearSVC(random_state=0)).fit(X_train, Y_train).predict(X_test)
#A = accuracy_score(Y_test, P)
#print(A)

clf = svm.SVC(decision_function_shape='ovo')
clf.fit(X_train, Y_train.ravel()) 
score = clf.score(X_test, Y_test.ravel())
print('Finish')
sys.exit(0)

#clf = tree.DecisionTreeClassifier()
#clf = clf.fit(X_train, Y_train.ravel())
#P = clf.predict(X_test)
#score = accuracy_score(Y_test, P)
#print(score )
