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
from sklearn.ensemble import ExtraTreesClassifier

print('Start')

# Input the Training matrix
X_ori = np.load( "X_train.npy" )
print(np.shape(X_ori ))

clf = ExtraTreesClassifier()
clf = clf.fit(X, y)
'''
# Input the label matrix
Data = sio.loadmat('Y_75_train.mat')
Y = Data['Y_75_train']

# This dataset is way to high-dimensional. Better do PCA:
pca = PCA(n_components=200)

# Maybe some original features where good, too?
selection = SelectKBest(k=1)

# Build estimator from PCA and Univariate selection:

combined_features = FeatureUnion([("pca", pca), ("univ_select", selection)])

# Use combined features to transform dataset:
X_train_com = combined_features.fit(X_ori, Y.ravel()).transform(X_ori)

# Use PCA only:
#pca = PCA(n_components=200)
#X_train_PCA = pca.fit(X_ori).transform(X_ori)
'''
'''
# Use lDA only:
lda = LinearDiscriminantAnalysis(n_components=20)
X_train_LDA = lda.fit(X_ori, Y.ravel()).transform(X_ori)

print(np.shape(X_train_LDA  ))
'''
print('Finish')
sys.exit(0)

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
'''
