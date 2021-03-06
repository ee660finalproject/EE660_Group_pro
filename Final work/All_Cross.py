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
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingClassifier

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


Nt = 30000;
Nv = 5000;
nc = 68;


X = np.load("X_ALL_1400_OLD.npy")
#X = np.load("X_ALL_1400_bia.npy")
Y = np.load("Y_ALL.npy")
index = np.load("index.npy")
#X_Test = np.load("X_Test_14000.npy")
print(X.shape)

X_Train  = X [index[:,0:Nt].ravel(),:];
Y_Train = Y [index[:,0:Nt].ravel(),:];
X_val = X[index[:,-Nv:].ravel(),:]
Y_val = Y[index[:,-Nv:].ravel(),:]
print(' Input Finish')
'''

###############################  Feature Processing   ##############################
X_Train_week = X_Train[:,0:7];
X_Train_Des = X_Train[:,7:75];
X_Train_new = X_Train[:,75:75+20];
X_Train_Fin = X_Train[:,75+20:75+20+5195];
X_Train_UPC = X_Train[:,75+20+5195:];

#X_Test_week = X_Test[:,0:7];
#X_Test_Des = X_Test[:,7:75];
#X_Test_new = X_Test[:,75:75+20];
#X_Test_Fin = X_Test[:,75+20:75+20+5195];
#X_Test_UPC = X_Test[:,75+20+5195:];

X_val_week = X_val[:,0:7];
X_val_Des = X_val[:,7:75];
X_val_new = X_val[:,75:75+20];
X_val_Fin = X_val[:,75+20:75+20+5195];
X_val_UPC = X_val[:,75+20+5195:];


#lda = LinearDiscriminantAnalysis(n_components=nc).fit(X_Train_Des,Y_Train.ravel());
#X_Train_Des_lda= lda.transform(X_Train_Des)
#X_Test_Des_lda = lda.transform(X_Test_Des)
#X_val_Des_lda = lda.transform(X_val_Des)

print(' 75 Finish')
'''
'''
#rdpca = RandomizedPCA(n_components=nc).fit_transform(X_Train_Des)
X_Train_Des = preprocessing.normalize(X_Train_Des)
X_Test_Des = preprocessing.normalize(X_Test_Des)
X_val_Des = preprocessing.normalize(X_val_Des)
lda = LinearDiscriminantAnalysis(n_components=nc).fit(X_Train_Des,Y_Train.ravel());
X_Train_Des_lda= lda.transform(X_Train_Des)
X_Test_Des_lda = lda.transform(X_Test_Des)
X_val_Des_lda = lda.transform(X_val_Des)
print(' 75 Finish')

rdpca = RandomizedPCA(n_components=nc).fit(X_Train_Fin)
X_Train_Fin_rpca = rdpca.transform(X_Train_Fin)
X_Test_Fin_rpca = rdpca.transform(X_Test_Fin)
X_val_Fin_rpca = rdpca.fit_transform(X_val_Fin)

print(' Fininumber Finish')

nc =nc*2;

rdpca = RandomizedPCA(n_components=nc).fit(X_Train_UPC)
X_Train_UPC_rpca = rdpca.transform(X_Train_UPC)
X_Test_UPC_rpca = rdpca.transform(X_Test_UPC)
X_val_UPC_rpca = rdpca.fit_transform(X_val_UPC)

print(' All withinin Finish')


X_Train_in = np.hstack([X_Train_Des_lda,X_Train_new,X_Train_Fin_rpca,X_Train_UPC_rpca])
#X_Test_in = np.hstack([X_Test_Des_lda,X_Test_new,X_Test_Fin_rpca,X_Test_UPC_rpca])
X_val_in = np.hstack([X_val_Des_lda,X_val_new,X_val_Fin_rpca,X_val_UPC_rpca])

#X_Train_in = np.hstack([X_Train_Des,X_Train_new])
#X_Test_in = np.hstack([X_Test_Des,X_Test_new])
#X_val_in = np.hstack([X_val_Des,X_val_new])


#np.save('X_Train_out_5000',X_Train_in)
#np.save('X_Test_out_5000',X_Test_in)
#np.save('X_val_out_5000',X_val_in)


X_Train_in = np.load('X_Train_out_5000.npy')
X_Test_in = np.load('X_Test_out_5000.npy')
X_val_in = np.load('X_val_out_5000.npy')


#X_Train_in = np.hstack([X_Train_Des])
#X_Test_in = np.hstack([X_Test_Des])
#X_val_in = np.hstack([X_val_Des])
#X_Train_in = np.hstack([X_Train_Des_rpca,X_Train_new])
#X_Test_in = np.hstack([X_Test_Des_rpca,X_Test_new])
#X_val_in = np.hstack([X_val_Des_rpca,X_val_new])
#X_Train_in = np.hstack([X_Train_Des,X_Train_new])
#X_Test_in = np.hstack([X_Test_Des,X_Test_new])
#X_val_in = np.hstack([X_val_Des,X_val_new])
'''
X_Train_in = np.load('X_Train_in_60000.npy');
X_Train_in = X_Train_in [0:30000,:]
#X_Test_in = np.load('X_Test_in_10000.npy');
#X_val_in = np.load('X_Val_in_10000.npy');
'''
print(X_Train_in.shape)
#print(X_Test_in.shape)
#print(X_val_in.shape)
'''
X_Corss = np.vstack([X_Train_in])
Y_Corss = np.vstack([Y_Train])
X_Corss = preprocessing.normalize(X_Train_in)

#clf = SVC( C=1000, gamma =0.01).fit(X_Train_in,Y_Train.ravel())
#score = clf.score(X_Train_in, Y_Train.ravel())
#print(score)

clf = AdaBoostClassifier(  DecisionTreeClassifier(max_depth=15),
    n_estimators=600,
    learning_rate=1).fit(X_Train_in,Y_Train.ravel())

score_1 = clf.score(X_Train_in, Y_Train.ravel())
#score_2 = clf.score(X_val_in, Y_val.ravel())
print(score_1)


#X_Corss = X_Corss[:,7:75];
print(X_Corss.shape)
'''
################### Random Forest ########################
# Split the dataset in two equal parts
X_train, X_test, y_train, y_test = train_test_split(
    X_Corss, Y_Corss.ravel(), test_size=0.5, random_state=0)

# Set the parameters by cross-validation

tuned_parameters = [{ 'max_depth': [28],
                    'n_estimators': [1000]}
                ]

scores = ['precision', 'recall']

for score in scores:
    print("# Tuning hyper-parameters for %s" % score)
    print()

    clf = GridSearchCV(RandomForestClassifier(), tuned_parameters, 
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


################### Multiclass SVM ########################
# Split the dataset in two equal parts
X_train, X_test, y_train, y_test = train_test_split(
    X_Corss, Y_Corss.ravel(), test_size=0.5, random_state=0)

# Set the parameters by cross-validation

tuned_parameters = [{'kernel': ['poly','rbf'], 'gamma': [ 10,1,0.1,0.01,0.001],
                     'C': [1000],'decision_function_shape': ['ovo']}
                ]


#tuned_parameters = [{'kernel': ['rbf']}]
                
scores = ['precision', 'recall']

for score in scores:
    print("# Tuning hyper-parameters for %s" % score)
    print()

    clf = GridSearchCV(SVC(), tuned_parameters, cv=5,
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


    ################### Adaboost Tree ########################
# Split the dataset in two equal parts
X_train, X_test, y_train, y_test = train_test_split(
    X_Corss, Y_Corss.ravel(), test_size=0.2 , random_state=0)

# Set the parameters by cross-validation

tuned_parameters = [{ 'learning_rate': [0.1,0.3,0.5,1,1.2],
                    'n_estimators': [50,100,200,400,600,800]}
                ]

scores = ['precision', 'recall']

for score in scores:
    print("# Tuning hyper-parameters for %s" % score)
    print()

    clf = GridSearchCV( AdaBoostClassifier(), tuned_parameters, 
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


#clf = SVC( clf.best_params_).fit(X_Train_in,Y_Train.ravel())
#score = clf.score(X_val_in, Y_val.ravel())
#print(score)

clf = RandomForestClassifier( clf.best_params).fit(X_Train_in,Y_Train.ravel())
score = clf.score(X_val_in, Y_val.ravel())
print(score)
'''
