# This code applies different binary classifiers 
# to two different groups of Control vs Disease classes
# Classifiers are SVM, Logistic Regression, Random forests,
# Neural Networks, XGBoost, k-Nearest Neighbors
# The output is the confusion matrix of classifiers.

# Author: Abouzar Ghavami
# Email: ghavamip@gmail.com



import pandas as pd
import random
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn import ensemble
import time

print('Start time = ', time.ctime())
fn = 'normalized.txt'
fny = 'blood_samples.txt'
df1 = pd.read_csv(fn, header = None, sep = '\t', dtype = object)
df = df1.transpose()
dfy = pd.read_csv(fny, sep = '\t', dtype = object)
time1 = time.time()
controls = dfy.loc[dfy['category'] == 'Cont']['sample']
diseaseds = dfy.loc[dfy['category'] == 'SSc']['sample']
others = dfy.loc[dfy['category'] == 'Other']['sample']
print('Headers found in ', time.time() - time1)
controls_headers = controls.tolist()
diseaseds_headers = list(diseaseds)
others_headers = list(others)
headers = df.iloc[0].values[1:]
v1 = df.values
v = v1[1:,1:]
vy = dfy.values

samples = [v1[i][0] for i in range(1, len(v1))]
samples_set = set(samples)
sample_dict = dict()
for i in range(len(samples)):
    sample_dict[samples[i]] = [i]
for i in range(len(vy)):
    if vy[i][0] in samples_set:
        sample_dict[vy[i][0]].append(vy[i][1])
y = [0 for i in range(len(v))]
for i in range(len(v)):
    if sample_dict[samples[i]][1] == 'Cont':
        y[i] = 0
    else:
        y[i] = 1
v_set = []
v_set_size = []
space_size = dict()
for j in range(1,len(headers)):
    v_j = [v[i][j] for i in range(len(v))]
    v_j_set = set(v_j)
    v_set.append(v_j_set)
    space_size[headers[j]] = len(v_j_set)

X = [v[i, :] for i in range(len(v))]
y = [float(v[i][len(headers) - 1]) for i in range(len(v))]
kfold = 10
indices = np.arange(len(v))
np.random.shuffle(indices)
rf_clfs = []
rf_cms = []
nn_clfs = []
nn_cms = []
logr_clfs = []
logr_cms = []
svm_clfs = []
svm_cms = []
knn_clfs = []
knn_cms = []
xg_cms = []
original_params = {'n_estimators': 1000, 'max_leaf_nodes': 4, 'max_depth': None, 'random_state': 2,
                   'min_samples_split': 5}
params = dict(original_params)

for i in range(kfold):
    val_indices = indices[int(i * len(v)/kfold) : int((i + 1) * len(v)/kfold)]
    train_indices = np.concatenate((indices[0 : int(i * len(v)/kfold)], indices[int((i + 1) * len(v)/kfold) : len(v)]))
    X_train = [X[j] for j in train_indices]
    y_train = [y[j] for j in train_indices]
    X_val = np.asarray([X[j] for j in val_indices])
    y_val = np.asarray([y[j] for j in val_indices])
    rf_clf = RandomForestClassifier(max_depth=1000, random_state=0)
    rf_clf.fit(X, y)
    rf_clfs.append(rf_clf)
    y_pred = rf_clf.predict(X_val)
    rf_cm = confusion_matrix(y_pred, y_val)
    rf_cms.append(rf_cm)
    
    nn_clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
                     hidden_layer_sizes=(40, 30, 20, 1), random_state=1)
    nn_clf.fit(X, y)
    nn_clfs.append(nn_clf)
    y_pred = nn_clf.predict(X_val)
    nn_cm = confusion_matrix(y_pred, y_val)
    nn_cms.append(nn_cm)
    
    logr_clf = LogisticRegression(random_state=0)
    logr_clf.fit(X, y)
    logr_clfs.append(logr_clf)
    y_pred = logr_clf.predict(X_val)
    logr_cm = confusion_matrix(y_pred, y_val)
    logr_cms.append(logr_cm)
    
    svm_clf = svm.SVC(gamma=0.001)
    svm_clf.fit(X, y)
    svm_clfs.append(svm_clf)
    y_pred = svm_clf.predict(X_val)
    svm_cm = confusion_matrix(y_pred, y_val)
    svm_cms.append(svm_cm)
    
    knn_clf = KNeighborsClassifier(n_neighbors=3)
    knn_clf.fit(X, y)
    knn_clfs.append(knn_clf)
    y_pred = knn_clf.predict(X_val)
    knn_cm = confusion_matrix(y_pred, y_val)
    knn_cms.append(knn_cm)
    
    
    xg_clf = ensemble.GradientBoostingClassifier(**params)
    xg_clf.fit(X_train, y_train)
    xg_clfs.append(knn_clf)
    y_pred = xg_clf.predict(X_val)
    xg_cm = confusion_matrix(y_pred, y_val)
    xg_cms.append(xg_cm)
    
print('random forest confusion matrices: ', rf_cms)
print('logistic regression confusion matrices: ', logr_cms)
print('k-nearest neighbors confusion matrices: ', knn_cms)
print('support vector machines confusion matrices: ', svm_cms)
print('Gradient Boost confusion matrices: ', xg_cms)
