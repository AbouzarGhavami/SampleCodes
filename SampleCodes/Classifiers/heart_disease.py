import pandas as pd
import random
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression

fn = 'heart.csv'
df = pd.read_csv(fn, dtype = object)

headers = df.columns.values
v = df.values
v_set = []
v_set_size = []
space_size = dict()
for j in range(len(headers)):
    v_j = [v[i][j] for i in range(len(v))]
    v_j_set = set(v_j)
    v_set.append(v_j_set)
    space_size[headers[j]] = len(v_j_set)

X = [v[i][0 : len(headers) - 1] for i in range(len(v))]
y = [int(v[i][len(headers) - 1]) for i in range(len(v))]
kfold = 10
indices = np.arange(len(v))
np.random.shuffle(indices)
rf_clfs = []
rf_cms = []
nn_clfs = []
nn_cms = []
logr_clfs = []
logr_cms = []
for i in range(kfold):
    val_indices = indices[int(i * len(v)/kfold) : int((i + 1) * len(v)/kfold)]
    train_indices = np.concatenate((indices[0 : int(i * len(v)/kfold)], indices[int((i + 1) * len(v)/kfold) : len(v)]))
    X_train = [X[j] for j in train_indices]
    y_train = [y[j] for j in train_indices]
    X_val = np.asarray([X[j] for j in val_indices])
    y_val = np.asarray([y[j] for j in val_indices])
    rf_clf = RandomForestClassifier(max_depth=9, random_state=0)
    rf_clf.fit(X, y)
    rf_clfs.append(rf_clf)
    y_pred = rf_clf.predict(X_val)
    cm = confusion_matrix(y_pred, y_val)
    rf_cms.append(cm)
    
    nn_clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
                     hidden_layer_sizes=(40, 30, 20, 1), random_state=1)
    nn_clf.fit(X, y)
    nn_clfs.append(nn_clf)
    y_pred = nn_clf.predict(X_val)
    cm = confusion_matrix(y_pred, y_val)
    nn_cms.append(cm)
    
    logr_clf = LogisticRegression(random_state=0).fit(X, y)
    logr_clf.fit(X, y)
    logr_clfs.append(nn_clf)
    y_pred = logr_clf.predict(X_val)
    logr_cm = confusion_matrix(y_pred, y_val)
    logr_cms.append(logr_cm)
    
