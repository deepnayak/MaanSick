from numpy import mean
from numpy import std
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn import svm

x = np.load("x.npy", allow_pickle=True)
y = np.load("y.npy", allow_pickle=True)
y=y.astype('int') 
# prepare the cross-validation procedure
cv = KFold(n_splits=15, random_state=1, shuffle=True)
# create model
model = svm.SVC(kernel='rbf')
# evaluate model
scores = cross_val_score(model, x, y, scoring='accuracy', cv=cv, n_jobs=-1)
# report performance
print('Accuracy: %.3f (%.3f)' % (mean(scores), std(scores)))

# y=y.astype('int') 
# clf = svm.SVC(kernel='linear', C=1)
# scores = sklearn.model_selection.cross_val_score(clf, x, y, cv=10)
# print (scores)