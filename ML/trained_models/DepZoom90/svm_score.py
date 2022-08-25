import numpy as np
from sklearn import svm
import pickle

from numpy import mean
from numpy import std
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn import svm

x_train = np.load('x_train.npy')
y_train = np.load('y_train.npy')

x_test = np.load('x_test.npy')
y_test = np.load('y_test.npy')

print(x_train.shape, x_test.shape)
print(y_train, y_test)

count_1 = 0
for i in y_train:
    if i == 1:
        count_1 +=1

count_2 = 0
for i in y_test:
    if i == 1:
        count_2 +=1

print(count_2)

# prepare the cross-validation procedure
cv = KFold(n_splits=15, random_state=1, shuffle=True)
# create model
model = svm.SVC(kernel='rbf')
model = model.fit(x_train, y_train)
# evaluate model
scores = cross_val_score(model, x_train, y_train, scoring='accuracy', cv=cv, n_jobs=-1)
# report performance
print('Accuracy: %.3f (%.3f)' % (mean(scores), std(scores)))

#Predict the response for test dataset
y_pred = model.predict(x_test)
from sklearn import metrics
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

x_new_data = np.load('dep_out_data_zoom_fa_6.npy', allow_pickle=True)
print(x_new_data.shape)