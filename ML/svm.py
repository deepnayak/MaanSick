import numpy as np
from sklearn import svm
import pickle

x_train = np.load("x_train.npy", allow_pickle=True)
x_test = np.load("x_test.npy", allow_pickle=True)
y_train = np.load("y_train.npy", allow_pickle=True)
y_test = np.load("y_test.npy", allow_pickle=True)


    # x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, test_size=0.5)
    # print(len(x_train),len(x_test),len(x_val))
# print(y_train)

clf = svm.SVC(kernel='rbf') # Linear Kernel

#Train the model using the training sets
clf.fit(x_train, y_train)
print("training finished.")

# testing the model
print("testing the model...")

#Predict the response for test dataset
y_pred = clf.predict(x_test)
from sklearn import metrics
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
# Model Precision: what percentage of positive tuples are labeled as such?
print("Precision:",metrics.precision_score(y_test, y_pred))
# Model Recall: what percentage of positive tuples are labelled as such?
print("Recall:",metrics.recall_score(y_test, y_pred))
# print(len(y_test))
# w = clf.coef_
# b = clf.intercept_
# print("Weights:",w)
# print("Bias:",b)
filename = 'svm_model_dep.sav'
pickle.dump(clf, open(filename, 'wb'))