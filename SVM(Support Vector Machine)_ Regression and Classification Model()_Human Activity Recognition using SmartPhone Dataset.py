import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import linear_model
import seaborn as sn
import matplotlib.pyplot as plt
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import learning_curve
x_train = pd.read_table('X_train.txt',header = None, sep = " ")
##print(x_train.head())
y=pd.read_table('y_train.txt',header = None, sep = " ")
##Convert column vector to 1 d array
y_train=y.values.flatten()
##print(y_train)
x_test = pd.read_table('X_test.txt',header = None, sep = " ")
yt=pd.read_table('y_test.txt',header = None, sep = " ")
y_test=yt.values.tolist()
##print(y_train)
"""Cs = np.logspace(-6, 3, 10)
parameters = [{'kernel': ['rbf'], 'C': Cs},
              {'kernel': ['linear'], 'C': Cs}]"""
C=1
Svc_classifier = SVC(kernel='linear', C=C)
Svc_classifier.fit(x_train,y_train)
print ("Coefficients :",Svc_classifier.coef_)
print ("Best Score :",Svc_classifier.intercept_)
import time
start=time.time()
y_pred=Svc_classifier.predict(x_test)
end=time.time()
print("time taken to Prediction Execution :",format(end-start))
print("Predicted Values:", y_pred)
print("Actual Values",y_test)
print("Accurancy of Support Vector Machines(SVM)classifier on test set :", Svc_classifier.score(x_test, y_test))
plt.figure(figsize=(18,7))
plt.scatter(y_test,y_pred,color="red")
plt.show()
























