import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import linear_model
import seaborn as sn
import matplotlib.pyplot as plt
import numpy as np
x_train = pd.read_table('X_train.txt',header = None, sep = " ")
##print(x_train.head())
y=pd.read_table('y_train.txt',header = None, sep = " ")
##Convert column to list
y_train=y.values.tolist()
##print(y_train)
x_test = pd.read_table('X_test.txt',header = None, sep = " ")
yt=pd.read_table('y_test.txt',header = None, sep = " ")
y_test=yt.values.tolist()
##print(y_train)
logreg=linear_model.LogisticRegression()
logreg.fit(x_train,y_train)
logreg.score(x_train,y_train)
print('Coefficient -b1,b2, etc:\n',logreg.coef_)
print('Intercept- bo:\n',logreg.intercept_)
import time
start=time.time()
y_pred=logreg.predict(x_test)
end=time.time()
print("time taken to Prediction Execution :",format(end-start))
print("Predicted Values:", y_pred)
print("Actual Values",y_test)
print('Accurancy of logistic regression classifier on test set : {:2f}'.format(logreg.score (x_test,y_test)))                         



"""
X_train=open('X_train.txt')
y_train=open('y_train.txt')
X_test=open('X_test.txt')
y_test=open('y_test.txt')
###open and split into columns and rows from tct file
X=open('X_train.txt')
x_temp=X.read()
x_t=x_temp.split('\n')
del x_temp
r=len(x_t)-1
c=x_t[0].count(' ')+1
print(c)
print(r)
y=open('y_train.txt')
y_temp=y.read()
y_t=y_temp.split('\n')
del y_temp
r=len(y_t)-1
c=y_t[0].count(' ')+1
print(c)
print(r)
X=open('X_test.txt')
x_temp=X.read()
x_te=x_temp.split('\n')
del x_temp
r=len(x_te)-1
c=x_te[0].count(' ')+1
print(c)
print(r)
logreg=linear_model.LogisticRegression()
logreg.fit(x_t,y_t)
logreg.score(x_t,y_t)
print('Coefficient -b1,b2, etc:\n',logreg.coef_)
print('Intercept- bo:\n',logreg.intercept_)
import time
start=time.time()
y_pred=logreg.predict(x_te)
end=time.time()
print("time taken to Prediction Execution :",format(end-start))
print(y_pred)
##print(y_test)

"""






















