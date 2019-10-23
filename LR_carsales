##import sklearn as skl
import pandas as pd
"""from sklearn import datasets
iris=datasets.load_boston()
print(iris.DESCR)"""
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import linear_model
df=pd.read_csv('carprices.csv')
##print(df)
X=df[['mileage','age']]
y=df['sellprice']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=0)
##print(X_train,X_test,y_train,y_test)
linear=linear_model.LinearRegression()
##train the mocel using training sets and check the score
linear.fit(X_train,y_train)
linear.score(X_train,y_train)
## equation coefficients and intercepts
print('Coefficient -b1,b2:\n',linear.coef_)
print('Intercept- bo:\n',linear.intercept_)
predicted=linear.predict(X_test)
print("**********Predicted Values*****")
print(predicted)
print("****************** Actual Taget Values @ y_test data")
print(y_test)
####print RMSE details using score 
##print("RMSE Value  : ",linear.score(X,y))
print('Accurancy of linear regression classifier on test set : {:2f}'.format(linear.score(X_test,y_test)))



