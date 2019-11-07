import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import linear_model
import seaborn as sn
import matplotlib.pyplot as plt
##url="https://archive.ics.uci.edu/ml/machine-learning-databases/00383/risk_factors_cervical_cancer.csv"
df=pd.read_csv('Titanic_logistic_models.csv')
##print(moddf)
## 'Sex' field is string and can't be used for regression so used function to conver to numeric like when
## when sex is male then '1' anf when female then '0'
def tran_Sex(x):
    if x=='Male':
        return 1
    if x=='Female':
        return 0
df['Tran_Sex']=df['Sex'].apply(tran_Sex)
print(df)
## Drop NaN rows in the data set
moddf=df.dropna()
##print(df)
X=moddf[['Age','Tran_Sex','Class','SiblingSpouse','ParentChild']]
y=moddf['Survived1st800']
##print(X)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=0)
##print(X_train,X_test,y_train,y_test)
logreg=linear_model.LogisticRegression()
logreg.fit(X_train,y_train)
logreg.score(X_train,y_train)
print('Coefficient -b1,b2, etc:\n',logreg.coef_)
print('Intercept- bo:\n',logreg.intercept_)
y_pred=logreg.predict(X_test)
print(y_pred)
print(y_test)
print('Accurancy of logistic regression classifier on test set : {:2f}'.format(logreg.score (X_test,y_test)))                         
print("RMSE Value  : ",logreg.score(X,y))
confusion_matrix = pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted'])
print(confusion_matrix)
## print classification report
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred) )
sn.heatmap(confusion_matrix,annot=True)
print('Accuracy: ',metrics.accuracy_score(y_test, y_pred))
confusion_matrix = pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted'])
sn.heatmap(confusion_matrix, annot=True)
plt.show()














