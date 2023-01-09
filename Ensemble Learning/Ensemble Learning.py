# -*- coding: utf-8 -*-
"""
Ensemble Learning
dataset: https://www.kaggle.com/datasets/itsmesunil/bank-loan-modelling
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


df = pd.read_excel(r"Bank_Personal_Loan_Modelling.xlsx",sheet_name="Data")
df.set_index("ID",drop=True,inplace=True)

print(df.head())

dfdata = df.drop(["ZIP Code","Experience","CCAvg","Personal Loan"],axis=1)
ydata = dfdata["CreditCard"]

#EDA
plt.figure()
sns.countplot(dfdata,x="Family",hue="CreditCard")
plt.xlabel("Family Size")
plt.title("Family Size vs Has Credit Card")
L=plt.legend()
L.get_texts()[0].set_text('No')
L.get_texts()[1].set_text('Yes')

plt.figure()
sns.countplot(dfdata,x="Education",hue="CreditCard")
plt.xlabel("Education")
plt.title("Education vs Has Credit Card")
L=plt.legend()
L.get_texts()[0].set_text('No')
L.get_texts()[1].set_text('Yes')



#imbalanced data
plt.figure()
sns.countplot(dfdata,x="CreditCard")
plt.xlabel("Credit Card")
plt.title("Has Credit Card")



X_train, X_test, y_train, y_test = train_test_split(dfdata, ydata, test_size=0.2, random_state=465)





"""
---------------------------------------
Use ensemble of distinct ML models
--------------------------------------
"""

from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix

#3 models
logreg = LogisticRegression(C=1,solver="liblinear")
svm = SVC(C=1,kernel="linear",gamma="auto")
nb = GaussianNB()


#create ensemple
eclf1 = VotingClassifier(estimators=[('lr', logreg), ('svm', svm), ('gnb', nb)], voting='hard')
eclf1.fit(X_train,y_train)


#predict on test data
y_pred = eclf1.predict(X_test)
print(accuracy_score(y_pred,y_test))
print(classification_report(y_pred,y_test))
print(confusion_matrix(y_pred,y_test))




"""
---------------------------------------
Use AdaBoost
--------------------------------------
"""
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import r2_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

df = pd.read_csv(r"concrete_data.csv")
print(df.head())
print(df.describe())
print(df.info())


dfx = df.drop(["csMPa"],axis=1)
dfy = df["csMPa"]

X_train, X_test, y_train, y_test = train_test_split(dfx, dfy, test_size=0.2, random_state=465)


k = [10,50,100,150,200,250,300]
score = []
#train model
for i in k:
    regr = AdaBoostRegressor(random_state=0, n_estimators=i).fit(X_train,y_train)
    y_pred = regr.predict(X_test)
    score.append(r2_score(y_test,y_pred))
    
        

plt.figure()
sns.lineplot(x=k,y=score)
plt.xlabel("# of Learners")
plt.ylabel("R^2 Score")
plt.title("R^2 vs Number of Weak Learners in AdaBoost")







#classification section
df = pd.read_excel(r"Bank_Personal_Loan_Modelling.xlsx",sheet_name="Data")
df.set_index("ID",drop=True,inplace=True)

print(df.head())

dfdata = df.drop(["ZIP Code","Experience","CCAvg","Personal Loan"],axis=1)
ydata = dfdata["CreditCard"]


X_train, X_test, y_train, y_test = train_test_split(dfdata, ydata, test_size=0.2, random_state=465)




k = [10,50,100,150,200,250,300]
scoreacc = []
scoref1 = []
#train model
for i in k:
    clas = AdaBoostClassifier(random_state=0, n_estimators=i).fit(X_train,y_train)
    y_pred = clas.predict(X_test)
    scoreacc.append(accuracy_score(y_test,y_pred))
    scoref1.append(f1_score(y_test,y_pred))


plt.figure()
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()

sns.lineplot(x=k,y=scoreacc,ax=ax1)
ax1.set_xlabel("# of Learners")
ax1.set_ylabel("Accuracy")
plt.title("Errors vs Number of Weak Learners in Adaboost")
plt.xticks(k)
ax2.set_ylabel("F1 Score",color="g")
sns.lineplot(x=k,y=scoref1,ax=ax2,color="g")




"""
---------------------------------------
Use Gradient Boosting
--------------------------------------
"""

from sklearn.ensemble import GradientBoostingRegressor

df = pd.read_csv(r"concrete_data.csv")
print(df.head())
print(df.describe())
print(df.info())


dfx = df.drop(["csMPa"],axis=1)
dfy = df["csMPa"]

X_train, X_test, y_train, y_test = train_test_split(dfx, dfy, test_size=0.2, random_state=465)



k = [10,50,100,150,200,250,300]
score1 = []
score2 = []
score3 = []
#train model
for i in k:
    gbr = GradientBoostingRegressor(max_depth=3,n_estimators=i,learning_rate=0.1).fit(X_train,y_train)
    y_pred = gbr.predict(X_test)
    score1.append(r2_score(y_test,y_pred))

    gbr = GradientBoostingRegressor(max_depth=3,n_estimators=i,learning_rate=0.5).fit(X_train,y_train)
    y_pred = gbr.predict(X_test)
    score2.append(r2_score(y_test,y_pred))
    
    gbr = GradientBoostingRegressor(max_depth=3,n_estimators=i,learning_rate=1).fit(X_train,y_train)
    y_pred = gbr.predict(X_test)
    score3.append(r2_score(y_test,y_pred))



plt.figure()
sns.lineplot(x=k,y=score1,label="L rate = 0.1")
plt.xlabel("# of Learners")
plt.ylabel("R^2 Score")
plt.title("R^2 vs Number of Learners in Gradient Boosting")
sns.lineplot(x=k,y=score2,label="L rate = 0.5")
sns.lineplot(x=k,y=score3,label="L rate = 1")
plt.legend(loc="lower right")