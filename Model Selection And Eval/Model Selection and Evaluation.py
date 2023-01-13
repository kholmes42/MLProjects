# -*- coding: utf-8 -*-
"""
Model Selection & Evaluation
dataset:https://www.kaggle.com/code/datafan07/heart-disease-and-some-scikit-learn-magic
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# reading from the file
df =  pd.read_csv('heart.csv', sep=",")


dfusedx = df[["thalach","trestbps","age","chol","fbs","exang","oldpeak","slope","ca"]]
y = df["target"]

#balanced data
plt.figure()
sns.countplot(data=df,x="target")

"""
----------------------------------
Classification Metrics
----------------------------------
"""


#train different models with no hyperparameters
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier,BaggingClassifier
from sklearn.metrics import confusion_matrix,accuracy_score,precision_score,recall_score,f1_score

log_reg = LogisticRegression(max_iter=5000).fit(dfusedx,y)
y_predlog= log_reg.predict(dfusedx)

knn = KNeighborsClassifier().fit(dfusedx,y)
y_predknn = log_reg.predict(dfusedx)


gbc = GradientBoostingClassifier().fit(dfusedx,y)
y_predgbc = gbc.predict(dfusedx)


bc = BaggingClassifier().fit(dfusedx,y)
y_predbc = bc.predict(dfusedx)

print()
print("Confusion Matrices:")
print("Logistic Regression")
print(confusion_matrix(y,y_predlog))

print("KNN")
print(confusion_matrix(y,y_predknn))

print("Gradient Boosted")
print(confusion_matrix(y,y_predgbc))

print("Bagged")
print(confusion_matrix(y,y_predbc))

print()
print("Accuracies:")
print("Logistic Regression")
print(accuracy_score(y,y_predlog))

print("KNN")
print(accuracy_score(y,y_predknn))

print("Gradient Boosted")
print(accuracy_score(y,y_predgbc))

print("Bagged")
print(accuracy_score(y,y_predbc))


print()
print("Precision:")
print("Logistic Regression")
print(precision_score(y,y_predlog))

print("KNN")
print(precision_score(y,y_predknn))

print("Gradient Boosted")
print(precision_score(y,y_predgbc))

print("Bagged")
print(precision_score(y,y_predbc))


print()
print("Recall:")
print("Logistic Regression")
print(recall_score(y,y_predlog))

print("KNN")
print(recall_score(y,y_predknn))

print("Gradient Boosted")
print(recall_score(y,y_predgbc))

print("Bagged")
print(recall_score(y,y_predbc))

print()
print("F1:")
print("Logistic Regression")
print(f1_score(y,y_predlog))

print("KNN")
print(f1_score(y,y_predknn))

print("Gradient Boosted")
print(f1_score(y,y_predgbc))

print("Bagged")
print(f1_score(y,y_predbc))


"""
----------------------------------
Regression Metrics
----------------------------------
"""

dfusedx2 = df[["thalach","trestbps","age"]]
y2 = df["chol"]

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score

linmod = LinearRegression().fit(dfusedx2,y2)
y_predlin = linmod.predict(dfusedx2)
print()
print("Linear Regression Errors:")
print("MSE: {}".format(mean_squared_error(y2, y_predlin)))
print("RMSE: {}".format(np.sqrt(mean_squared_error(y2, y_predlin))))
print("MAE: {}".format(mean_absolute_error(y2, y_predlin)))
print("R^2: {}".format(r2_score(y2, y_predlin)))



"""
----------------------------------
Train Test Split Accuracy for Classification
----------------------------------
"""

from sklearn.model_selection import train_test_split


X_train, X_test, y_train, y_test = train_test_split(dfusedx, y, test_size=0.2, random_state=45)

logr = LogisticRegression().fit(X_train,y_train)

y_pred = logr.predict(X_test)

print()
print("Logistic Regression Accuracy:")
print("Training: {}".format(accuracy_score(logr.predict(X_train),y_train)))
print("Testing: {}".format(accuracy_score(y_pred,y_test)))


"""
----------------------------------
Cross Validation Using Accuracy for Classification
----------------------------------
"""

from sklearn.model_selection import cross_val_score

logr = LogisticRegression(max_iter=5000)
avgacclogreg = np.mean(cross_val_score(logr, dfusedx, y, cv=5))

knn = KNeighborsClassifier()
avgaccknn= np.mean(cross_val_score(knn, dfusedx, y, cv=5))

gbc = GradientBoostingClassifier()
avgaccgbc= np.mean(cross_val_score(gbc, dfusedx, y, cv=5))

print()
print("CV Accuracy:")
print("Logistic Reg: {}".format(avgacclogreg ))
print("KNN"" {}".format(avgaccknn ))
print("Grad Boost: {}".format(avgaccgbc))