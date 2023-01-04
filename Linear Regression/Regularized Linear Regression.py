# -*- coding: utf-8 -*-
"""
Regularized Regression
@author: kholm
dataset: https://www.kaggle.com/datasets/roger1315/automobiles?resource=download&select=auto-mpg.csv
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score


from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNet



df = pd.read_csv(r"auto-mpg-processed.csv")


print(df.head())
print(df.dtypes)


x = df.drop(["mpg"],axis=1)
y = df["mpg"]

#train test split
X_train, X_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=42)


#scale data
scl = StandardScaler()
X_train_scaled = scl.fit_transform(X_train)
X_test_scaled = scl.transform(X_test)



"""
-------
Linear Regression
-------

"""

lin_reg = LinearRegression(fit_intercept = True)
lin_reg.fit(X_train_scaled,y_train)

trainscore = lin_reg.score(X_train_scaled,y_train)
y_pred = lin_reg.predict(X_test_scaled)

print()
print("Lin Reg R^2")
print("Training: {}".format(trainscore))
print("Testing: {}".format(r2_score(y_pred,y_test)))



"""
-------
LASSO Regression
-------
"""

lin_reg = Lasso(fit_intercept = True,alpha=1)
lin_reg.fit(X_train_scaled,y_train)

trainscore = lin_reg.score(X_train_scaled,y_train)
y_pred = lin_reg.predict(X_test_scaled)

print()
print("LASSO R^2")
print("Training: {}".format(trainscore))
print("Testing: {}".format(r2_score(y_pred,y_test)))
print()
print("Coefficients Selected")
for i in range(0,len(lin_reg.coef_)):
    print("{}: {}".format(X_train.columns[i],lin_reg.coef_[i]))



"""
-------
Ridge Regression
-------
"""


lin_reg = Ridge(fit_intercept = True,alpha=1)
lin_reg.fit(X_train_scaled,y_train)

trainscore = lin_reg.score(X_train_scaled,y_train)
y_pred = lin_reg.predict(X_test_scaled)

print()
print("Ridge R^2")
print("Training: {}".format(trainscore))
print("Testing: {}".format(r2_score(y_pred,y_test)))




"""
-------
Elastic Net Regression
-------
"""


lin_reg = ElasticNet(fit_intercept = True,alpha=1,l1_ratio=0.5)
lin_reg.fit(X_train_scaled,y_train)

trainscore = lin_reg.score(X_train_scaled,y_train)
y_pred = lin_reg.predict(X_test_scaled)

print()
print("Elastic Net R^2")
print("Training: {}".format(trainscore))
print("Testing: {}".format(r2_score(y_pred,y_test)))

