# -*- coding: utf-8 -*-
"""
Exams Categorical Linear Regression
dataset: http://roycekimmons.com/tools/generated_data/exams
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression


df = pd.read_csv(r"exams.csv")

#EDA
print(df.head())
print(df.shape)
print(df.describe())
print(df.dtypes)

plt.figure()
sns.pairplot(df)


df["math score"] = preprocessing.scale(df["math score"])
df["reading score"] = preprocessing.scale(df["reading score"])
df["writing score"] = preprocessing.scale(df["writing score"])


"""
---------
Converting text categorical data to numeric
---------
"""

print(df["parental level of education"].unique())



#ordinal categorical encoding
lvlsofschool = ['some high school' ,'high school','some college', "associate's degree", "bachelor's degree", "master's degree"]

lab_en = preprocessing.LabelEncoder()
lab_en = lab_en.fit(lvlsofschool)


df["parental level of education"] = lab_en.transform(df["parental level of education"].astype(str))

#dummy variable encoding
df = pd.get_dummies(df,columns=["race/ethnicity","lunch","test preparation course","gender"])


#create dataset
y = df["math score"]
x = df.drop(["math score","reading score","writing score"],axis=1)


"""
---------
Running Regression
---------
"""


#train test split
X_train, X_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=42)

lin_reg = LinearRegression(fit_intercept = False)

lin_reg.fit(X_train,y_train)


y_pred = lin_reg.predict(X_test)

print()
print("R^2:")
print("Training: {}".format(lin_reg.score(X_train,y_train)))
print("Testing: {}".format(r2_score(y_pred,y_test)))


