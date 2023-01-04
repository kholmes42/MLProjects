# -*- coding: utf-8 -*-
"""
Automobile Sklearn Simple and Multiple Linear Regression
@author: kholm
dataset: https://www.kaggle.com/datasets/roger1315/automobiles?resource=download&select=auto-mpg.csv
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt



"""
-----------------------------------
Perform EDA and data wrangling
-----------------------------------
"""


df = pd.read_csv(r"auto-mpg.csv")

#EDA
print(df.head())
print(df.shape)
print(df.describe())
print(df.dtypes)

plt.figure()
sns.pairplot(df)


df = df.replace("?",np.nan)

df["horsepower"] = pd.to_numeric(df["horsepower"])
df.dropna(inplace=True)
df.drop(["origin","car name"],axis=1,inplace=True)

print(df.shape)


#create age column
df["model year"] = "19" + df["model year"].astype(str)
df["age"] = pd.Timestamp.today().year - df["model year"].astype(int) 

plt.figure()
sns.heatmap(df.corr(), annot=True,cmap="cubehelix")


#create cleaned dataset
df.to_csv("auto-mpg-processed.csv",index=False)


"""
-----------------------------------
Perform Simple Linear Regression
-----------------------------------
"""
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

scale = StandardScaler()

varname = "horsepower"

df = pd.read_csv(r"auto-mpg-processed.csv")

#train test split
X_train, X_test, y_train, y_test = train_test_split(df[[varname]], df["mpg"], test_size=0.2, random_state=42)

#normalize data
X_train_scale = scale.fit_transform(X_train)
X_test_scale = scale.transform(X_test)

lin_reg = LinearRegression()


lin_reg.fit(X_train_scale,y_train)
y_pred = lin_reg.predict(X_test_scale)
print()
print("Simple LR")
print("Training R^2: {}".format(lin_reg.score(X_train_scale,y_train)))
print("Testing R^2: {}".format(r2_score(y_pred,y_test)))

plt.figure()
plt.scatter(X_test_scale*(np.sqrt(scale.var_))+scale.mean_, y_test)
plt.plot(X_test_scale*(np.sqrt(scale.var_))+scale.mean_,y_pred,color="red")
plt.ylabel("MPG")
plt.xlabel(varname)


"""
-----------------------------------
Perform Multiple Linear Regression
-----------------------------------
"""

from statsmodels.stats.outliers_influence import variance_inflation_factor

varnames = ["displacement","acceleration"]

df = pd.read_csv(r"auto-mpg-processed.csv")


# VIF dataframe
vif_data = pd.DataFrame()
vif_data["feature"] = df[varnames].columns
  
# calculating VIF for each feature
vif_data["VIF"] = [variance_inflation_factor(df[varnames].values, i)
                          for i in range(len(df[varnames].columns))]
print()
print(vif_data)

#OK to proceed, VIF < 5

scale = StandardScaler()


#train test split
X_train, X_test, y_train, y_test = train_test_split(df[varnames], df["mpg"], test_size=0.2, random_state=42)

#normalize data
X_train_scale = scale.fit_transform(X_train)
X_test_scale = scale.transform(X_test)

lin_reg = LinearRegression()


lin_reg.fit(X_train_scale,y_train)
y_pred = lin_reg.predict(X_test_scale)

print("Multiple LR")
print("Training R^2: {}".format(lin_reg.score(X_train_scale,y_train)))
print("Testing R^2: {}".format(r2_score(y_pred,y_test)))

print(lin_reg.coef_)

