# -*- coding: utf-8 -*-
"""
XGBoost End-to-End
dataset: https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import warnings
import pickle

#read in data
warnings.filterwarnings("ignore")
df = pd.read_csv(r"titanic.csv")

print(df.head())
print(df.info())


dataused = df[["Pclass","Sex","Age","Survived"]]

dataused["Sex"] = dataused["Sex"].map({"male":0,"female":1})

dataused.dropna(inplace=True)

x = dataused[["Pclass","Sex","Age"]]
y = dataused["Survived"]




plt.figure()
sns.countplot(data=dataused,x="Survived")
plt.title("Survivor Breakdown")



X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=1)


"""
--------------------------------------
Naive use of XGBoost
--------------------------------------
"""
model = XGBClassifier( eval_metric="logloss",use_label_encoder=False).fit(X_train,y_train)

y_pred = model.predict(X_test)
print()
print("XGBoost Test Accuracy: {}".format(accuracy_score(y_pred,y_test)))



"""
--------------------------------------
Serialize and Load XGBoost Model
--------------------------------------
"""

pickle.dump(model,open("titantic_xgboost_mod_prod","wb"))

load_mod = pickle.load(open("titantic_xgboost_mod_prod","rb"))
y_pred2 = load_mod.predict(X_test)

print()
print("XGBoost Test Accuracy: {}".format(accuracy_score(y_pred2,y_test)))



"""
--------------------------------------
Examine feature importance
--------------------------------------
"""


plt.figure()
sns.barplot(x=load_mod.get_booster().feature_names,y=load_mod.feature_importances_)
plt.title("Feature Importance XGBoost")
plt.xlabel("Feature")
plt.ylabel("Importance Score")




"""
--------------------------------------
Regularize XGBoost l2 norm test
--------------------------------------
"""


lm = [0.00001,0.0001,0.001,0.01,0.1,1,50,500,100,1000,2000]
acc = []
for i in lm:
    model = XGBClassifier(reg_lambda=i, eval_metric="logloss",use_label_encoder=False).fit(X_train,y_train)
    acc.append(accuracy_score(model.predict(X_test),y_test))


#suggests max regularization
plt.figure()
sns.lineplot(x=lm,y=acc)
plt.xlabel("$\lambda$")
plt.ylabel("Accuracy")
plt.title("Regularization of XGBoost")






