# -*- coding: utf-8 -*-
"""
Dimensionality Reduction
dataset: 
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split




"""
-------------------------------------------
Loading data
-------------------------------------------
"""


df = pd.read_csv(r"kc_house_data.csv")

#EDA
print(df.head())
print(df.describe())
print(df.info())

df.drop(["id","date","zipcode"],axis=1,inplace=True)

df["age"] = 2018 - df["yr_built"]

df.drop(["yr_built"],axis=1,inplace=True)


df["renovated"] =  df["yr_renovated"].apply(lambda x:1 if x > 0 else 0)
df.drop(["yr_renovated"],axis=1,inplace=True)


"""
-------------------------------------------
all feature regression
-------------------------------------------
"""
import statsmodels.api as sm
X_train, X_test, y_train, y_test = train_test_split(df.drop(["price"],axis=1), df["price"], test_size=0.2, random_state=465)


model = sm.OLS(y_train,sm.add_constant(X_train)).fit()

print(model.summary())



"""
-------------------------------------------
brute force k best using univariate regressions
-------------------------------------------
"""
from sklearn.feature_selection import f_regression,SelectKBest
selfeat = SelectKBest(f_regression,k=3)

X_new = pd.DataFrame(data=selfeat.fit_transform(X_train,y_train),columns=selfeat.get_feature_names_out())

#much poorer on test set
model = sm.OLS(y_train.reset_index( drop=True),sm.add_constant(X_new)).fit()
print(model.summary())





"""
------------------------------------------------
PCA demo
------------------------------------------------
"""
from sklearn.datasets import load_boston
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

data = load_boston()

df2 = pd.DataFrame(data=data.data,columns=data.feature_names)
df2["target"] = data.target
df2.drop("B",inplace=True,axis=1)
print(df2.head())
print(df2.info())



pca = PCA(n_components=10).fit(df2.drop("target",axis=1))

print(pca.explained_variance_ratio_)
fig,ax = plt.subplots(1,2,figsize=(12, 6))
sns.barplot(x=np.arange(1,11),y=pca.explained_variance_ratio_,ax=ax[0])
ax[0].set_ylabel("Variance Explained")
ax[0].set_xlabel("PC")
ax[0].set_title("PCA Scree Plot")


new_x = pca.transform(df2.drop("target",axis=1))



sns.scatterplot(x=new_x[:,0],y=new_x[:,1],ax=ax[1])
ax[1].set_ylabel("PC2")
ax[1].set_xlabel("PC1")
ax[1].set_title("Data in PC1 vs PC2")


modonlyPC1_2 = LinearRegression().fit(new_x[:,:2],df2["target"])
modalldata = LinearRegression().fit(df2.drop("target",axis=1),df2["target"])

print()
print("R^2 score")
print("Full Regression: {}".format(r2_score(df2["target"],modalldata.predict(df2.drop("target",axis=1)))))
print("PC1/2 Regression: {}".format(r2_score(df2["target"],modonlyPC1_2.predict(new_x[:,:2]))))

"""
------------------------------------------------
LDA demo
------------------------------------------------
"""

from sklearn.datasets import load_iris
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix
data = load_iris()

df3 = pd.DataFrame(data=data.data,columns=data.feature_names)
df3["target"] = data.target

print(df3.head())
print(df3.info())

xdata = df3.drop("target",axis=1)

lda = LinearDiscriminantAnalysis(n_components=2)

newxdata = lda.fit_transform(xdata,df3["target"])

plt.figure()

sns.scatterplot(x=newxdata[:,0],y=newxdata[:,1],hue=df3["target"])
L=plt.legend()
plt.title("LDA of Iris Data")
plt.xlabel("LDA dim 1")
plt.ylabel("LDA dim 2")
for i in range(len(data.target_names)):
    L.get_texts()[i].set_text(data.target_names[i])
 

#fit logistic regression
logmod = LogisticRegression()
logmod.fit(newxdata,df3["target"])
y_pred = logmod.predict(newxdata)


print()
print("Logistic Reg Accuracy {}".format(accuracy_score(y_pred,df3["target"])))
print("Confusion Matrix")
print(confusion_matrix(y_pred,df3["target"]))










