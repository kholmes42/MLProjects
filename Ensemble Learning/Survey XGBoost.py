# -*- coding: utf-8 -*-
"""
Created on Sat May 20 19:43:27 2023
XGBoost for classification
@author: kholm
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from xgboost import XGBClassifier,plot_importance
from sklearn.model_selection import train_test_split,validation_curve,GridSearchCV, learning_curve
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score,PrecisionRecallDisplay,plot_roc_curve
from sklearn.inspection import plot_partial_dependence
import shap

#read dataset in
df = pd.read_csv(r"kaggle-survey-2018\multipleChoiceResponses.csv",header=[0])

questions = df.iloc[0,:]
df.drop(index=[0],inplace=True)



###
#DATA PREPARATION
###

df = df[(df["Q1"] == "Male") | (df["Q1"] == "Female")]

df_relevant_info = df[["Q1","Q2","Q3","Q4","Q5","Q6","Q8","Q9","Q16_Part_1","Q16_Part_2","Q16_Part_3"]]

#preprocess/convert to relevant numerical representation
df_gender = pd.get_dummies(df_relevant_info["Q1"],drop_first=True)

df_age = ((df["Q2"].str[:2]).astype(int) + (df["Q2"].replace("80+","80-85").str[3:]).astype(int))/2
df_age.name = "Age"
df_country = pd.get_dummies(df["Q3"],drop_first=True).loc[:,["United States of America","China","India"]]

df_future_edu = df["Q4"].replace({"Bachelor’s degree":16,"Master’s degree":18,"Doctoral degree":20,"Professional degree":19,
                                 "Some college/university study without earning a bachelor’s degree":13,"No formal education past high school":12,
                                 "I prefer not to answer":None})
df_future_edu.name="EducationYrs"
df_undergrad = df["Q5"].replace({"Computer science (software engineering, etc.)":"CS",
                               "Engineering (non-computer focused)":"Eng",
                               "Mathematics or statistics":"Math"})
df_undergrad[~df_undergrad.isin(["CS","Eng","Math"])] = "Other"

df_undergrad = pd.get_dummies(df_undergrad,drop_first=True)


df_job = pd.get_dummies(df["Q6"],drop_first=True).loc[:,["Data Scientist","Software Engineer","Consultant"]] #class predict value
df_job["Other"] = np.where((df_job["Data Scientist"] != 1) & (df_job["Software Engineer"] != 1) & (df_job["Consultant"] != 1),1,0)



df_exp = pd.to_numeric(df["Q8"].replace("30 +","30").str.split("-",expand=True).iloc[:,0])
df_exp.name="Experience"
df_comp = pd.to_numeric(df["Q9"].str.replace(",","").str.replace("+","").str.split("-",expand=True).iloc[:,1])
df_comp.name="Compensation"

df_py = pd.to_numeric(df["Q16_Part_1"].str.replace("Python","1")).fillna(0)
df_py.name="Python"
df_r = pd.to_numeric(df["Q16_Part_2"].str.replace("R","1")).fillna(0)
df_r.name="R"
df_sql = pd.to_numeric(df["Q16_Part_3"].str.replace("SQL","1")).fillna(0)
df_sql.name="SQL"

#concat preprocessed data
X_data = pd.concat([df_gender,df_age,df_country,df_future_edu,df_undergrad,df_exp,df_comp,df_py,df_r,df_sql],axis=1)
Y_data = df_job.idxmax(axis=1)#np.argmax(df_job.values,axis=1)

X_data = X_data[(Y_data == "Data Scientist") | (Y_data == "Software Engineer")] 
Y_data = Y_data[(Y_data == "Data Scientist") | (Y_data == "Software Engineer")] 

print(X_data.dtypes)



###
#EDA
###

#train test split
X_train, X_test, y_train, y_test = train_test_split(X_data, Y_data, test_size=0.2, random_state=42)

fix,ax = plt.subplots()
sns.countplot(x=y_train)
plt.title("Training Classes")

ax.set_ylabel("Count")
ax.set_xlabel("Class")

plt.figure()
sns.heatmap(X_train.corr(),cmap="Spectral")
plt.title("Feature Correlations")



print(X_train.isna().mean())

le = preprocessing.LabelEncoder()

y_encoded = le.fit_transform(y_train)

###
# ML Classifier using XGBoost
###

modxgb_clf = XGBClassifier(random_state =42,use_label_encoder=False,n_estimators=250)


#Manual CV on 2 parameters
fig,ax = plt.subplots(1,2,figsize=(18,7))

curves_L1 = validation_curve(modxgb_clf,X_train,y_encoded,param_name="alpha",param_range=np.linspace(0,1,10),cv=5,scoring="f1")
sns.lineplot(x=np.linspace(0,1,10),y=np.mean(curves_L1[0],axis=1),ax=ax[0])
sns.lineplot(x=np.linspace(0,1,10),y=np.mean(curves_L1[1],axis=1),ax=ax[0])
ax[0].legend(labels=["Train","Validation"])
ax[0].set_title(r"F1-Score vs L1 Regulariztion")
ax[0].set_ylabel("F1-Score")
ax[0].set_xlabel(r"Alpha $\alpha$")

curves_L2 = validation_curve(modxgb_clf,X_train,y_encoded,param_name="lambda",param_range=np.linspace(0,1,10),cv=5,scoring="f1")
sns.lineplot(x=np.linspace(0,1,10),y=np.mean(curves_L2[0],axis=1),ax=ax[1])
sns.lineplot(x=np.linspace(0,1,10),y=np.mean(curves_L2[1],axis=1),ax=ax[1])
ax[1].legend(labels=["Train","Validation"])
ax[1].set_title(r"F1-Score vs. L2 Regularization")
ax[1].set_ylabel("F1-Score")
ax[1].set_xlabel(r"Lambda $\lambda$")

fig.suptitle("5-Fold CV")





#Run GridSearch for final model
parameters = {'alpha':np.linspace(0,0.5,5),
             'learning_rate':[0.01,0.1],"gamma":[0,1],"n_estimators":[250]}

clf_cv = GridSearchCV(modxgb_clf, parameters,scoring="f1")
clf_cv.fit(X_train,y_encoded,verbose=True)


#retrain XGBoost on entire training dataset
xgb_final_model = XGBClassifier(**clf_cv.best_params_)
xgb_final_model.fit(X_train,y_train)

#learning Curve
plt.figure()
train_sizes, train_scores, test_scores, fit_times, _ = learning_curve(xgb_final_model, X_train, y_encoded, cv=5,return_times=True,scoring="f1")
plt.plot(train_sizes,np.mean(train_scores,axis=1),label="Training")
plt.plot(train_sizes,np.mean(test_scores,axis=1),label="Validation")
plt.title("Learning Curve")
plt.xlabel("Training Instances")
plt.ylabel("F1-Score")
plt.legend()


y_pred = xgb_final_model.predict(X_test)


fig,ax= plt.subplots(1,3,figsize=(18,7))
#confusion matrix
cm = confusion_matrix(y_test, y_pred, labels=xgb_final_model.classes_)
ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=xgb_final_model.classes_).plot(ax=ax[0])

#PR - Curve
PrecisionRecallDisplay.from_estimator(
    xgb_final_model, X_test, y_test, name="XGBoost").plot(ax=ax[1])


#AUC/ROC
plot_roc_curve(
    xgb_final_model, X_test, y_test, name="XGBoost").plot(ax=ax[2])

fig.suptitle("Confusion Matrix | Precision-Recall Graph | AUC Curve")

print()
print("CV Best F1-Score: {}".format(clf_cv.best_score_))
print("Testing F1-Score: {}".format(f1_score(y_pred,y_test,pos_label="Software Engineer")))



###
#Model Interpretability
###


#feature importance
plt.figure()
plot_importance(xgb_final_model,importance_type="gain")


#PDP
fig,ax = plt.subplots(1,1,figsize=(15,5))
plot_partial_dependence(xgb_final_model,X_train,["EducationYrs","Age","Experience"],feature_names = list(X_train.columns),ax=ax)
plt.suptitle("Partial Dependence Plots (PDP)")


#SHAP
shap_ex = shap.TreeExplainer(xgb_final_model)
vals = shap_ex.shap_values(X_train)
plt.figure(figsize=(10,6))
shap.summary_plot(vals,X_train,alpha=0.5)

