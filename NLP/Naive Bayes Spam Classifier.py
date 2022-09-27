# -*- coding: utf-8 -*-
"""
Created on Tue Sep 27 10:38:23 2022
Naive Bayes Spam Detection
@author: kholm
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import f1_score
from sklearn import metrics
from wordcloud import WordCloud

df = pd.read_csv(r"spam.csv",usecols=[0,1])

#check class balance
print(df["v1"].value_counts())
plt.figure()
sns.countplot(df["v1"])
plt.title("Class Balance")

#we have imbalanced classes so split train and test while maintaining class ratio
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
sss.get_n_splits(df["v2"], df["v1"])


#vectorize text data
tfidfvectorizer = TfidfVectorizer()
for train_index, test_index in sss.split(df["v2"], df["v1"]):
    print("TRAIN:", train_index, "TEST:", test_index)
    traindata_x = tfidfvectorizer.fit_transform(df["v2"].iloc[train_index])
    train_y = df["v1"].iloc[train_index]
    testdata_x = tfidfvectorizer.transform(df["v2"].iloc[test_index])
    test_y = df["v1"].iloc[test_index]
    

#use classifier
clf = MultinomialNB()
clf.fit(traindata_x,train_y)

y_pred = clf.predict(testdata_x)
f1 = f1_score(test_y, y_pred,pos_label="ham")
print("F1 SCORE: {}".format(f1))
metrics.plot_roc_curve(clf, testdata_x, test_y) 



#visualize spam
plt.figure()
alltxt = ""
for txt in df[df["v1"] == "spam"]["v2"]:
    alltxt += txt.lower() + " "
wrdcld  = WordCloud().generate(alltxt)

plt.imshow(wrdcld)
plt.axis("off")
plt.show()

plt.figure()
