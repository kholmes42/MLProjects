# -*- coding: utf-8 -*-
"""
Created on Tue Sep 27 12:46:55 2022
Sentiment Analysis with Logistic Regression
@author: kholm
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score,confusion_matrix,accuracy_score

from wordcloud import WordCloud


df = pd.read_csv(r"AirlineTweets.csv")
df = df[(df["airline_sentiment"] == "positive") | (df["airline_sentiment"] == "negative" )]
print(df.head())

import pandas as pd


#check class balance
print(df["airline_sentiment"].value_counts())
plt.figure()
sns.countplot(df["airline_sentiment"])
plt.title("Class Balance")

#look at by airline
sentbyairlines = df[["airline_sentiment","airline"]]
plt.figure()
sns.countplot(data=sentbyairlines,x="airline",hue="airline_sentiment")




#we have imbalanced classes so split train and test while maintaining class ratio
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
sss.get_n_splits(df["text"], df["airline_sentiment"])


#vectorize text data
tfidfvectorizer = CountVectorizer(max_features=2000)
for train_index, test_index in sss.split(df["text"], df["airline_sentiment"]):
    print("TRAIN:", train_index, "TEST:", test_index)
    traindata_x = tfidfvectorizer.fit_transform(df["text"].iloc[train_index])
    train_y = df["airline_sentiment"].iloc[train_index]
    testdata_x = tfidfvectorizer.transform(df["text"].iloc[test_index])
    test_y = df["airline_sentiment"].iloc[test_index]
    
    
#use classifier
clf = LogisticRegression(max_iter=500)
clf.fit(traindata_x,train_y)


y_pred = clf.predict(testdata_x)


print(test_y.value_counts())


print()
print("Multi Class")
print("----------------------")
print("Accuracy: {}".format(accuracy_score(test_y, y_pred)))
f1 = f1_score(test_y, y_pred,average="weighted")
print("F1 SCORE: {}".format(f1))
print("Confusion Matrix")
print(confusion_matrix(test_y, y_pred))



#visualize words
plt.figure()
alltxt = ""
for txt in df[df["airline_sentiment"] == "positive"]["text"]:
    alltxt += txt.lower() + " "
wrdcld  = WordCloud().generate(alltxt)

plt.imshow(wrdcld)
plt.axis("off")
plt.show()

#analyze values
plt.figure()
sns.histplot(clf.coef_[0],bins=30)

coefvals = clf.coef_[0]
vocab = tfidfvectorizer.vocabulary_


ind = np.argsort(coefvals)[:15]

print(np.array(list(vocab.keys()))[ind])


