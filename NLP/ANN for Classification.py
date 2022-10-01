# -*- coding: utf-8 -*-
"""
Created on Sat Oct  1 10:07:14 2022
Using a simple ANN to do text classification on TF-IDF matrix of articles
@author: kholm
"""

import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix


#download data
df = pd.read_csv(r"bbc_text_cls.csv")
#encode response variable for keras ann format
df["labnumeric"] = df["labels"].astype('category').cat.codes

#examine if balanced classes
plt.figure()
df["labels"].hist()
plt.title("Class Distribution")


X_train, X_test, y_train, y_test = train_test_split(df["text"], df[["labels","labnumeric"]], test_size=0.2, random_state=42)



tfidf = TfidfVectorizer(stop_words=("english"))


#create training data
X_train = tfidf.fit_transform(X_train)

#create testing data
X_test = tfidf.transform(X_test)



#build model
ann = tf.keras.models.Sequential()
# Add the input layer and first hidden layer
ann.add(tf.keras.layers.Dense(units=300, activation='relu', input_shape=(X_train[0].shape[1],)))
# Add the output layer
ann.add(tf.keras.layers.Dense(units=len(df["labnumeric"].unique()), activation='softmax'))


#compile model
ann.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])

#train model, rquire array format for tf
results = ann.fit(X_train.toarray(), y_train["labnumeric"], validation_data=(X_test.toarray(), y_test["labnumeric"]),batch_size = 128, epochs = 10)

test_results = ann.evaluate(X_test.toarray(), y_test["labnumeric"], verbose=2)


#graph restults
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
sns.lineplot(x=np.arange(1,11),y=results.history["loss"], ax=ax)
sns.lineplot(x=np.arange(1,11),y=results.history["val_loss"], ax=ax)
plt.title("Cross Entropy")
plt.xlabel("Epoch")
plt.ylabel("Loss")
ax.legend(['Training', 'Testing'])

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
sns.lineplot(x=np.arange(1,11),y=results.history["accuracy"], ax=ax)
sns.lineplot(x=np.arange(1,11),y=results.history["val_accuracy"], ax=ax)
plt.title("Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Loss")
ax.legend(['Training', 'Testing'])

# get confusion matrix
pred_vals = ann.predict(X_test.toarray())
print()
print("Testing Confusion Matrix")
print(confusion_matrix(y_test["labnumeric"], np.argmax(pred_vals,axis=1)))
