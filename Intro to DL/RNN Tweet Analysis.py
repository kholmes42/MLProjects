# -*- coding: utf-8 -*-
"""
Created on Sun May 14 15:58:45 2023
Coursera Intro DL Boulder Colorado: Week 4 Kaggle Competition on Disaster Tweets
Dataset: https://www.kaggle.com/competitions/nlp-getting-started/overview/faq
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from sklearn.model_selection import train_test_split
import zipfile
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
from tensorflow.keras.layers import Embedding
from sklearn.model_selection import train_test_split
from keras import backend as K

###load datasets
df_train = pd.read_csv(r"train.csv")
df_test = pd.read_csv(r"test.csv")


##EDA
fig,ax = plt.subplots(1,2,figsize=(15,5))
sns.countplot(x=df_train["target"],ax=ax[0])
ax[0].set_title("Training Class Balance")
ax[0].set_xlabel("Target")
ax[0].set_ylabel("Count")


df_train["Tweet Length"] = df_train["text"].apply(lambda x: len(x.split(" ")))
sns.histplot(data=df_train,hue="target",x="Tweet Length",ax=ax[1])

ax[1].set_title("Tweet Length Histogram")
ax[1].set_xlabel("Length of Tweet")
ax[1].set_ylabel("Count")



#load GloVe embedding
z = zipfile.ZipFile(r"C:\Users\kholm\Downloads\glove.6B.zip", "r")

embeddings_index = {}
with z.open("glove.6B.50d.txt") as f:
    for line in f:
        word, coefs = line.split(maxsplit=1)
        coefs = np.fromstring(coefs, "f", sep=" ")
        embeddings_index[word.decode('utf-8')] = coefs

print("Found %s word vectors." % len(embeddings_index))


vectorizer = TextVectorization(max_tokens=20000, output_sequence_length=25)
text_ds = tf.data.Dataset.from_tensor_slices(df_train["text"]).batch(128)
vectorizer.adapt(text_ds)

voc = vectorizer.get_vocabulary()
word_index = dict(zip(voc, range(len(voc))))


num_tokens = len(voc) + 1
embedding_dim = 50
hits = 0
misses = 0

# Prepare embedding matrix
embedding_matrix = np.zeros((num_tokens, embedding_dim))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)

    if embedding_vector is not None:
        # Words not found in embedding index will be all-zeros.
        # This includes the representation for "padding" and "OOV"
        embedding_matrix[i] = embedding_vector
        hits += 1
    else:
        misses += 1
print("Converted %d words (%d misses)" % (hits, misses))


def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

###
#Create Simple RNN
###

#create embedding layer (200 word tweet with each represented by 50 dim emebedding vector)
embedding_layer = Embedding(
    input_dim=num_tokens,
    output_dim=50,
    input_length = 25,
    embeddings_initializer=tf.keras.initializers.Constant(embedding_matrix),
    trainable=False
)

model = models.Sequential()
model.add(embedding_layer) #pretrained embedding GloVe 50 dim
model.add(layers.Dropout(0.2))
model.add(layers.GRU(64,dropout=0.2))
#model.add(layers.Dense(8, activation="relu"))
model.add(layers.Dense(1, activation='sigmoid')) #binary classification problem

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=["AUC",f1_m])

model.summary()



#train test split and convert data into vectorized format
X_train, X_validate, y_train, y_validate = train_test_split(df_train["text"], df_train["target"], test_size=0.15, random_state=42)

x_train = vectorizer(np.array([[s] for s in X_train])).numpy()
x_val = vectorizer(np.array([[s] for s in X_validate])).numpy()

y_train = np.array(y_train)
y_val = np.array(y_validate)



#train model
callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=25)
history = model.fit(x_train, y_train, epochs=500, 
                      validation_data=(x_val, y_val),callbacks=[callback],verbose=True)


fig,ax= plt.subplots(1,3,figsize=(20,5))


# summarize history for accuracy
ax[0].plot(history.history['auc'])
ax[0].plot(history.history['val_auc'])
ax[0].set_title('Model AUC')
ax[0].set_ylabel('AUC')
ax[0].set_xlabel('Epoch')
ax[0].legend(['Train', 'Validation'], loc='upper left')

ax[1].plot(history.history['f1_m'])
ax[1].plot(history.history['val_f1_m'])
ax[1].set_title('Model F1-Score')
ax[1].set_ylabel('F1-Score')
ax[1].set_xlabel('Epoch')
ax[1].legend(['Train', 'Validation'], loc='upper left')


# summarize history for loss
ax[2].plot(history.history['loss'])
ax[2].plot(history.history['val_loss'])
ax[2].set_title('Model Loss (Binary Cross-Entropy)')
ax[2].set_ylabel('Loss')
ax[2].set_xlabel('Epoch')
ax[2].legend(['Train', 'Validation'], loc='upper right')

model.save('Final GRU Tweet Model.keras')


