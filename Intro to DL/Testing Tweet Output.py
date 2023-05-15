# -*- coding: utf-8 -*-
"""
Created on Fri May 12 20:46:31 2023
This module loads a RNN model and creates the prediction testing
@author: kholm
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from sklearn.model_selection import train_test_split
from keras_preprocessing.image import ImageDataGenerator
import zipfile
from PIL import Image



#load test data and vectorize
df_testing = pd.read_csv(r"test.csv")


vectorizer = TextVectorization(max_tokens=20000, output_sequence_length=25)
text_ds = tf.data.Dataset.from_tensor_slices(df_train["text"]).batch(128)
vectorizer.adapt(text_ds)

voc = vectorizer.get_vocabulary()
word_index = dict(zip(voc, range(len(voc))))


x_test = vectorizer(np.array([[s] for s in df_testing["text"]])).numpy()




#load pretrained model
final_model = models.load_model('Final RNN Tweet Model.keras',compile=False)
final_model.compile()

#predict on test data
yhat = (final_model.predict(x_test) > 0.5).astype("int32")

df_testing["target"] = yhat
df_testing[["id","target"]].to_csv("outputsubmission.csv",index=False)



