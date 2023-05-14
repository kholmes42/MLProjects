# -*- coding: utf-8 -*-
"""
Created on Fri May 12 20:46:31 2023
This module loads model and prepares output of weeks homework.
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


history = pd.read_csv(r"history.csv")

fig,ax= plt.subplots(1,2,figsize=(15,5))


# summarize history for accuracy
ax[0].plot(history['auc'])
ax[0].plot(history['val_auc'])
ax[0].set_title('Model AUC')
ax[0].set_ylabel('AUC')
ax[0].set_xlabel('Epoch')
ax[0].legend(['Train', 'Validation'], loc='upper left')

# summarize history for loss
ax[1].plot(history['loss'])
ax[1].plot(history['val_loss'])
ax[1].set_title('Model Loss (Binary Cross-Entropy)')
ax[1].set_ylabel('Loss')
ax[1].set_xlabel('Epoch')
ax[1].legend(['Train', 'Validation'], loc='upper right')


###begin testing, load testing data
test_images = np.zeros((57458,96, 96, 3),dtype=np.int8)

dfout = pd.DataFrame(index=range(0,57458),columns=['id','label'])

z = zipfile.ZipFile(r"C:\Users\kholm\Downloads\histopathologic-cancer-detection.zip", "r")
i = 0
for filename in z.namelist():
    if "test" in filename:
       print (filename)
       test_images[i,::] = np.asarray(Image.open(z.open(filename)))
       dfout.iloc[i,0] = filename[5:-4]
       i+=1
       
print(i)

#load pretrained model
final_model = models.load_model('Final CNN Cancer Model.keras',compile=False)
final_model.compile()

#predict on test data
yhat = final_model.predict(test_images)

dfout["label"] = yhat
dfout.to_csv("outputsubmission.csv",index=False)
