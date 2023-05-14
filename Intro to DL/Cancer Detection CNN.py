# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 20:29:15 2023
Coursera Intro DL Boulder Colorado: Week 3 Kaggle Competition on Image Binary Tumor Identification
https://www.kaggle.com/competitions/histopathologic-cancer-detection/data?select=train_labels.csv
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image
import zipfile
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from sklearn.model_selection import train_test_split
from keras_preprocessing.image import ImageDataGenerator


###
#downloading dataset
###

z = zipfile.ZipFile(r"C:\Users\kholm\Downloads\histopathologic-cancer-detection.zip", "r")
i = 1
# for filename in z.namelist():
#     print (filename)
#     bytes = z.read(filename)
#     i +=  1
#     print(i)
    
    
     
df = pd.read_csv(z.open('train_labels.csv'))


###
#EDA 
###

plt.figure()
sns.countplot(x=df["label"])
plt.ylabel("Count")
plt.xlabel("Label")
plt.title("Training Data Breakdown")

#show 1 positive/negative sample
possample = df[df["label"] == 1].iloc[0]["id"]
negsample = df[df["label"] == 0].iloc[0]["id"]

image = Image.open(z.open(r"train/" + possample + ".tif"))
print("Image size is: {} pixels (x3 for RGB)".format(image.size))



fig,ax = plt.subplots(1,2,figsize=(12,6))

ax[0].imshow(Image.open(z.open(r"train/" + possample + ".tif")))
ax[1].imshow(Image.open(z.open(r"train/" + negsample + ".tif")))
ax[0].set_title("Positive Sample")
ax[1].set_title("Negative Sample")

X_train, X_test, train_labels, test_labels = train_test_split(df["id"], df["label"], test_size=0.25, random_state=42)


# plt.figure()
# plt.imshow((Image.open(r"C:/Users/kholm/Downloads/histopathologic-cancer-detection.zip/train/" + possample + ".tif")))
#create validation and training dataset
subsample = True
if subsample == True:

    ind = np.random.choice(np.arange(X_train.shape[0]),100,replace=False)
    train_images = np.zeros((len(ind),image.size[0], image.size[1], 3))
    test_images = np.zeros((len(ind),image.size[0], image.size[1], 3))
    
    #load training images (SUBSAMPLE FOR TESTING STRUCTURE)
    i2 = 0
    for i in ind:
        train_images[i2,::] = np.asarray(Image.open(z.open(r"train/" + df.iloc[i]["id"] + ".tif")))
        test_images[i2,::] = np.asarray(Image.open(z.open(r"train/" + df.iloc[i+10]["id"] + ".tif")))
        i2+=1
    
    train_labels = df.iloc[ind]["label"]
    test_labels = df.iloc[ind+10]["label"]

    
else:
 
    df["label"] = df['label'].astype(str)
    df["id"] = df["id"] + ".tif"
    datagen=ImageDataGenerator(rescale=1./255,validation_split = 0.2)
    
    train_generator=datagen.flow_from_dataframe(dataframe=df, directory="C:/Users/kholm/Downloads/testing/", x_col="id", y_col="label", 
                                                class_mode="binary", target_size=(image.size[0],image.size[1]), batch_size=32,subset="training")
    valid_generator=datagen.flow_from_dataframe(dataframe=df, directory="C:/Users/kholm/Downloads/testing/", x_col="id", y_col="label",
                                                class_mode="binary", target_size=(image.size[0],image.size[1]), batch_size=32,subset="validation")
    


###
#build CNN model
###

model = models.Sequential()
#convolutional section
model.add(layers.Conv2D(8, (3, 3),strides=2, activation='relu', input_shape=(image.size[0], image.size[1], 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(32, (3, 3),strides=2, activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3),strides=2, activation='relu'))
#classification/NN section
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid')) #binary classification problem




model.summary()

#compile model
model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['AUC','binary_accuracy'])


#train model
callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)
# history = model.fit(train_images, train_labels, epochs=100, 
#                     validation_data=(test_images, test_labels),verbose=True)
history = model.fit_generator(generator=train_generator, epochs=100, 
                    validation_data=valid_generator,callbacks=[callback],verbose=True)




model.save('/content/Final CNN Cancer Model2.keras')


hist_df = pd.DataFrame(history.history) 


# or save to csv: 
hist_csv_file = '/content/history2.csv'
with open(hist_csv_file, mode='w') as f:
    hist_df.to_csv(f)

