# -*- coding: utf-8 -*-
"""
Created on Sat May 27 12:28:04 2023
Coursera Intro DL Boulder Colorado: Week 5 GANs with Fashion Mnist

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


(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

training = False

#preprocess
x_train = x_train /255.0
x_test = x_test/255.0


#show 1 positive/negative sample
sample1 = x_train[0,::].reshape((28,28))
sample2 = x_train[1,::].reshape((28,28))
sample3 = x_train[2,::].reshape((28,28))
sample4 = x_train[3,::].reshape((28,28))

fig,ax = plt.subplots(2,2,figsize=(6,6))

ax[0][0].imshow(Image.fromarray(sample1*255))
ax[0][1].imshow(Image.fromarray(sample2*255))
ax[1][0].imshow(Image.fromarray(sample3*255))
ax[1][1].imshow(Image.fromarray(sample4*255))
ax[0][0].set_title("Shoe Sample")
ax[0][1].set_title("Shirt Sample")
ax[1][0].set_title("Blouse Sample")
ax[1][1].set_title("Skirt Sample")
ax[0][0].set_xticks([])
ax[0][1].set_xticks([])
ax[1][0].set_xticks([])
ax[1][1].set_xticks([])
ax[0][0].set_yticks([])
ax[0][1].set_yticks([])
ax[1][0].set_yticks([])
ax[1][1].set_yticks([])


#create batches of images
bat_size = 32
dataset = tf.data.Dataset.from_tensor_slices(x_train).shuffle(1000)
dataset = dataset.batch(bat_size,drop_remainder=True).prefetch(1)



###
#build Generator model
###

rand_input_vect_size = 100

gen_model = models.Sequential([layers.Dense(7*7*128,input_shape=[rand_input_vect_size]),
                               layers.Reshape([7,7,128])
                    ,layers.BatchNormalization()
                    ,layers.Conv2DTranspose(64, kernel_size=(5, 5),strides=(2,2),padding="same", activation='selu')
                    ,layers.BatchNormalization()
                    ,layers.Conv2DTranspose(1,kernel_size=(5, 5),strides=(2,2),padding="same", activation='tanh')
                    ])

noise = tf.random.normal(shape=[1,rand_input_vect_size])
gen_image = gen_model(noise,training=False)

plt.figure()
plt.imshow(gen_image[0],cmap="gray")


###
#build Discriminator model
###

disc_model = models.Sequential([
   layers.Conv2D(64, (5, 5),strides=2, activation=layers.LeakyReLU(0.3),padding="same", input_shape=(possample.shape[0], possample.shape[1], 1)),
   layers.Dropout(0.3),
   layers.Conv2D(128, (5, 5),strides=2, activation=layers.LeakyReLU(0.3),padding="same", input_shape=(possample.shape[0], possample.shape[1], 1)),
   layers.Dropout(0.3),
   layers.Flatten(),
   layers.Dense(1,activation="sigmoid")
])


decision = disc_model(gen_image)


disc_model.compile(loss="binary_crossentropy",optimizer="rmsprop")
disc_model.trainable=False

###
#build GAN model
###

gan = models.Sequential([gen_model,disc_model])
gan.compile(loss="binary_crossentropy",optimizer="rmsprop")



def train_gan(model,data,bsize=32,rand_feat=100,epochs=100):
    """
    create training loop that iterates updating GAN
    """
    gen,disc = model.layers
    
    # for each epoch
    for i in range(0,epochs):
        print("Epoch {}".format(i))
        
        #for all batches
        for bat in data:
            #create random vectors
            noise = tf.random.normal(shape=[bsize,rand_feat])
            gen_images = gen(noise)
            bat = tf.reshape(tf.cast(bat, tf.float32),[bsize,28,28,1])
            
            X_FR = tf.concat([gen_images,bat],axis=0)
            ys = tf.constant([[0.0]]*bsize + [[1.0]]*bsize)
            
            
            #train the discrimator (sees both fake and real images)
            disc.trainable=True
            disc.train_on_batch(X_FR,ys)
            
            y2 = tf.constant([[1.0]]*bsize)
            disc.trainable=False
             
            #train the generator
            model.train_on_batch(noise,y2)
            
    return model



if training == True:
    trained_gan = train_gan(gan,dataset,bsize=32)
    
    trained_gan.save('Final GAN Fashion mnist.keras')

else:

    #load pretrained model
    trained_gan = models.load_model('Final GAN Fashion mnist.keras')
    
    for i in range(1,10):
        noise = tf.random.normal(shape=[1,rand_input_vect_size])
        final_generated_image = trained_gan.layers[0](noise,training=False)*255
        
        plt.figure()
        plt.imshow(final_generated_image[0],cmap="gray")
        plt.title("Generated Image {}".format(i))
        plt.xticks([])
        plt.yticks([])

