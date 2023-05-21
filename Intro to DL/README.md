This folder follows along with the homeworks for the Coursera Boulder Colorado Introduction to Deep Learning Course.

## Convolutional Neural Networks (CNN)

In week 3 we look at CNNs. CNNs are an architecture that allow neural networks to work more efficiently with image data. In a typical neural network input must be in a flat vector format which can mean that large high quality images might not be efficient to train neural networks on because it leads to lots of weights in the networks. A CNN adds convolutional layers to the front of a network to learn filters within the image and reduce the dimensionality of the problem.

The goal of this week is to examine the Kaggle Histopathologic Cancer Detection dataset which contains ~270,000 images of cells that contain cancer or not. I train a CNN with a binary cross entropy loss function in my attempt and with an AUC metric. Ultimately given the slow training time (which means I end training early), I get very poor results on the test data. However, the goal of this project is to more or less understand the workflow and design parameters for a CNN.

![plot](https://github.com/kholmes42/MLProjects/blob/main/imgs/cnntraining.jpg)


## Recurrent Neural Networks (RNN)

In week 4 we look at RNNs. RNNs are an architecture that allow neural networks to work more efficiently with sequential data, and allow recurrent connections through the network instead of strictly enforcing a feedforward sequence.

The goal of this week is to examine the Kaggle Natural Language Processing with Disaster Tweets dataset which contains ~7,000 tweets that refer to real disasters or not. I train a RNN with a binary cross entropy loss function in my attempt and with a F1-Score metric. I use a pre-trained GloVe 50-demension embedding for the encoding of the textual tweet data, before feeding it into the RNN. I additionally add dropout regularization throughout the network to avoid overfitting since we are dealing with a relatively small corpus.

![plot](https://github.com/kholmes42/MLProjects/blob/main/imgs/rnntrain.png)
