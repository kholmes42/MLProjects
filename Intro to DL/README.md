This folder follows along with the homeworks for the Coursera Boulder Colorado Introduction to Deel Learning Course.

## Convolutional Neural Networks (CNN)

In week 3 we look at CNNs. CNNs are an architecture that allow neural networks to work more efficiently with image data. In a typical neural network input must be in a flat vector format which can mean that large high quality images might not be efficient to train neural networks on because it leads to lots of weights in the networks. A CNN adds convolutional layers to the front of a network to learn filters within the image and reduce the dimensionality of the problem.

The goal of this week is to examine the kaggle Histopathologic Cancer Detection dataset which contains ~270,000 images of cells that contain cancer or not. I train a CNN with a binary cross entropy loss function in my attempt and with an AUC metric. Ultimately given the slow training time (which means I end training early), I get very poor results on the test data. However, the goal of this project is to more or less understand the workflow and design parameters for a CNN.

![plot](https://github.com/kholmes42/MLProjects/blob/main/imgs/cnntraining.jpg)
