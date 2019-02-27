# Optical Character Recognition

## [Digit Recognition](https://github.com/Fournierp/OCR/tree/master/Digit%20Recognition)

![alt text](https://github.com/Fournierp/OCR/blob/master/Digit%20Recognition/digits.png)

This notebook is the source code used for the submissions for [Kaggle Competition](https://www.kaggle.com/c/digit-recognizer). The goal of the competition is to classify images from the [MNIST handwritten digit database](http://yann.lecun.com/exdb/mnist/). The solution developed produced 99.928% accuracy and got me to 60th place. (top 3%).

The approach I took was to perform an Exploratory Data Analysis which enabled me to notice that the data was not noisy but that not all pixels in the images were useful. Thus I could do some Dimensionality Reduction. I build a simple Convolutional Neural Network (CNN) using Keras. The steps I followed are (as described in the [Jupyter Notebook](https://github.com/Fournierp/OCR/tree/master/Digit%20Recognition/Digit%20Recognition.ipynb)) to do normalization, reshaping, data augmentation and training with an Adam Optimizer and a ReduceLROnPlateau callback.


## [CAPTCHA](https://github.com/Fournierp/OCR/tree/master/CAPTCHA)

![alt text](https://raw.githubusercontent.com/Fournierp/OCR/master/CAPTCHA/samples/2b827.png?token=AS-TbW2Fft3Z2B4Ak55XnhNl8oYrE1Xgks5bTKQlwA%3D%3D)

This directory is an attempt at recognizing [CAPTCHA](https://en.wikipedia.org/wiki/CAPTCHA) (Completely Automated Public Turing test to tell Computers and Humans Apart) images. Built in 1997 as way for users to identify and block bots (in order to prevent spam, DDOS etc.). They have since then been replace by reCAPTCHA because they are breakable using Artificial Intelligence as we will see.

The approach taken by the CAPTCHA creators to make the task of classifying the images impossible for computers, is to distort the letters. Thus the letters have noise, extra lines crossing the words... To solve this, I built a conventional CNN with a twist. I stacked at the deeper level of the model, 5 branching Convolutional Layers such that each one would be specifically trained to classify a single letter.
