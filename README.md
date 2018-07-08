# OCR

## [Digit Recognition](https://github.com/Fournierp/OCR/tree/master/Digit%20Recognition)

![alt text](https://upload.wikimedia.org/wikipedia/commons/thumb/2/27/MnistExamples.png/220px-MnistExamples.png)

This notebook is an attempt at solving the [MNIST handwritten digit database](http://yann.lecun.com/exdb/mnist/). I built a Convolutional Neural Network (CNN) using Keras on top of a Tensorflow backend. The steps I followed are (as described in the [Jupyter Notebook](https://github.com/Fournierp/OCR/tree/master/Digit%20Recognition/Digit%20Recognition.ipynb)) to do normalization, reshaping, data augmentation and training with an Adam Optimizer and a ReduceLROnPlateau callback. I participated to a competition on a Kaggle and achieved 99.614% accuracy and got 262 position (top 10%) (see [here](https://www.kaggle.com/c/digit-recognizer)).

## [CAPTCHA](https://github.com/Fournierp/OCR/tree/master/CAPTCHA)

![alt text](https://github.com/Fournierp/OCR/tree/master/CAPTCHA/samples/2b827.png)

This directory is an attempt at recognizing [CAPTCHA](https://en.wikipedia.org/wiki/CAPTCHA) (Completely Automated Public Turing test to tell Computers and Humans Apart) images. Built in 1997 as way for users to identify and block bots (in order to prevent spam, DDOS etc.). They have since then been replace by reCAPTCHA because they are breakable using Artificial Intelligence as we will see. The steps I followed are to remove noise and separate the characters in the images.
