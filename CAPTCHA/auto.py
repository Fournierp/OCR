import numpy as np
import pandas as pd
import os.path
from imutils import paths
import cv2
from helpers import resize_to_fit

from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split

from keras.utils.np_utils import to_categorical
from keras.layers import Activation, Conv2D, Dense, Dropout, Flatten, Input, MaxPooling2D, UpSampling2D
from keras.models import Model
import keras.optimizers
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras import backend as K

from helpers import resize_to_fit

# Pour chaque dossier, choisir une lettre representative et train le AE dessus.
# Use data augmentation pour avoir plus de donnes.
# Ajouter noise et retrain le model.

LETTER_FOLDER = "letters"
data = []
labels = []

for image_file in paths.list_images(LETTER_FOLDER):
    # Load the image
    image = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)
    # # Resize the letter so it fits in a 20x20 pixel box
    image = resize_to_fit(image, 20, 20)

    # Add a third channel dimension to the image to make Keras happy
    image = np.expand_dims(image, axis=2)

    # Get the folder name (ie. the true character value)
    label = image_file.split(os.path.sep)[-2]

    # Add the image and char to the dictionary/
    data.append(image)
    if label == '2':
        labels.append(1)
    elif label == '3':
        labels.append(2)
    elif label == '4':
        labels.append(3)
    elif label == '5':
        labels.append(4)
    elif label == '6':
        labels.append(5)
    elif label == '7':
        labels.append(6)
    elif label == '8':
        labels.append(7)
    elif label == 'b':
        labels.append(8)
    elif label == 'c':
        labels.append(9)
    elif label == 'd':
        labels.append(10)
    elif label == 'e':
        labels.append(11)
    elif label == 'f':
        labels.append(12)
    elif label == 'g':
        labels.append(13)
    elif label == 'm':
        labels.append(14)
    elif label == 'n':
        labels.append(15)
    elif label == 'p':
        labels.append(16)
    elif label == 'w':
        labels.append(17)
    elif label == 'x':
        labels.append(18)
    elif label == 'y':
        labels.append(0)

# Normalization
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)

# Categorization
labels = to_categorical(labels, num_classes = 19)
# print(labels)
# Train-test split
(x_train, x_test, y_train, y_test) = train_test_split(data, labels, test_size=0.25, random_state=0)
print(x_train.shape)
print(x_test.shape)
# x_train = np.reshape(x_train, (len(x_train), 20, 20, 1))
# x_test = np.reshape(x_test, (len(x_test), 20, 20, 1))
x_train = x_train.reshape(-1,20,20,1)
x_test = x_test.reshape(-1,20,20,1)
# # Convolutional autoencoder
# input_img = Input(shape=(20, 20,1))
#
# # Encoder
# x = Conv2D(16, (5, 5), activation='relu', padding='same')(input_img)
# x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)
# x = Conv2D(8, (5, 5), activation='relu', padding='same')(x)
# x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)
# x = Conv2D(8, (5, 5), activation='relu', padding='same')(x)
# encoded = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)
# # encoder = Model(input_img, encoded)
#
# # Decoder
# x = Conv2D(8, (5, 5), activation='relu', padding='same')(encoded)
# x = UpSampling2D((2, 2))(x)
# x = Conv2D(8, (5, 5), activation='relu', padding='same')(x)
# x = UpSampling2D((2, 2))(x)
# x = Conv2D(16, (5, 5), activation='relu', padding='same')(x)
# x = UpSampling2D((2, 2))(x)
# # x = Flatten()(x)
# decoded = Conv2D(1, (5, 5), activation='sigmoid', padding='same')(x)

input_img = Input(shape=(20, 20,1))  # adapt this if using `channels_first` image data format

x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
encoded = MaxPooling2D((2, 2), padding='same')(x)

# at this point the representation is (7, 7, 32)

x = Conv2D(32, (3, 3), activation='relu', padding='same')(encoded)
x = UpSampling2D((2, 2))(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

# optimizer = keras.optimizers.Adam(
#     lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
learning_rate_reduction = ReduceLROnPlateau(
    monitor='val_acc', patience=2, verbose=1, factor=0.4, min_lr=0.000001)
#
# autoencoder = Model(input_img, decoded)
# autoencoder.compile(optimizer=optimizer, loss='binary_crossentropy')

# TERMINAL CMD : tensorboard --logdir=/tmp/autoencoder
from keras.callbacks import TensorBoard

autoencoder.fit(x_train, x_train,
                epochs=1,
                batch_size=128,
                # shuffle=True,
                validation_data=(x_test, x_test),
                callbacks=[TensorBoard(log_dir='/tmp/autoencoder'), learning_rate_reduction])

# out = Dense(19, activation='softmax')(encoder.output)
# newmodel = Model(encoder.input, out)
#
# newmodel.compile(loss='categorical_crossentropy',
#                  optimizer=optimizer,
#                  metrics=['accuracy'])
#
# newmodel.fit(x_train, y_train,
#              epochs=1,
#              batch_size=128,
#              # shuffle=True,
#              validation_data=(x_test, y_test),
#              callbacks=[TensorBoard(log_dir='/tmp/autoencoder'), learning_rate_reduction])

# scores = newmodel.evaluate(x_test, y_test, verbose=1)
# print("Accuracy: ", scores[1])
#
# # Model is trained on encoding and decoding regular mnist, now we train it to ignore noise.
# # Salt and peper noise
# # TODO: figure out if salt and pepper noise applies to captcha noise.
# noise_factor = 0.5
# x_train_noisy = x_train + noise_factor * \
#     np.random.normal(loc=0.0, scale=1.0, size=x_train.shape)
# x_test_noisy = x_test + noise_factor * \
#     np.random.normal(loc=0.0, scale=1.0, size=x_test.shape)
#
# x_train_noisy = np.clip(x_train_noisy, 0., 1.)
# x_test_noisy = np.clip(x_test_noisy, 0., 1.)
#
# autoencoder.fit(x_train_noisy, x_train,
#                 epochs=100,
#                 batch_size=128,
#                 shuffle=True,
#                 validation_data=(x_test_noisy, x_test),
#                 callbacks=[TensorBoard(log_dir='/tmp/tb', histogram_freq=0, write_graph=False)])
