import numpy as np
from keras.datasets import mnist
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras import backend as K

# Convolutional autoencoder
input_img = Input(shape=(28, 28, 1))

# Encoder
# Might want to use same kernel size for all.
x = Conv2D(64, (5, 5), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
encoded = MaxPooling2D((2, 2), padding='same')(x)
encoder = Model(input_img, encoded)

# at this point the representation is (4, 4, 8) i.e. 128-dimensional
# Decoder
x = Conv2D(32, (3, 3), activation='relu', padding='same')(encoded)
x = UpSampling2D((2, 2))(x)
x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(64, (5, 5), activation='relu')(x)
x = UpSampling2D((2, 2))(x)
decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

optimizer = keras.optimizers.Adam(
    lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
learning_rate_reduction = ReduceLROnPlateau(
    monitor='val_acc', patience=2, verbose=1, factor=0.4, min_lr=0.000001)

autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer=optimizer, loss='binary_crossentropy')

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.

x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))
x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))

# TERMINAL CMD : tensorboard --logdir=/tmp/autoencoder
from keras.callbacks import TensorBoard

autoencoder.fit(x_train, x_train,
                epochs=10,
                batch_size=128,
                # shuffle=True,
                validation_data=(x_test, x_test),
                callbacks=[TensorBoard(log_dir='/tmp/autoencoder'), learning_rate_reduction])

out = Dense(10, activation='softmax')(encoder.output)
newmodel = Model(encoder.input, out)

newmodel.compile(loss='categorical_crossentropy',
                 optimizer=optimizer,
                 metrics=['accuracy'])

newmodel.fit(x_train, y_train,
             epochs=10,
             batch_size=128,
             # shuffle=True,
             validation_data=(x_test, y_test),
             callbacks=[TensorBoard(log_dir='/tmp/autoencoder'), learning_rate_reduction])

scores = newmodel.evaluate(x_test, y_test, verbose=1)
print("Accuracy: ", scores[1])

# Model is trained on encoding and decoding regular mnist, now we train it to ignore noise.
# Salt and peper noise
# TODO: figure out if salt and pepper noise applies to captcha noise.
noise_factor = 0.5
x_train_noisy = x_train + noise_factor * \
    np.random.normal(loc=0.0, scale=1.0, size=x_train.shape)
x_test_noisy = x_test + noise_factor * \
    np.random.normal(loc=0.0, scale=1.0, size=x_test.shape)

x_train_noisy = np.clip(x_train_noisy, 0., 1.)
x_test_noisy = np.clip(x_test_noisy, 0., 1.)

autoencoder.fit(x_train_noisy, x_train,
                epochs=100,
                batch_size=128,
                shuffle=True,
                validation_data=(x_test_noisy, x_test),
                callbacks=[TensorBoard(log_dir='/tmp/tb', histogram_freq=0, write_graph=False)])
