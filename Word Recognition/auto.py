import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('cap.png',0)

img = cv2.medianBlur(img,5)
th2 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)

kernel = np.ones((2,2), np.uint8)
erosion = cv2.erode(th2, kernel, iterations = 1)
dilation = cv2.dilate(th2, kernel, iterations = 1)
kernel = np.ones((2,2), np.uint8)
# dilation2 = cv2.dilate(dilation, kernel, iterations = 10)

im2,contours,hierarchy = cv2.findContours(dilation, 1, 2)

cnt = contours[0]
M = cv2.moments(cnt)
# print( M )
rect = cv2.minAreaRect(cnt)
box = cv2.boxPoints(rect)
box = np.int0(box)
cv2.drawContours(img,[box],0,(0,0,255),2)

titles = ['Original Image', 'Adaptive Mean Thresholding', 'erosion', 'dilation']
images = [img, th2, im2, dilation]


# cv2.imshow('res2', approx)

for i in range(4):
    plt.subplot(2,2,i+1),plt.imshow(images[i],'gray')
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])

plt.show()

# import numpy as np
# import cv2
# import matplotlib as plt
#
# im = cv2.imread('cap.png')
# imgray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
# ret,thresh = cv2.threshold(imgray,127,255,0)
# image, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
#
# # img = cv2.drawContours(im, contours, -1, (0,255,0), 3)
# # print(contours)
# cv2.namedWindow("im", cv2.WINDOW_AUTOSIZE)        # create windows, use WINDOW_AUTOSIZE for a fixed window size
# cv2.namedWindow("img", cv2.WINDOW_AUTOSIZE)           # or use WINDOW_NORMAL to allow window resizing
#
# cv2.imshow("im", im)         # show windows
# cv2.imshow("img", image)
#
# cv2.waitKey()                               # hold windows open until user presses a key
# cv2.destroyAllWindows()                     # remove windows from memory

# CannyStill.py

# import cv2
# import numpy as np
# import os
#
# ###################################################################################################
# def main():
#     imgOriginal = cv2.imread("cap.png")               # open image
#
#     if imgOriginal is None:                             # if image was not read successfully
#         print("error: image not read from file \n\n")        # print error message to std out
#         os.system("pause")                                  # pause so user can see error message
#         return                                              # and exit function (which exits program)
#
#     imgGrayscale = cv2.cvtColor(imgOriginal, cv2.COLOR_BGR2GRAY)        # convert to grayscale
#
#     imgBlurred = cv2.GaussianBlur(imgGrayscale, (5, 5), 0)              # blur
#
#     imgCanny = cv2.Canny(imgBlurred, 100, 200)                          # get Canny edges
#
#     cv2.namedWindow("imgOriginal", cv2.WINDOW_AUTOSIZE)        # create windows, use WINDOW_AUTOSIZE for a fixed window size
#     cv2.namedWindow("imgCanny", cv2.WINDOW_AUTOSIZE)           # or use WINDOW_NORMAL to allow window resizing
#
#     cv2.imshow("imgOriginal", imgOriginal)         # show windows
#     cv2.imshow("imgCanny", imgCanny)
#
#     cv2.waitKey()                               # hold windows open until user presses a key
#
#     cv2.destroyAllWindows()                     # remove windows from memory
#
#     return
#
# ###################################################################################################
# if __name__ == "__main__":
#     main()

# import numpy as np
# from keras.datasets import mnist
# from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
# from keras.models import Model
# from keras import backend as K
#
# # Convolutional autoencoder
# input_img = Input(shape=(28, 28, 1))
#
# # Encoder
# # Might want to use same kernel size for all.
# x = Conv2D(64, (5, 5), activation='relu', padding='same')(input_img)
# x = MaxPooling2D((2, 2), padding='same')(x)
# x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
# x = MaxPooling2D((2, 2), padding='same')(x)
# x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
# encoded = MaxPooling2D((2, 2), padding='same')(x)
# encoder = Model(input_img, encoded)
#
# # at this point the representation is (4, 4, 8) i.e. 128-dimensional
# # Decoder
# x = Conv2D(32, (3, 3), activation='relu', padding='same')(encoded)
# x = UpSampling2D((2, 2))(x)
# x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
# x = UpSampling2D((2, 2))(x)
# x = Conv2D(64, (5, 5), activation='relu')(x)
# x = UpSampling2D((2, 2))(x)
# decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)
#
# optimizer = keras.optimizers.Adam(
#     lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
# learning_rate_reduction = ReduceLROnPlateau(
#     monitor='val_acc', patience=2, verbose=1, factor=0.4, min_lr=0.000001)
#
# autoencoder = Model(input_img, decoded)
# autoencoder.compile(optimizer=optimizer, loss='binary_crossentropy')
#
# (x_train, y_train), (x_test, y_test) = mnist.load_data()
#
# x_train = x_train.astype('float32') / 255.
# x_test = x_test.astype('float32') / 255.
#
# x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))
# x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))
#
# # TERMINAL CMD : tensorboard --logdir=/tmp/autoencoder
# from keras.callbacks import TensorBoard
#
# autoencoder.fit(x_train, x_train,
#                 epochs=10,
#                 batch_size=128,
#                 # shuffle=True,
#                 validation_data=(x_test, x_test),
#                 callbacks=[TensorBoard(log_dir='/tmp/autoencoder'), learning_rate_reduction])
#
# out = Dense(10, activation='softmax')(encoder.output)
# newmodel = Model(encoder.input, out)
#
# newmodel.compile(loss='categorical_crossentropy',
#                  optimizer=optimizer,
#                  metrics=['accuracy'])
#
# newmodel.fit(x_train, y_train,
#              epochs=10,
#              batch_size=128,
#              # shuffle=True,
#              validation_data=(x_test, y_test),
#              callbacks=[TensorBoard(log_dir='/tmp/autoencoder'), learning_rate_reduction])
#
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
