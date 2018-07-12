import numpy as np
import pandas as pd
import os.path
from imutils import paths
import cv2
import glob

# import matplotlib.pyplot as plt
# # %matplotlib inline

from keras.models import load_model

letter_contours = []

# After the trials done in the notebook we can get the contours of each letters.
x, y, w, h = 30, 12, 20, 38
for i in range(5):
    letter_contours.append((x, y, w, h))
    x += w

array_letter = np.array(['y', '2', '3', '4', '5', '6', '7', '8', 'b', 'c',
                         'd', 'e', 'f', 'g', 'm', 'n', 'p', 'w', 'x'])

model = load_model("labelling.model")

CAPTCHA_IMAGE_FOLDER = "samples"

# List of captchas.
captcha_images = glob.glob(os.path.join(CAPTCHA_IMAGE_FOLDER, "*"))

# Go through each image in the folder.
for (i, captcha_image) in enumerate(captcha_images):
    print("Image {}/{}".format(i + 1, len(captcha_image)))
    # Get the letter labels from the imgae names.
    filename = os.path.basename(captcha_image)
    captcha_correct_text = os.path.splitext(filename)[0]

    # Load image and convert to grey.
    img = cv2.imread(captcha_image, 0)

    # From RGB to BW
    # Adaptive thresholding
    th = cv2.adaptiveThreshold(
        img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 17, 2)

    # Erode and dilate (Because it is black on white, erosion dilates and dilation erodes).
    kernel = np.ones((3, 3), np.uint8)
    dilation = cv2.dilate(th, kernel, iterations=1)
    erosion = cv2.erode(dilation, kernel, iterations=1)
    kernel = np.ones((3, 1), np.uint8)
    dilation = cv2.dilate(erosion, kernel, iterations=1)

    prediction = ''
    for letter_contour, letter_text in zip(letter_contours, captcha_correct_text):
        # Grab the coordinates of the letter in the image.
        x, y, w, h = letter_contour

        # Extract the letter from the original image with a 2-pixel margin around the edge.
        letter_image = dilation[y - 2:y + h + 2, x - 2:x + w + 2]
        # Reshape for model.
        letter_image = np.expand_dims(letter_image, axis=2)
        data = [letter_image]
        data = np.array(data, dtype="float") / 255.0

        # Predict letter.
        decoded_images = model.predict(data)
        # Get letter that the model is most confident about.
        array_pred = np.where(
            decoded_images == np.amax(decoded_images), 1, 0)
        array_pred = np.array(array_pred[0])
        i, = np.where(array_pred == 1)
        prediction = prediction + array_letter[i][0]

    if(prediction is captcha_correct_text):
        print("Correct !")
