import os
import os.path
import glob
import cv2
import numpy as np

letter_contours = []
counts = {}

#After the trials done in the notebook we can get the contours of each letters.
x, y, w, h = 30, 12, 20, 38
for  i in range(5):
    letter_contours.append((x, y, w, h))
    x += w

# Image folders.
CAPTCHA_IMAGE_FOLDER = "samples"
OUTPUT_FOLDER = "letters"

# List of captchas.
captcha_images = glob.glob(os.path.join(CAPTCHA_IMAGE_FOLDER, "*"))

# Go through each image in the folder.
for (i, captcha_image) in enumerate(captcha_images):
    print("Image {}/{}".format(i + 1, len(captcha_image)))

    # Get the letter labels from the imgae names.
    filename = os.path.basename(captcha_image)
    captcha_correct_text = os.path.splitext(filename)[0]

    #Load image and convert to grey.
    img = cv2.imread(captcha_image, 0)

    # From RGB to BW
    # Adaptive thresholding
    th = cv2.adaptiveThreshold(
        img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 17, 2)

    #Erode and dilate (Because it is black on white, erosion dilates and dilation erodes).
    kernel = np.ones((3,3), np.uint8)
    dilation = cv2.dilate(th, kernel, iterations=1)

    erosion = cv2.erode(dilation, kernel, iterations=1)

    kernel = np.ones((3,1), np.uint8)
    dilation = cv2.dilate(erosion, kernel, iterations=1)

    for letter_contour, letter_text in zip(letter_contours, captcha_correct_text):
        # Grab the coordinates of the letter in the image.
        x, y, w, h = letter_contour

        # Extract the letter from the original image with a 2-pixel margin around the edge.
        letter_image = dilation[y - 2:y + h + 2, x - 2:x + w + 2]

        # Get the folder to save the image in
        save_path = os.path.join(OUTPUT_FOLDER, letter_text)

        # If the output directory does not exist, create it
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        # Write the letter image to a file
        count = counts.get(letter_text, 1)
        p = os.path.join(save_path, "{}.png".format(str(count).zfill(6)))
        cv2.imwrite(p, letter_image)

        # increment the count for the current key
        counts[letter_text] = count + 1
