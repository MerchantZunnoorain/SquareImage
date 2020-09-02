import sys
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.filters import gaussian
from skimage.transform import rescale, resize

def square_image(image_path, background_sigma=5):
    img = imread(image_path) / 255.
    h, w, _ = img.shape

    max_dimen = max(w, h)

    scale_factor = max_dimen / min(w, h)
    blur_img = gaussian(img, sigma=background_sigma, multichannel=True)
    blur_img = rescale(blur_img, scale_factor, multichannel=True)
    blur_img = blur_img[:max_dimen, :max_dimen]

    y_offset = int((max_dimen - h) / 2.)
    x_offset = int((max_dimen - w) / 2.)
    blur_img[y_offset: y_offset+h, x_offset:x_offset+w, :] = img[:, :, :]

    plt.imshow(blur_img)
    plt.show()

image_path = sys.argv[1]
square_image(image_path)

