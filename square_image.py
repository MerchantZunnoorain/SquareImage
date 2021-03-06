import sys
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.filters import gaussian
from skimage.transform import rescale, resize

def square_image(image_path, background_sigma=5):
    """ Convert any image into square image """
    img = imread(image_path) / 255.
    h, w, _ = img.shape

    # calculate dimension of final image
    max_dimen = max(w, h)

    # generate background image
    scale_factor = max_dimen / min(w, h)
    blur_img = gaussian(img, sigma=background_sigma, multichannel=True)
    blur_img = rescale(blur_img, scale_factor, multichannel=True)
    blur_img = blur_img[:max_dimen, :max_dimen]

    # paste original image on top of this
    y_offset = int((max_dimen - h) / 2.)
    x_offset = int((max_dimen - w) / 2.)
    blur_img[y_offset: y_offset+h, x_offset:x_offset+w, :] = img[:, :, :]

    # plot the image
    fig, axes = plt.subplots(nrows=1, ncols=2, sharex=True, sharey=True, figsize=(10, 10))
    ax = axes.ravel()

    ax[0].imshow(img, cmap=plt.cm.gray)
    ax[0].set_title('Original Image')

    ax[1].imshow(blur_img, cmap=plt.cm.gray)
    ax[1].set_title('Squared Image')

    fig.tight_layout()
    plt.show()

# Make sure code is executed directly from the shell
if __name__ == '__main__':

    # check number of argument passed to the program
    if len(sys.argv) < 2:
        # if less than 2, then it is incorrect.
        # print help and exit
        print('Usage:\n\tpython %s <image_path>' % sys.argv[0])
        exit(0)

    # get image path from program arguments
    image_path = sys.argv[1]

    # call square image function
    square_image(image_path)

