"""
Some basic utilities for all detectors/classifiers.

:author:
    Qazi T. Ashikin

"""

import cv2 as cv
import numpy as np


def crop_image(image: np.ndarray, x, y, a, b) -> np.ndarray:
    """
    Crop an image to a specific region.

    The cropped image will be the area enclosed by the points:

    (x, y - b)                     (x + a, y - b)
             .<-------- a -------->.
             ^
             |
             b
             |
             v
             .                     .
        (x, y)                     (x + a, y)

    Parameters
    ----------
    x:
        The horizontal pixel coordinate of the bottom-left point in cropped image.
    y:
        The vertical pixel coordinate of the bottom-left point in cropped image.
    a:
        The width (number of columns) of the cropped region.
    b:
        The height (number of rows) of the cropped region.

    Returns
    -------
     :
        The cropped image.

    """

    height, width, *_ = image.shape

    top_left_x = max(0, x)
    top_left_y = max(0, y - b)
    bottom_right_x = min(width, x + a)
    bottom_right_y = min(height, y)

    cropped_img = image[top_left_y:bottom_right_y, top_left_x:bottom_right_x]

    return cropped_img


def make_greyscale(image: np.ndarray) -> np.ndarray:
    """
    Convert an RGB image to greyscale.

    Parameters
    ----------
    image:
        The image to be converted to greyscale.

    Returns
    -------
     :
        The image in greyscale.

    """

    if len(image.shape) == 3:
        greyscale_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    else:
        greyscale_image = image

    return greyscale_image


def rescale(image: np.ndarray) -> np.ndarray:
    """
    Rescale an image to span the full range of int values.

    Parameters
    ----------
    image:
        The image to be rescaled.

    Returns
    -------
     :
        The rescaled image.

    """

    return cv.normalize(image, None, 0, 255, cv.NORM_MINMAX).astype("uint8")
