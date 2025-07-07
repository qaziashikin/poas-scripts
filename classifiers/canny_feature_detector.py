"""
Detection of features based on whether the sum of edges detected using the Canny edge
detection algorithm exceeds some specified threshold.

:author:
    Qazi T. Ashikin

"""

import cv2 as cv
import numpy as np

from .utils import crop_image, make_greyscale, rescale


def detect_feature(image, config):
    """
    Determine if a feature is visible in an image based on whether the sum of the Canny
    edges exceeds some specified threshold.

    This detection method applies the following processing steps, some of which are
    optional.

    - Image crop
    - Make greyscale and rescale to min/max range
    - Apply Gaussian blur
    - Apply CLAHE to enhance contrast
    - Denoising
    - Detect Canny edges

    Parameters
    ----------
    image:
        The image in which the feature is to detected.
    config:
        Contains parameters for each stage of the processing.

    Returns
    -------
     :
        Whether or not the feature has been detected.

    """

    image = crop_image(image, *config["crop_to"])

    if config["greyscale"]:
        image = make_greyscale(image)

    if config["rescale"]:
        image = rescale(image)

    if "gaussian_blur" in config.keys():
        image = cv.GaussianBlur(image, **config["gaussian_blur"])

    if "CLAHE" in config.keys():
        clahe = cv.createCLAHE(**config["CLAHE"])
        image = clahe.apply(image)

    if "denoise" in config.keys():
        image = cv.fastNlMeansDenoising(image, None, **config["denoise"])

    edge_sum = np.sum(cv.Canny(image, **config["canny"]))

    return edge_sum
