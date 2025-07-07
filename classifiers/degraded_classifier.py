"""
Classification of degraded images.

This conditional classifier is run prior to other classifiers, as degraded
images are generally not useful.

:author:
    Qazi T. Ashikin

"""

import cv2 as cv
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

from .utils import make_greyscale


def _calculate_column_pixel_means(
    image: np.ndarray, column_width: int = 40
) -> np.ndarray:
    """
    Compute the average pixel values in the image for columns of a specified pixel
    width.

    Parameters
    ----------
    image:
        Array of pixel values that constitute the image.
    column_width:
        The number of horizontal pixels in each column. Default: 40.

    Returns
    -------
     :
        The average pixel values in fixed width columns across the image.

    """

    assert image.shape[1] % column_width == 0

    n_columns = image.shape[1] // column_width
    blocks = image.reshape(n_columns, image.shape[0], column_width)
    block_averages = [np.mean(block.flatten()) for block in blocks]

    return np.asarray(block_averages)


def _fit_quadratic(column_averages: np.ndarray) -> tuple[float, float, float]:
    """
    Compute the best-fitting coefficients for a quadratic fit to the column pixels
    averages.

    Parameters
    ----------
    column_averages:
        The average pixel values in fixed width columns across the image.

    Returns
    -------
     :
        Best-fitting quadratic coefficients, determined using linear regression.

    """

    X = np.array(range(len(column_averages))).reshape(-1, 1)
    X_poly = PolynomialFeatures(degree=2).fit_transform(X)

    model = LinearRegression()
    model.fit(X_poly, column_averages)

    a = model.coef_[2]
    b = model.coef_[1]
    c = model.intercept_

    return a, b, c


def is_degraded(image: np.ndarray, threshold: float = 0.4) -> bool:
    """
    Determine whether an image exhibits the characteristic 'degraded' feature that is
    observed for the Calibir GXM IR camera.

    This degradation, thought to arise during periods when the camera is adjusting the
    dynamic range of the sensor, appears as a characteristic pattern of brighter pixels
    in the outer columns, through to darker pixels in the centre of the images. This is
    detected by computing the average pixel values for fixed-width columns in the image
    and fitting a polynomial function to it. The polynomial for degraded images will
    have a ~quadratic shape.

    Parameters
    ----------
    image:
        The image to be classified.
    threshold:
        The threshold above which an image is classified as degraded. Default: 0.4.

    Returns
    -------
     :
        Classification of the image, where True indicates the image is degraded.

    """

    greyscale_image = make_greyscale(image)
    blurred_image = cv.GaussianBlur(greyscale_image, (35, 35), 15)
    a, _, _ = _fit_quadratic(_calculate_column_pixel_means(blurred_image))

    return a > threshold
