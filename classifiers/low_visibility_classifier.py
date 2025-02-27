'''
    Classification of low visibility images. Conditional classifier - depends on other detections to finish prior.
'''
import cv2 as cv
import pandas as pd
import numpy as np
from utilities import PoasImageUtilities

class LowVisibilityClassifier:
    def __init__(self):
        """
        Initialize the class LowVisibilityClassifier.
        """
        self.utilities = PoasImageUtilities()


    def get_edges_in_plume_area(self, img):
        # Crop the image to consider only the plume area of interest
        cropped = self.utilities.crop_image(img, 40, 345, 300, 120)
        gray = cv.cvtColor(cropped, cv.COLOR_BGR2GRAY)

        blurred = cv.GaussianBlur(gray, (5, 5), 3)
        clahe = cv.createCLAHE(clipLimit = 2.5, tileGridSize = (8, 8))
        new = clahe.apply(blurred)

        return cv.Canny(new, 20, 60)
    

    def has_visible_components_in_plume_area(self, img):
        return np.sum(self.get_edges_in_plume_area(img)) > 15000