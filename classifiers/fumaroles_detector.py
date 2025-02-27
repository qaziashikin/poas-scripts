'''
    Detection of fumaroles.
'''
import cv2 as cv
import pandas as pd
import numpy as np
from utilities import PoasImageUtilities

class FumarolesDetector:
    def __init__(self):
        """
        Initialize the FumarolesDetector.
        """
        self.utilities = PoasImageUtilities()


    def get_fumaroles_area_contours(self, img):
        """
        Process image and detect edges in the fumaroles area.

        Args:
            img: The image to find the fence area for
        """
        image_width = 640
        cropped = self.utilities.crop_image(img, 545, 190, image_width - 545, 50)

        if len(img.shape) == 3:
            gray = cv.cvtColor(cropped, cv.COLOR_BGR2GRAY)
        else:
            gray = img.copy()

        # Apply CLAHE for contrast enhancement
        clahe = cv.createCLAHE(clipLimit = 3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        denoised = cv.fastNlMeansDenoising(enhanced, None, h=10)

        thresh = cv.adaptiveThreshold(denoised, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 21, 2)

        # Morphological operations to clean up
        kernel = np.ones((3, 3), np.uint8)
        cleaned = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel)

        contours, _ = cv.findContours(cleaned, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        return contours


    def has_fumaroles(self, img, sum_threshold):
        """
        Determine if fumaroles visible in the image.

        Args:
            img: The image to detect the fumaroles for
            sum_threshold: The threshold to use to binarily classify an image
        """
        contours = self.get_fumaroles_area_contours(img)
        contour_sum = np.sum([cv.contourArea(c) for c in contours])
        
        return contour_sum > sum_threshold
    
    
    def detect_all_fumaroles(self, images_info, sum_threshold):
        '''
        Detect all fumaroles in the list of images.

        Args:
            images_info: The hashmap containing information used by the classifier for lookup
            sum_threshold: The threshold to use to binarily classify an image
        '''
        for _, image_row in self.images.iterrows():
            img = cv.imread(f"/{image_row['image'].split('?d=')[-1]}")[:1890, :, :]

            if self.has_fumaroles(img, sum_threshold):
                images_info[image_row["id"]]["assignments"].add("Fumaroles")

        return images_info
