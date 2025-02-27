'''
    Detection of plumes.
'''
import cv2 as cv
import pandas as pd
import numpy as np
from utilities import PoasImageUtilities
from classifiers.fence_detector import FenceDetector
from classifiers.fumaroles_detector import FumarolesDetector

class PlumeDetector:
    def __init__(self):
        """
        Initialize the PlumeDetector.
        """
        self.definitely_plume_threshold = self.compute_threshold()
        self.fence_detector = FenceDetector()
        self.fumaroles_detector = FumarolesDetector()
        self.utilities = PoasImageUtilities()

    def compute_threshold(self):
        '''
        Computes the threshold for Stage 1 Detection. Value should be hard coded in classifier.
        '''
        # l, c = [], []

        # for _, image_row in self.images.iterrows():
        #     img = cv.imread(f"/{image_row['image'].split('?d=')[-1]}")[:1890, :, :]

        #     cropped_img = crop_image(img, 40, 365, 300, 130)
        #     edges = cv.Canny(cropped_img, 150, 220)

        #     if image_row['choice'] == "Cloud cover":
        #         c.append(np.sum(edges))
        #     if image_row['choice'] == "Low visibility":
        #         l.append(np.sum(edges))

        # low_visibility_stats = get_stats_dictionary(l)
        # cloud_cover_stats = get_stats_dictionary(c)

        # print("Plume Stage 1 Edge Threshold: " + str(max(low_visibility_stats["max"], cloud_cover_stats["max"])))
        # return max(low_visibility_stats["max"], cloud_cover_stats["max"])

        return 91000
    

    def stage_1_has_plume(self, img):
        '''
        Determine if the image has a plume based on the Stage 1 classification criteria.

        Args:
            img: The image to detect the plume for
        '''
        cropped_img = self.utilities.crop_image(img, 40, 365, 300, 130)
        gray = cv.cvtColor(cropped_img, cv.COLOR_BGR2GRAY)

        blurred = cv.GaussianBlur(gray, (5, 5), 3)
        clahe = cv.createCLAHE(clipLimit = 3.0, tileGridSize = (8, 8))
        new = clahe.apply(blurred)
        edges = cv.Canny(new, 80, 150)

        return np.sum(edges) > self.definitely_plume_threshold
    

    def stage_2_has_plume(self, img):
        '''
        Determine if the image has a plume based on the Stage 2 classification criteria.

        Args:
            img: The image to detect the plume for
        '''
        # Look for STRONG fences and STRONG fumaroles
        return self.fence_detector.has_fence(img, 150, 220, 65000) and self.fumaroles_detector.has_fumaroles(img, 450)


    def detect_all_plumes(self, images_info):
        '''
        Detect all plumes in the list of images.

        Args:
            images_info: The hashmap containing information used by the classifier for lookup
            sum_threshold: The threshold to use to binarily classify an image
        '''
        for _, image_row in self.images.iterrows():
            img = cv.imread(f"/{image_row['image'].split('?d=')[-1]}")[:1890, :, :]

            if self.stage_1_has_plume(img) or self.stage_2_has_plume(img):
                images_info[image_row["id"]]["assignments"].add("Plume")

        return images_info