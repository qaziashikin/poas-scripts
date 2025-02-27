'''
    Detection of fences to assist other areas of detection.
'''
import cv2 as cv
import pandas as pd
import numpy as np
from utilities import PoasImageUtilities

class FenceDetector:
    def __init__(self):
        """
        Initialize the FenceDetector.
        """
        self.utilities = PoasImageUtilities()


    def get_fence_area_edges(self, img, canny_lower_threshold, canny_upper_threshold):
        """
        Process image and detect edges in the fence area.

        Args:
            img: The image to find the fence area for
            canny_lower_threshold: The lower bound for Canny edge detection
            canny_upper_threshold: The upper bound for Canny edge detection
        """
        image_width = 640
        image_height = 480
        cropped = self.utilities.crop_image(img, 0, image_height, image_width, 100)
        
        if len(cropped.shape) == 3:
            gray = cv.cvtColor(cropped, cv.COLOR_BGR2GRAY)
        else:
            gray = img
        
        gray = cv.normalize(gray, None, 0, 255, cv.NORM_MINMAX).astype('uint8')
        blurred = cv.GaussianBlur(gray, (5, 5), 3)
        
        clahe = cv.createCLAHE(clipLimit = 2.0, tileGridSize = (8, 8))
        enhanced_img = clahe.apply(blurred)
        denoised = cv.fastNlMeansDenoising(enhanced_img, None, h=6)
        
        return cv.Canny(denoised, canny_lower_threshold, canny_upper_threshold)


    def has_fence(self, img, canny_lower_threshold, canny_upper_threshold, edge_sum_threshold):
        """
        Determine if a fence is visible in the image.

        Args:
            img: The image to detect the fence for
            canny_lower_threshold: The lower bound for Canny edge detection
            canny_upper_threshold: The upper bound for Canny edge detection
            edge_sum_threshold: The threshold to use to binarily classify an image
        """
        edge_sum = np.sum(self.get_fence_area_edges(
            img, 
            canny_lower_threshold, 
            canny_upper_threshold
        ))
        
        return edge_sum > edge_sum_threshold
    
    
    def detect_all_fences(self, images_info, canny_lower_threshold, canny_upper_threshold, edge_sum_threshold):
        '''
        Detect all fences in the list of images.

        Args:
            images_info: The hashmap containing information used by the classifier for lookup
            canny_lower_threshold: The lower bound for Canny edge detection
            canny_upper_threshold: The upper bound for Canny edge detection
            edge_sum_threshold: The threshold to use to binarily classify an image
        '''
        for _, image_row in self.images.iterrows():
            img = cv.imread(f"/{image_row['image'].split('?d=')[-1]}")[:1890, :, :]

            if self.has_fence(img, canny_lower_threshold, canny_upper_threshold, edge_sum_threshold):
                 images_info[image_row["id"]]["assignments"].add("Fence")

        return images_info