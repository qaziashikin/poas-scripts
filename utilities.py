'''
    Utility functions to be used across files.
'''
import cv2 as cv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from datetime import datetime, date

images = pd.read_csv("csv/labels.csv")

class PoasImageUtilities:
    def __init__(self):
        self.labels = ["Cloud cover", "Fumaroles", "Low visibility", "No plume", "Plume", "Degraded", "Obscured", "Fence"]
        self.comparison_date = date(2024, 10, 10) # The date after which multi-labeling was introduced in the IR Image set


    def show_image_by_id(self, image_id):
        '''
        Helper function to show an image by its id.

        Args:
            image_id: The ID for an image
        '''
        for idx, image_row in images.iterrows():
            if str(image_row['id']) == str(image_id):
                img = cv.imread(f"/{image_row['image'].split('?d=')[-1]}")[:1890, :, :]
                plt.imshow(img)


    def get_image_by_id(self, image_id):
        '''
        Helper function to get an image by its id.

        Args:
            image_id: The ID for an image
        '''
        for idx, image_row in images.iterrows():
            if str(image_row['id']) == str(image_id):
                return cv.imread(f"/{image_row['image'].split('?d=')[-1]}")[:1890, :, :]
            

    def crop_image(self, img, x, y, h, v):
        '''
        A function that can crop the image. It requires:

        Args: 
            - (x, y): a point in the original image. This will be the bottom left point in the cropped image.
            - h: the horizontal distance that would be the width of the cropped image
            - v: the vertical distance that would be the height of the cropped image

        Thus, the cropped image will be the area enclosed by the points (x, y), (x + h, y), (x, y - v), and (x + h, y - v).
        '''
        height, width = 480, 640

        top_left_x = max(0, x)  
        top_left_y = max(0, y - v)  
        bottom_right_x = min(width, x + h)
        bottom_right_y = min(height, y)

        cropped_img = img[top_left_y:bottom_right_y, top_left_x:bottom_right_x]

        return cropped_img


    def get_stats_dictionary(self, arr):
        '''
        Creates a dictionary of statistics based on the input array.

        Args:
            arr: The input array to compute the statistics for.
        '''
        return {
            "average": np.average(arr),
            "min": np.min(arr),
            "max": np.max(arr),
            "q1": np.quantile(arr, 0.25),
            "median": np.median(arr),
            "q3": np.quantile(arr, 0.75)
        }


    def compute_edge_strength_ratio(self, current_edges, ref_edges):
        '''
        Compute the edge strength ratio of a particular image based on the edges in the image and the edges of the reference image.
        The computation sums the current edges and divides it by the sum of the reference edges.

        Args:
            current_edges = Edges in the image of interest
            ref_edges = Edges in the reference image to use for comparison
        '''
        current_sum_edges = np.sum(current_edges)
        ref_sum_edges = np.sum(ref_edges)
        edge_strength_ratio = current_sum_edges / ref_sum_edges if ref_sum_edges != 0 else 0
        
        return edge_strength_ratio


    def enhance_contrast_in_image(self, img):
        '''
        Enhance the contrast in an image using OpenCV CLAHE

        Args:
            img = An image to enhance the contrast for
        '''
        if len(img.shape) == 3:
            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        else:
            gray = img.copy()

        # Apply CLAHE for contrast enhancement
        clahe = cv.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)

        return cv.cvtColor(enhanced, cv.COLOR_GRAY2BGR)


    def parse_labels(self, choices_column):
        '''
        Parse the CSV choices column to retrieve all assigned labels. Returns a set of all found labels.

        Args:
            choices_column: The CSV choices column string
        ''' 
        labels = ["Cloud cover", 
                "Fumaroles", 
                "Low visibility",
                "No plume",
                "Plume",
                "Degraded",
                "Obscured",
                "Fence"]
        
        pattern = re.compile(r'\b(?:' + '|'.join(labels) + r')\b', re.IGNORECASE)
        
        matches = pattern.findall(choices_column)
        
        return set(matches)


    def is_multi_label(self, timestamp_str):
        parsed_date = datetime.strptime(timestamp_str, "%Y-%m-%dT%H:%M:%S.%fZ").date()
                
        # Return True if parsed_date is after October 10, 2024, else False
        return parsed_date > self.comparison_date


    def compute_accuracy_of_label(self, images_info, label):
        '''
        Compute the total accuracy of a particular label in the data set.

        Args:
            images_info: The hashmap containing information used by the classifier for lookup
            label: The label to compute the accuracy for
        '''
        expected_total, actual_total = 0, 0

        missing_assignments = []
        false_positives = []

        for _, image_row in images.iterrows():
            if label != "Degraded" and "Degraded" in images_info[image_row["id"]]["assignments"]:
                continue
            if label not in self.parse_labels(str(image_row["choice"])):
                if self.is_multi_label(image_row["updated_at"]): 
                    if label in images_info[image_row["id"]]["assignments"]:
                        false_positives.append(image_row["id"])         
                continue

            expected_total += 1
            if label in images_info[image_row["id"]]["assignments"]:
                actual_total += 1
            else:
                missing_assignments.append(image_row["id"])

        accuracy = actual_total / expected_total * 100
        print("Number of expected images: " + str(expected_total))

        return accuracy, missing_assignments, false_positives