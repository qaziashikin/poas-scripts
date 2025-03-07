import cv2 as cv
import numpy as np

class GeometricCorrector():
    def resize_cropped_image_for_geometric_correction(img):
        top = 220
        bottom = 110
        left = 0 
        right = 640 - 360 

        padded_img = cv.copyMakeBorder(img, top, bottom, left, right, cv.BORDER_CONSTANT, value=[0, 0, 0])
        return padded_img
    
    def apply_geometric_correction(year, day): 
        # Parameters
        resolution = (640, 480) 
        pixel_pitch = 17e-6 
        focal_length = 14e-3 
        height_above_lake = 254  # meters

        sensor_width = resolution[0] * pixel_pitch
        sensor_height = resolution[1] * pixel_pitch

        horizontal_distance = 954.3
        effective_height = np.sqrt(height_above_lake ** 2 + horizontal_distance ** 2)

        gsd_horizontal = (sensor_width * effective_height) / (focal_length * resolution[0])
        gsd_vertical = (sensor_height * effective_height) / (focal_length * resolution[1])

        extension = "_n.png" if year == 2024 else ".png"
        image_path = f"lake_extent/results/{year}/{day}_lake_extent{extension}"
        
        image = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"Image not found at {image_path}")
        
        _, binary = cv.threshold(image, 180, 255, cv.THRESH_BINARY) 
        contours, _ = cv.findContours(binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        total_lake_area_pixels = sum(cv.contourArea(contour) for contour in contours)
        total_lake_area_real = total_lake_area_pixels * gsd_horizontal * gsd_vertical

        return total_lake_area_real