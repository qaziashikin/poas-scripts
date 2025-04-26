import cv2 as cv
import numpy as np

import cv2
import numpy as np
from scipy.spatial import distance
from math import radians, sin, cos, sqrt, atan2

def haversine(coord1, coord2):
    R = 6371000  # Earth radius in meters
    lat1, lon1 = coord1
    lat2, lon2 = coord2

    phi1 = radians(lat1)
    phi2 = radians(lat2)
    d_phi = radians(lat2 - lat1)
    d_lambda = radians(lon2 - lon1)

    a = sin(d_phi / 2.0) ** 2 + cos(phi1) * cos(phi2) * sin(d_lambda / 2.0) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    return R * c


class GeometricCorrector():
    def __init__(self, mask_path):
        self.mask_path = str(mask_path)


    def compute_geographic_distances(self, lat_lng_points):
        num_points = len(lat_lng_points)
        real_world_distances = np.zeros((num_points, num_points))

        for i in range(num_points):
            for j in range(i + 1, num_points):
                dist = haversine(lat_lng_points[i], lat_lng_points[j])
                real_world_distances[i, j] = real_world_distances[j, i] = dist

        return real_world_distances


    def calculate_real_world_area(self, k_nearest=7):
        img_mask = cv2.imread(str(self.mask_path), cv2.IMREAD_GRAYSCALE)

        image_points = np.array([
            [416, 352],
            [225, 362],
            [316, 356],
            [365, 304],
            [274, 277],
            [279, 250],
            [328, 279]
        ], dtype=np.float32)

        lat_lng_points = [
            (10.19532854, -84.22990477),
            (10.19542399, -84.23096819),
            (10.19554582, -84.23054764),
            (10.19587382, -84.23006543),
            (10.19824790, -84.23000994),
            (10.19859818, -84.22970983),
            (10.19756384, -84.22954752)
        ]

        real_world_distances = self.compute_geographic_distances(lat_lng_points)
            
        white_pixels = np.argwhere(img_mask > 200)
        if white_pixels.size == 0:
            return 0.0
        
        xy_pixels = white_pixels[:, [1, 0]].astype(np.float32)
        total_area = 0.0
        
        for pixel in xy_pixels:
            pixel_distances_to_controls = distance.cdist([pixel], image_points)[0]
            nearest_indices = np.argsort(pixel_distances_to_controls)[:k_nearest]
            nearest_points = image_points[nearest_indices]
            
            local_ratios = []
            for i in range(k_nearest):
                for j in range(i + 1, k_nearest):
                    idx1, idx2 = nearest_indices[i], nearest_indices[j]
                    if real_world_distances[idx1, idx2] > 0:
                        dist_img = np.linalg.norm(nearest_points[i] - nearest_points[j])
                        if dist_img > 0:
                            local_ratios.append(real_world_distances[idx1, idx2] / dist_img)
            
            if local_ratios:
                local_scale = np.mean(local_ratios)
                total_area += local_scale ** 2
        
        return total_area
        