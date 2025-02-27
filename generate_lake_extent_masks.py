import cv2 as cv
import numpy as np
from pathlib import Path
import shutil
from utilities import PoasImageUtilities
from poas_ir_classifier import IRImageClassifier
import sys

class LakeExtentMaskGenerator(): 
    '''
        This code must be run on an IR image taken from the VPMI imaging system at the Poas Volcano.
        Changes to the image type will shift the results.
    '''       
    def __init__(self):
        self.poas_ir_classifier = IRImageClassifier()
        self.utilities = PoasImageUtilities()


    def get_images_by_day(self, day, year, folder_path):
        absolute = Path(f"/data/vulcand/archive/imagery/infrared/345040/{year}/VPMI/still")

        classifier_on_desired_data = folder_path
        classifier_on_desired_data.mkdir(parents=True, exist_ok=True)
        
        day_str = f"{day:03}"
        day_folder = absolute / day_str
        
        if day_folder.exists() and day_folder.is_dir():
            for image_file in day_folder.glob("*.jpg"):  
                shutil.copy(image_file, classifier_on_desired_data)
        else:
            print(f"Day folder does not exist: {day_folder}")


    def determine_threshold(self, plume_count, total_used_images):
        min_white_val = None
        if plume_count < 1000:
            if plume_count < 500:
                min_white_val = 210
            else:
                min_white_val = 220
        else:
            min_white_val = 230

        if plume_count <= 750:
            return 3, min_white_val
        
        thresholds = [
            (850, 3), (950, 4), (1050, 6), (1075, 8), (1100, 10), (1125, 13), (1150, 15), (1200, 16), (1250, 17)
        ]
        
        for threshold, percentage in thresholds:
            if plume_count <= threshold:
                return int(total_used_images * (percentage / 100)), min_white_val
        
        return int(total_used_images * (18 / 100)), min_white_val


    def overlay_white_areas(self, ir_images_path_name, lake_extent_mask_path_name, lake_extent_mask_file_name):
        image_files = []

        Path("outputs").mkdir(parents=True, exist_ok=True)
        plume_percent_file = open(f"outputs/{year}_plume_percent.txt", "a") # TXT file generation for plots
        Path(f"outputs/{lake_extent_mask_path_name}").mkdir(parents=True, exist_ok=True)

        plume_percent_file_path = f"outputs/{year}_plume_percent.txt"

        existing_lines = set()
        if Path(plume_percent_file_path).exists():
            with open(plume_percent_file_path, "r") as f:
                existing_lines = set(line.strip() for line in f.readlines())

        plume_count = 0
        total_images = 0

        for image_file in ir_images_path_name.glob("*.jpg"):
            total_images += 1

            img = cv.imread(str(image_file))
            if img is None:
                continue

            classifications = self.poas_ir_classifier.classify_ir_img(img)

            if "Plume" in classifications:
                plume_count += 1
                if not "Degraded" in classifications:
                    image_files.append(image_file)

        total_used_images = len(image_files)
        if len(image_files) < 2:
            print("Need at least two images to perform overlay.")
            return
        
        result_str = f"{ir_images_path_name}: {plume_count} Plumes; {(plume_count / total_images) * 100}% of all images"
        if result_str not in existing_lines:
            with open(plume_percent_file_path, "a") as plume_percent_file:
                plume_percent_file.write(result_str + "\n")
                print(f"Written to plume percent file: {result_str}")

        plume_percent_file.write(result_str + "\n")

        white_pixel_counts = None

        vote_threshold, min_white_val = self.determine_threshold(plume_count, total_used_images)
        
        for image_file in image_files:
            img = cv.imread(str(image_file))
            img = self.utilities.crop_image(img, 0, 370, 360, 150) # Changing this will affect the geometry correction code

            blurred = cv.GaussianBlur(img, (5, 5), 3)
            _, white_areas = cv.threshold(blurred, min_white_val, 255, cv.THRESH_BINARY)

            if white_pixel_counts is None:
                white_pixel_counts = white_areas.astype(np.int32) // 255
            else:
                # Add 1 for each white pixel in this image - voting
                white_pixel_counts += white_areas // 255

        final_overlay = (white_pixel_counts >= vote_threshold).astype(np.uint8) * 255
        
        output = f"outputs/{lake_extent_mask_path_name}/{lake_extent_mask_file_name}"
        cv.imwrite(output, final_overlay)
        print(f"Lake extent overlay written to {output}")
        print(f"Overlaying white areas for {ir_images_path_name}, saving to {lake_extent_mask_path_name}/{lake_extent_mask_file_name}")
        print(f"Overlay created using a threshold of {vote_threshold} votes")


if __name__ == "__main__":
    if len(sys.argv) not in [3, 4]:
        print("Usage: lake_extent.py <year> <start_day> [<end_day>]")
        sys.exit(1)

    try:
        year = int(sys.argv[1])
        start_day = int(sys.argv[2])
        end_day = int(sys.argv[3]) if len(sys.argv) == 4 else start_day
    except ValueError:
        print("Year, start_day, and end_day must be integers.")
        sys.exit(1)

    if start_day < 1 or start_day > 366 or end_day < 1 or end_day > 366:
        print("Days must be between 1 and 366.")
        sys.exit(1)

    if end_day < start_day:
        print("End day cannot be less than start day.")
        sys.exit(1)

    lake_extent_mask_generator = LakeExtentMaskGenerator()

    for day in range(start_day, end_day + 1):
        ir_images_path_name = Path(f"lake_extent/images/{year}/{day}_lake_extent")
        lake_extent_mask_generator.get_images_by_day(day, year, ir_images_path_name)

        lake_extent_mask_file_name = f"{day}_lake_extent.png"
        lake_extent_mask_path_name = f"lake_extent_masks/{year}"

        lake_extent_mask_generator.overlay_white_areas(ir_images_path_name, lake_extent_mask_path_name, lake_extent_mask_file_name)