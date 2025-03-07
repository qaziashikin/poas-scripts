import cv2 as cv
import numpy as np
from pathlib import Path
import shutil
from utilities import PoasImageUtilities
from poas_ir_classifier import IRImageClassifier

class PlotImageFinder:
    '''
        This code allows for identifying the "best" IR and visible image to use on the AVERT website within 
        the plots of the Poás Laguna Caliente lake extent level. This must be run after the IR images are 
        copied locally, as done in generate_lake_extent_masks.
    '''    
    def __init__(self):
        self.poas_ir_classifier = IRImageClassifier()
        self.utilities = PoasImageUtilities()
    

    def extract_numbers(self, filename):
        parts = filename.split("_")
        if len(parts) < 2:
            return None, None
        julian_day = parts[0][-3:].zfill(3)
        time = parts[1][:6]  
        return julian_day, time


    def find_whitest_ir_image(self, folder_path, output_folder):
        print(f"Processing folder: {folder_path}")
        image_files = sorted(folder_path.glob("*.jpg")) 
        
        if not image_files:
            print(f"No images found in {folder_path}.")
            return
        
        output_folder = Path(output_folder)
        output_folder.mkdir(parents=True, exist_ok=True)
        
        julian_day = self.extract_numbers(image_files[0].stem)[0]
        if julian_day is None:
            print(f"Skipping {folder_path}, could not determine Julian day.")
            return
        
        if any(f.name.startswith(julian_day) and f.suffix == ".png" for f in output_folder.iterdir()):
            print(f"Skipping {folder_path}, processed file already exists.")
            return
        
        max_white_pixels = 0
        best_image = None
        best_filename = None
        
        for image_file in image_files:
            _, time = self.extract_numbers(image_file.stem)
            if not time:
                continue
            
            destination = output_folder / f"{julian_day}_{time}.png"
            
            img = cv.imread(str(image_file))
            if img is None or "Degraded" in self.poas_ir_classifier.classify_ir_img(img):
                continue
            
            cropped_img = self.utilities.crop_image(img, 0, 370, 360, 150)
            blurred = cv.GaussianBlur(cropped_img, (5, 5), 3)
            _, white_areas = cv.threshold(blurred, 200, 255, cv.THRESH_BINARY)
            
            white_pixel_count = np.sum(white_areas == 255)
            
            if white_pixel_count > max_white_pixels:
                max_white_pixels = white_pixel_count
                best_image = image_file
                best_filename = destination
        
        if best_image:
            shutil.copy(best_image, best_filename)
            print(f"Copied {best_image.name} to {best_filename}")
        else:
            print(f"No suitable image found for {folder_path.name}.")


    def process_all_days(self, base_folder, output_base_folder):
        base_folder = Path(base_folder)
        output_base_folder = Path(output_base_folder)
        
        for year_folder in sorted(base_folder.iterdir()):
            if not year_folder.is_dir() or not year_folder.name.isdigit():
                continue
            
            year = year_folder.name
            day_folders = sorted(year_folder.iterdir(), key=lambda x: int(x.name.split("_")[0]))
            
            for day_folder in day_folders:
                if not day_folder.is_dir() or not day_folder.name.split("_")[0].isdigit():
                    continue
                
                output_folder = output_base_folder / year
                self.find_whitest_ir_image(day_folder, output_folder)


    def find_closest_visible_image(self, source_folder, target_timestamp):
        closest_image = None
        min_time_diff = float('inf')
        
        for image_file in sorted(source_folder.glob("*.jpg")):
            _, timestamp = self.extract_numbers(image_file.stem)
            if not timestamp:
                continue
            
            image_time = int(timestamp)
            target_time = int(target_timestamp)
            time_diff = abs(image_time - target_time)
            
            if time_diff <= 300 and time_diff < min_time_diff:  
                closest_image = image_file
                min_time_diff = time_diff
        
        return closest_image


    def process_matching_images(self):
        ir_images_folder = Path("outputs/plot_images/ir")
        visible_images_folder = Path("outputs/plot_images/visible")
        archive_folder = Path("/data/vulcand/archive/imagery/visible/345040")
        
        visible_images_folder.mkdir(parents=True, exist_ok=True)
        
        for year_folder in ir_images_folder.iterdir():
            if not year_folder.is_dir() or not year_folder.name.isdigit():
                continue
            
            year = year_folder.name
            archive_year_folder = archive_folder / year
            if not archive_year_folder.exists():
                continue
            
            for ir_image in year_folder.glob("*.png"):
                julian_day, timestamp = self.extract_numbers(ir_image.stem)
                if not julian_day or not timestamp:
                    continue
                
                matching_folders = [f for f in archive_year_folder.glob(f"**/{julian_day}*") if f.is_dir()]
                for folder in matching_folders:
                    closest_image = self.find_closest_visible_image(folder, timestamp)
                    if closest_image:
                        destination = visible_images_folder / year / ir_image.name
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy(closest_image, destination)
                        print(f"Copied {closest_image.name} to {destination}")
                        break  


if __name__ == "__main__":
    plot_image_finder = PlotImageFinder()
    plot_image_finder.process_all_days("outputs/images", "outputs/plot_images/ir")
    plot_image_finder.process_matching_images()