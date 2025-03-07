from pathlib import Path
import cv2
import numpy as np
import csv
from datetime import datetime
from geometric_corrector import GeometricCorrector

class LakeExtentStatisticGenerator:
    '''
        This code generates the statistics on the lake extent based on binary masks generated previously. 
        It will compute the amount of white pixels and create a CSV file output for the particular day(s).
        This must be run after the IR images are copied locally, as done in generate_lake_extent_masks.
    '''       
    def count_white_pixels(self, image_path):
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            return None  
        return np.count_nonzero(image > 200)


    def extract_date_from_filename(self, filename, year):
        julian_day = int(filename.split('_')[0])  
        date = datetime.strptime(f"{year}-{julian_day}", "%Y-%j").strftime("%Y-%m-%d")
        return date


    def generate_lake_extent_statistics(self, root_output_dir):
        path_to_lake_extent_masks = Path("outputs/lake_extent_masks")

        for folder in path_to_lake_extent_masks.iterdir():
            if not folder.is_dir():
                continue  
            
            year = folder.name 
            Path(root_output_dir).mkdir(parents=True, exist_ok=True)
            csv_dir = f"{root_output_dir}/csv"
            Path(csv_dir).mkdir(parents=True, exist_ok=True)
            output_file = f"{csv_dir}/lake_extent_estimates_{year}.csv"
            
            mask_png_files = [f for f in folder.glob("*.png") if f.is_file()]

            with open(output_file, "w", newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["Date", "White Pixel Count", "Lake Area (sq meters)"])
                
                for mask_file in mask_png_files:
                    white_pixel_count = self.count_white_pixels(mask_file)
                    if white_pixel_count is not None:
                        date = self.extract_date_from_filename(mask_file.name, year)
                        lake_area = GeometricCorrector.apply_geometric_correction(int(year), int(mask_file.stem.split('_')[0]))
                        writer.writerow([date, white_pixel_count, lake_area])

            print(f"Statistics saved to {output_file}")


if __name__ == "__main__":
    lake_extent_stats_generator = LakeExtentStatisticGenerator()

    root_output_dir = "outputs"
    lake_extent_stats_generator.generate_lake_extent_statistics(root_output_dir)