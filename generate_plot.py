import pathlib
import textwrap
from datetime import datetime, timedelta as td

import cv2 
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.interpolate import CubicSpline
from poas_ir_classifier import IRImageClassifier
from utilities import PoasImageUtilities

plt.style.use("basic_style")

class PlotGenerator:
    def __init__(self):
        self.utilities = PoasImageUtilities()
        self.poas_ir_classifier = IRImageClassifier()

    def crop_image(self, img, x, y, h, v):
        """Crop an image based on given coordinates and dimensions."""

        height, width = 480, 640

        top_left_x = max(0, x)
        top_left_y = max(0, y - v)
        bottom_right_x = min(width, x + h)
        bottom_right_y = min(height, y)

        cropped_img = img[top_left_y:bottom_right_y, top_left_x:bottom_right_x]

        return cropped_img


    def resize_cropped_image_for_geometric_correction(self, img):
        """Add padding to an image for geometric correction."""
        top = 220
        bottom = 110
        left = 0
        right = 640 - 360

        padded_img = cv2.copyMakeBorder(
            img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0]
        )
        return padded_img


    def extract_numbers(self, filename):
        parts = filename.split("_")
        if len(parts) < 2:
            return None, None
        julian_day = parts[0][-3:].zfill(3)
        time = parts[1][:6]  
        return julian_day, time


    def find_best_ir_image(self, folder_path):
        print(f"Processing folder: {folder_path}")
        image_files = sorted(folder_path.glob("*.jpg")) 
        
        if not image_files:
            print(f"No images found in {folder_path}.")
            return
        
        max_white_pixels = 0
        best_image = None
        
        for image_file in image_files:
            _, time = self.extract_numbers(image_file.stem)
            if not time:
                continue
                        
            img = cv2.imread(str(image_file))
            if img is None or "Degraded" in self.poas_ir_classifier.classify_ir_img(img):
                continue
            
            cropped_img = self.utilities.crop_image(img, 0, 370, 360, 150)
            blurred = cv2.GaussianBlur(cropped_img, (5, 5), 3)
            _, white_areas = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY)
            
            white_pixel_count = np.sum(white_areas == 255)
            
            if white_pixel_count > max_white_pixels:
                max_white_pixels = white_pixel_count
                best_image = image_file
        
        if best_image is None:
            print(f"Warning: no IR image found in {folder_path}")
            return
        return best_image


    from datetime import datetime

    def find_closest_visible_image(self, source_folder, target_timestamp, threshold_seconds=300):
        """
        Find the visible image whose timestamp is within ±threshold_seconds
        of the target IR timestamp.
        """
        print(f"Processing folder: {source_folder}")
        try:
            t0 = datetime.strptime(target_timestamp, "%H%M%S")
        except ValueError:
            print(f"Warning: couldn't parse target timestamp {target_timestamp}")
            return

        target_secs = t0.hour * 3600 + t0.minute * 60 + t0.second

        closest_image = None
        min_diff = float('inf')

        for image_file in sorted(source_folder.glob("*.jpg")):
            _, ts = self.extract_numbers(image_file.stem)
            if not ts:
                continue

            try:
                ti = datetime.strptime(ts, "%H%M%S")
            except ValueError:
                continue

            img_secs = ti.hour * 3600 + ti.minute * 60 + ti.second
            diff = abs(img_secs - target_secs)

            if diff <= threshold_seconds and diff < min_diff:
                closest_image = image_file
                min_diff = diff

        if closest_image is None:
            print(f"Warning: No visible image found for Julian Day {str(source_folder).split('/')[-1]} within ±{threshold_seconds}s of {target_timestamp}")
            return

        img = cv2.imread(str(closest_image))
        if img is None:
            print(f"Error loading {closest_image}")
            return

        return img
                    

    def overlay_mask_on_ir(
        self,
        ir_image, 
        mask_path,
        mask_color=(127, 255, 212),
        ir_fade_factor=0.9,
        mask_fade_factor=0.95,
    ):
        """
        Overlays a colored mask on top of a faded infrared (IR) image.

        Args:
            ir_image: The IR image file.
            mask_path (str): Path to the mask image file (should have the same dimensions as the IR image).
                            The mask should typically be a grayscale or binary image where the masked
                            region has non-zero values.
            output_path (str): Path to save the output overlaid image.
            mask_color (tuple): RGB tuple representing the color of the mask (default is blue: (255, 0, 0)).
            ir_fade_factor (float): Factor to control the fading of the IR image (0.0 to 1.0).
                                    A value of 1.0 means no fading, and 0.0 means completely black.
            mask_fade_factor (float): Factor to control the fading of the mask color (0.0 to 1.0).
                                       A value of 1.0 means no fading, and 0.0 means completely transparent.
        """
        try:
            if isinstance(ir_image, (str, Path)):
                ir = cv2.imread(str(ir_image), cv2.IMREAD_GRAYSCALE)
            else:
                if ir_image.ndim == 3:
                    ir = cv2.cvtColor(ir_image, cv2.COLOR_BGR2GRAY)
                else:
                    ir = ir_image.copy()

            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            mask = self.resize_cropped_image_for_geometric_correction(mask)

            faded_ir = (ir * ir_fade_factor).astype(np.uint8)
            faded_ir_colored = cv2.cvtColor(faded_ir, cv2.COLOR_GRAY2BGR)

            colored_mask = np.zeros_like(faded_ir_colored)
            colored_mask[:] = mask_color

            mask_bool = mask > 0  

            blended = (
                (1 - mask_fade_factor) * faded_ir_colored
                + mask_fade_factor * colored_mask
            ).astype(np.uint8)

            overlaid = faded_ir_colored.copy()

            overlaid[mask_bool] = blended[mask_bool]

            return overlaid
        except Exception as e:
            print(f"Error while overlaying mask on IR image: {e}")


    def load_and_prepare_ir_and_visible_image(
        self,
        ir_img_folder_path,
        vis_img_folder_path,
        julian_day,
        year,
        ax,
        plot_ir: bool = True,
        plot_vis: bool = True,
    ):
        ir_img = self.find_best_ir_image(ir_img_folder_path)
        if plot_ir:
            if ir_img is None:
                return
            mask_path = (
                pathlib.Path.cwd() /
                f"outputs/lake_extent_masks/{year}/{int(julian_day)}_lake_extent.png"
            )
            overlaid_image = self.overlay_mask_on_ir(ir_image=ir_img, mask_path=str(mask_path))
            ax.imshow(overlaid_image[12:435, :620])
            ax.set_axis_off()

        if plot_vis:
            if ir_img is None:
                return
            _, timestamp = self.extract_numbers(ir_img.stem)
            vis_img = self.find_closest_visible_image(vis_img_folder_path, timestamp)
            if vis_img is None:
                return
            vis_img = cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB)
            vis_img_resized = cv2.resize(vis_img, (640, 480))

            tx, ty = -65, -78
            M = np.float32([[1, 0, tx], [0, 1, ty]])
            rows, cols, _ = vis_img_resized.shape
            shifted = cv2.warpAffine(vis_img_resized, M, (cols, rows))
            crop = shifted[abs(ty):, abs(tx):]
            stretched = cv2.resize(crop, (640, 480))

            h, w, _ = stretched.shape
            start_row, end_row = 0, h - (abs(ty) + 17)
            start_col, end_col = 0, w - (abs(tx) + 10)

            ax.imshow(stretched[start_row:end_row, start_col:end_col])
            ax.set_axis_off()


    # Old Method
    # def detect_outliers(self, pixels, window_size=4, threshold=0.75):
    #     """Detect outliers in pixel data."""

    #     def is_outlier(i, pixels):
    #         if i < 1 or i > len(pixels) - (window_size - 1):
    #             return False

    #         group = pixels[i - 1 : i + (window_size - 1)]
    #         for j in range(1, len(group)):
    #             if abs(group[j] - group[j - 1]) > threshold * max(group[j - 1], group[j]):
    #                 return True
    #         return False

    #     outliers = np.zeros(len(pixels), dtype=bool)
    #     for i in range(len(pixels)):
    #         if i == 0 or i == len(pixels) - 1:
    #             continue
    #         if is_outlier(i, pixels):
    #             outliers[i] = True
    #         if i < window_size:
    #             continue
    #         if (
    #             abs(pixels[i] - pixels[i - window_size])
    #             > threshold * pixels[i - (window_size - 1)]
    #         ):
    #             outliers[i] = True

    #     return outliers


    # Running Z-score method
    def detect_outliers_zscore(self, data, window_size=10, threshold=1):
        """Detect outliers using a rolling Z-score."""
        outliers = np.zeros(len(data), dtype=bool)
        rolling_mean = pd.Series(data).rolling(window=window_size, center=True).mean()
        rolling_std = pd.Series(data).rolling(window=window_size, center=True).std()
        z_scores = np.abs((data - rolling_mean) / rolling_std)
        outliers = z_scores > threshold
        return outliers.fillna(False).values  # Handle NaN at the edges


    def prepare_lake_extent_data(self, path_to_csvs):
        """Load and prepare lake extent data for all available years in the specified folder."""
        path = Path(path_to_csvs)
        all_csvs = list(path.glob("lake_extent_estimates_*.csv"))

        lake_extent_dfs = []
        for csv_file in all_csvs:
            try:
                df = pd.read_csv(csv_file)
                df["Date"] = pd.to_datetime(df["Date"])
                lake_extent_dfs.append(df)
            except Exception as e:
                print(f"Skipping {csv_file.name} due to error: {e}")

        if not lake_extent_dfs:
            raise FileNotFoundError("No valid lake extent CSVs found.")

        lake_extents = pd.concat(lake_extent_dfs).sort_values(by="Date").reset_index(drop=True)
        lake_extents["Lake Area (sq meters)"] = pd.to_numeric(lake_extents["Lake Area (sq meters)"], errors='coerce')

        pixels = lake_extents["White Pixel Count"].values
        
        # 1. Original method (comment out to use Z-score)
        # detected_outliers = detect_outliers(pixels)

        # 2. Z-score method (uncomment to use Z-score, comment out the original method call above)
        detected_outliers = self.detect_outliers_zscore(pixels)

        eval_mask = lake_extents['Outlier'].astype(str).str.lower().str.strip().isin(['true', 'false'])

        if 'Outlier' not in lake_extents.columns:
            lake_extents['Outlier'] = pd.Series([None] * len(lake_extents))

        lake_extents.loc[eval_mask, 'Outlier'] = detected_outliers[eval_mask]

        return lake_extents


    def plot_lake_info(self, ax, date, pixel_count, surface_area):
        """Plot lake information in the info panel."""
        wrapped_date = textwrap.fill(date.strftime("%Y-%m-%d"), width=20)
        wrapped_pixel_count = textwrap.fill(str(pixel_count), width=20)
        wrapped_surface_area = textwrap.fill(f"{surface_area:.2f}", width=20)
        heading_fontsize = 12
        val_fontsize = 11

        ax.text(
            0.45,
            0.65,
            "Date:",
            va="center",
            ha="center",
            fontsize=heading_fontsize,
            weight="bold",
            linespacing=1.2,
        )
        ax.text(
            0.45,
            0.59,
            f"{wrapped_date}",
            va="center",
            ha="center",
            fontsize=val_fontsize,
            linespacing=1.2,
        )
        ax.text(
            0.45,
            0.515,
            "Pixel Coverage:",
            va="center",
            ha="center",
            fontsize=heading_fontsize,
            weight="bold",
            linespacing=1.2,
        )
        ax.text(
            0.45,
            0.455,
            f"{wrapped_pixel_count} / 307200",
            va="center",
            ha="center",
            fontsize=val_fontsize,
            linespacing=1.2,
        )
        ax.text(
            0.45,
            0.38,
            "Surface Area:",
            va="center",
            ha="center",
            fontsize=heading_fontsize,
            weight="bold",
            linespacing=1.2,
        )
        ax.text(
            0.45,
            0.32,
            f"{wrapped_surface_area} sq. m",
            va="center",
            ha="center",
            fontsize=val_fontsize,
            linespacing=1.2,
        )


    def plot_lake_extent_timeseries(self, ax, lake_extents, non_outliers_df, outliers_df, date):
        """Plot the lake extent timeseries with outliers marked."""
        non_outliers_df = non_outliers_df.copy()

        # Ensure 'Date' column is datetime objects
        non_outliers_df["Date"] = pd.to_datetime(non_outliers_df["Date"])

        # Ensure 'Lake Area (sq meters)' is numeric and handle potential non-numeric values
        non_outliers_df.loc[:, "Lake Area (sq meters)"] = pd.to_numeric(
            non_outliers_df["Lake Area (sq meters)"], errors="coerce"
        )
        non_outliers_df = non_outliers_df.dropna(subset=["Lake Area (sq meters)"])

        non_outliers_df.loc[:, "Averages"] = non_outliers_df.rolling(
            "8D", on="Date", center=True
        ).mean(numeric_only=True)["Lake Area (sq meters)"]

        spline = CubicSpline(
            [pd.to_datetime(date).timestamp() for date in non_outliers_df["Date"].values],
            non_outliers_df["Averages"].values,
        )

        smooth_counts = spline(
            [pd.to_datetime(date).timestamp() for date in lake_extents["Date"]]
        )

        ax.scatter(
            non_outliers_df["Date"],
            non_outliers_df["Lake Area (sq meters)"],
            color="k",
            marker="+",
            s=10,
            label="Daily Data",
        )
        ax.scatter(
            outliers_df["Date"],
            outliers_df["Lake Area (sq meters)"],
            color="gray",
            marker="x",
            s=5,
            label="Omitted Outliers",
            alpha=0.7,
        )
        ax.plot(
            lake_extents["Date"],
            smooth_counts,
            color=(0 / 255, 128 / 255, 128 / 255),
            alpha=0.7,
            linewidth=1.5,
            label="Adjusted Running Average",
        )

        ax.axvline(date, color="navy", linestyle="dashdot", linewidth=1)

        ax.set_xlim([lake_extents["Date"].min(), lake_extents["Date"].max()])
        ax.set_ylim([0, lake_extents["Lake Area (sq meters)"].max() * 1.1])
        ax.set_ylabel("Lake Area (square meters)")
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        ax.set_xlabel("Date")
        ax.legend()

        tick_labels = ax.get_xticklabels()
        if tick_labels:
            for label in tick_labels[1:]:
                label_text = label.get_text()
                try:
                    label_date = datetime.strptime(label_text, "%b %Y")
                    if label_date.month != 1:
                        label.set_text(label_date.strftime("%b"))
                except ValueError:
                    pass


    def save_updated_data(self, lake_extents: pd.DataFrame, path_to_csvs: str):
        """Save updated lake extent data split by year."""
        path = Path(path_to_csvs)
        for year in lake_extents['Date'].dt.year.unique():
            df_year = lake_extents[lake_extents['Date'].dt.year == year].copy()
            out_path = path / f"lake_extent_estimates_{year}.csv"
            df_year.to_csv(out_path, index=False, columns=["Date", "White Pixel Count", "Lake Area (sq meters)", "Outlier"])


    def plot_lake_extent_with_grayed_out_outliers(
        self,
        path_to_csvs: str, 
        vis_imgs_for_plot_folder_path: str,
        ir_imgs_for_plot_folder_path: str,
        julian_day: int, 
        year: int):
        """Create lake extent visualization, showing outliers in gray but not counting them."""
        fig = plt.figure(figsize=(29 / 2.54, 18 / 2.54), layout="constrained")
        axes = fig.subplot_mosaic(
            """
            AABBC
            DDDDD
            """
        )

        for ax_label in "ABC":
            axes[ax_label].set_xticklabels([])
            axes[ax_label].set_yticklabels([])
            axes[ax_label].set_xticks([])
            axes[ax_label].set_yticks([])

        # Plot visible and IR images
        vis_img_folder_path = Path(vis_imgs_for_plot_folder_path) / str(year) / "VPMI" / "still" / f"{julian_day:03d}"
        ir_img_folder_path = Path(ir_imgs_for_plot_folder_path) / str(year) / "VPMI" / "still" / f"{julian_day:03d}"

        self.load_and_prepare_ir_and_visible_image(
            ir_img_folder_path,
            vis_img_folder_path,
            julian_day,
            year,
            axes["B"],
            plot_ir=True,
            plot_vis=False,
        )
        self.load_and_prepare_ir_and_visible_image(
            ir_img_folder_path,
            vis_img_folder_path,
            julian_day,
            year,
            axes["A"],
            plot_ir=False,
            plot_vis=True,
        )

        # Plot lake assessment information
        axes["C"].spines[:].set_visible(False)
        date = datetime(year, 1, 1) + td(days=julian_day - 1)

        try:
            lake_extents = self.prepare_lake_extent_data(path_to_csvs)
            outliers_df = lake_extents[
                lake_extents["Outlier"].astype(str).str.lower().str.startswith("true")
            ]
            non_outliers_df = lake_extents[
                lake_extents["Outlier"].astype(str).str.lower().str.startswith("false")
            ]

            # Filter out rows with NaN in 'Lake Area (sq meters)' for non_outliers_df
            non_outliers_df = non_outliers_df[
                pd.notna(non_outliers_df["Lake Area (sq meters)"])
            ]

            day_data = lake_extents[lake_extents["Date"] == date]
            pixel_count = (
                day_data["White Pixel Count"].values[0] if not day_data.empty else 0
            )
            surface_area = (
                day_data["Lake Area (sq meters)"].values[0] if not day_data.empty else 0
            )

            self.plot_lake_info(axes["C"], date, pixel_count, surface_area)
            self.plot_lake_extent_timeseries(
                axes["D"], lake_extents, non_outliers_df, outliers_df, date
            )
            self.save_updated_data(
                lake_extents=lake_extents,
                path_to_csvs=path_to_csvs
            )

        except FileNotFoundError as e:
            print(f"Error: Could not find lake extent data: {e}")

        return fig, axes

if __name__ == "__main__":
    plot_generator = PlotGenerator()
    julian_day = 325
    year = 2024
    fig, axes = plot_generator.plot_lake_extent_with_grayed_out_outliers(
        path_to_csvs="outputs/data",
        vis_imgs_for_plot_folder_path="/data/vulcand/archive/imagery/visible/345040",
        ir_imgs_for_plot_folder_path="/data/vulcand/archive/imagery/infrared/345040",
        julian_day=julian_day, 
        year=year
    )
    # Ensure the "outputs/plots" directory exists
    output_dir = Path("outputs/plots")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save the figure
    output_path = output_dir / f"poas-lake-extent_{year}-{julian_day}.png"
    fig.savefig(output_path, dpi=400)