import pathlib
import textwrap
from datetime import datetime, timedelta as td
import cv2
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
import os

def crop_image(img, x, y, h, v):
    """Crop an image based on given coordinates and dimensions."""
    height, width = 480, 640

    top_left_x = max(0, x)
    top_left_y = max(0, y - v)
    bottom_right_x = min(width, x + h)
    bottom_right_y = min(height, y)

    cropped_img = img[top_left_y:bottom_right_y, top_left_x:bottom_right_x]

    return cropped_img


def resize_cropped_image_for_geometric_correction(img):
    """Add padding to an image for geometric correction."""
    top = 220
    bottom = 110
    left = 0
    right = 640 - 360

    padded_img = cv2.copyMakeBorder(
        img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0]
    )
    return padded_img


def overlay_mask_on_ir(
    ir_image_path,
    mask_path,
    mask_color=(127, 255, 212),
    ir_fade_factor=0.9,
    mask_fade_factor=0.95,
):
    """Overlay a colored mask on top of a faded infrared (IR) image."""
    try:
        ir_image = cv2.imread(ir_image_path, cv2.IMREAD_GRAYSCALE)
        if ir_image is None:
            print(f"Error: Could not load IR image from {ir_image_path}")
            return None

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"Error: Could not load mask from {mask_path}")
            return None

        mask = resize_cropped_image_for_geometric_correction(mask)

        faded_ir = (ir_image * ir_fade_factor).astype(np.uint8)
        faded_ir_colored = cv2.cvtColor(faded_ir, cv2.COLOR_GRAY2BGR)

        colored_mask = np.zeros_like(faded_ir_colored)
        colored_mask[:] = mask_color

        masked_region = mask > 0

        overlaid_image = faded_ir_colored.copy()
        overlaid_image[masked_region] = (
            (1 - mask_fade_factor) * faded_ir_colored[masked_region]
            + mask_fade_factor * colored_mask[masked_region]
        ).astype(np.uint8)

        return overlaid_image
    except Exception as e:
        print(f"Error while overlaying mask on IR image: {e}")
        return None


def load_and_prepare_visible_image(vis_img_path, julian_day, year, ax):
    """Load and prepare visible image for plotting."""
    vis_img_files = list((vis_img_path).glob(f"{julian_day:03d}_*.png"))
    if not vis_img_files:
        print(
            f"Warning: No visible image found for Julian day {julian_day} and year {year} at {vis_img_path}"
        )
        return

    img = cv2.imread(str(vis_img_files[0].resolve()), cv2.IMREAD_COLOR)
    if img is None:
        print(f"Error: Could not load visible image from {vis_img_files[0].resolve()}")
        return

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img, (640, 480))

    tx = -65
    ty = -78
    translation_matrix = np.float32([[1, 0, tx], [0, 1, ty]])
    rows, cols, _ = img_resized.shape
    shifted_img = cv2.warpAffine(img_resized, translation_matrix, (cols, rows))

    cropped_img = shifted_img[abs(ty) :, abs(tx) :]
    stretched_img = cv2.resize(cropped_img, (640, 480))

    height, width, _ = stretched_img.shape
    top_crop = 0
    bottom_crop = abs(ty) + 17
    left_crop = 0
    right_crop = abs(tx) + 10

    start_row = top_crop
    end_row = height - bottom_crop
    start_col = left_crop
    end_col = width - right_crop

    ax.imshow(stretched_img[start_row:end_row, start_col:end_col])


def load_and_prepare_ir_image(ir_img_path, julian_day, year, ax):
    """Load and prepare IR image with mask overlay for plotting."""
    ir_img_files = list((ir_img_path).glob(f"{julian_day:03d}_*.png"))
    if not ir_img_files:
        print(
            f"Warning: No IR image found for Julian day {julian_day} and year {year} at {ir_img_path}"
        )
        return

    ir_img = ir_img_files[0].resolve()
    extension = "_lake_extent_n.png" if year == 2024 else "_lake_extent.png"
    mask_path = (
        pathlib.Path.cwd() / f"../data/lake_extent/results/{year}/{int(julian_day)}{extension}"
    )

    if not mask_path.exists():
        print(
            f"Warning: No mask found for Julian day {julian_day} and year {year} at {mask_path}"
        )
        return

    overlaid_image = overlay_mask_on_ir(
        ir_image_path=str(ir_img), mask_path=str(mask_path)
    )
    if overlaid_image is not None:
        ax.imshow(overlaid_image[12:435, :620])


# Old Method
def detect_outliers(pixels, window_size=4, threshold=0.75):
    """Detect outliers in pixel data."""

    def is_outlier(i, pixels):
        if i < 1 or i > len(pixels) - (window_size - 1):
            return False

        group = pixels[i - 1 : i + (window_size - 1)]
        for j in range(1, len(group)):
            if abs(group[j] - group[j - 1]) > threshold * max(group[j - 1], group[j]):
                return True
        return False

    outliers = np.zeros(len(pixels), dtype=bool)
    for i in range(len(pixels)):
        if i == 0 or i == len(pixels) - 1:
            continue
        if is_outlier(i, pixels):
            outliers[i] = True
        if i < window_size:
            continue
        if (
            abs(pixels[i] - pixels[i - window_size])
            > threshold * pixels[i - (window_size - 1)]
        ):
            outliers[i] = True

    return outliers


def detect_outliers_zscore(data, window_size=10, threshold=1):
    """Detect outliers using a rolling Z-score."""
    outliers = np.zeros(len(data), dtype=bool)
    rolling_mean = pd.Series(data).rolling(window=window_size, center=True).mean()
    rolling_std = pd.Series(data).rolling(window=window_size, center=True).std()
    z_scores = np.abs((data - rolling_mean) / rolling_std)
    outliers = z_scores > threshold
    return outliers.fillna(False).values  


def prepare_lake_extent_data():
    """Load and prepare lake extent data, evaluating True/False strings."""
    lake_extents_2024 = pd.read_csv("../data/lake_extent/lake_extent_estimates_2024.csv")
    lake_extents_2025 = pd.read_csv("../data/lake_extent/lake_extent_estimates_2025.csv")
    lake_extents = pd.concat([lake_extents_2024, lake_extents_2025]).reset_index(
        drop=True
    )
    lake_extents["Date"] = pd.to_datetime(lake_extents["Date"])
    lake_extents = lake_extents.sort_values(by=["Date"]).reset_index(drop=True)

    lake_extents["Lake Area (sq meters)"] = pd.to_numeric(
        lake_extents["Lake Area (sq meters)"], errors="coerce"
    )

    pixels = lake_extents["White Pixel Count"].values

    # Choose which outlier detection method to use:
    # 1. Original method (comment out to use Z-score)
    # detected_outliers = detect_outliers(pixels)

    # 2. Z-score method (uncomment to use Z-score, comment out the original method call above)
    detected_outliers = detect_outliers_zscore(pixels)

    # Create a mask for rows where 'Outlier' is 'True' or 'False' (case-insensitive)
    eval_mask = (
        lake_extents["Outlier"]
        .astype(str)
        .str.lower()
        .str.strip()
        .isin(["true", "false"])
    )

    if "Outlier" not in lake_extents.columns:
        lake_extents["Outlier"] = pd.Series([None] * len(lake_extents))
    else:
        pass

    lake_extents.loc[eval_mask, "Outlier"] = detected_outliers[eval_mask]

    return lake_extents


def plot_lake_extent_timeseries(ax, lake_extents, non_outliers_df, outliers_df, date, second_date):
    """Plot the lake extent timeseries with outliers marked and a crosshair for the second date."""
    non_outliers_df = non_outliers_df.copy()

    non_outliers_df["Date"] = pd.to_datetime(non_outliers_df["Date"])

    non_outliers_df.loc[:, "Lake Area (sq meters)"] = pd.to_numeric(
        non_outliers_df["Lake Area (sq meters)"], errors="coerce"
    )
    non_outliers_df = non_outliers_df.dropna(subset=["Lake Area (sq meters)"])

    non_outliers_df.loc[:, "Averages"] = non_outliers_df.rolling(
        "8D", on="Date", center=True
    ).mean(numeric_only=True)["Lake Area (sq meters)"]

    spline = CubicSpline(
        [pd.to_datetime(date_val).timestamp() for date_val in non_outliers_df["Date"].values],
        non_outliers_df["Averages"].values,
    )

    smooth_counts = spline(
        [pd.to_datetime(date_val).timestamp() for date_val in lake_extents["Date"]]
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

    second_date_data = lake_extents[lake_extents["Date"] == second_date]
    if not second_date_data.empty:
        ax.plot(
            second_date_data["Date"],
            second_date_data["Lake Area (sq meters)"],
            color="blue",
            marker="+",
            markersize=9,
            linestyle="none",
            label='_nolegend_',
            markeredgecolor='blue',
            markeredgewidth=1
        )

    ax.set_xlim([lake_extents["Date"].min(), lake_extents["Date"].max()])
    ax.set_ylim([0, lake_extents["Lake Area (sq meters)"].max() * 1.1])
    ax.set_ylabel("Lake Area (square meters)")
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
    ax.set_xlabel("Date")
    ax.legend()

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    
    return second_date_data


def save_updated_data(lake_extents):
    """Save updated lake extent data with outliers marked."""
    lake_extents_2024_updated = lake_extents[
        lake_extents["Date"].dt.year == 2024
    ].copy()
    lake_extents_2025_updated = lake_extents[
        lake_extents["Date"].dt.year == 2025
    ].copy()

    columns_order = ["Date", "White Pixel Count", "Lake Area (sq meters)", "Outlier"]
    lake_extents_2024_updated.to_csv(
        "../data/lake_extent/lake_extent_estimates_2024.csv", index=False, columns=columns_order
    )
    lake_extents_2025_updated.to_csv(
        "../data/lake_extent/lake_extent_estimates_2025.csv", index=False, columns=columns_order
    )


def plot_lake_extent_with_grayed_out_outliers(top_julian_day: int, top_year: int, bottom_julian_day: int, bottom_year: int, output_filename: str = 'lake_extent_overview.png'):
    """Create lake extent visualization, showing outliers in gray but not counting them."""
    fig = plt.figure(figsize=(17.5 / 2.54, 15 / 2.54), layout="constrained")
    
    axes = fig.subplot_mosaic(
        """
        AABBC
        AABBC
        DDDDD
        DDDDD
        DDDDD
        EEFFG
        EEFFG
        """
    )
    
    for ax_label in "ABCEFG":
        axes[ax_label].set_xticklabels([])
        axes[ax_label].set_yticklabels([])
        axes[ax_label].set_xticks([])
        axes[ax_label].set_yticks([])
        axes[ax_label].set_aspect('equal', adjustable='box')

    vis_img_path = pathlib.Path.cwd() / f"../data/lake_extent/visible_images_for_plot/{top_year}"
    load_and_prepare_visible_image(vis_img_path, top_julian_day, top_year, axes["A"])

    ir_img_path = pathlib.Path.cwd() / f"../data/lake_extent/ir_images_for_plot/{top_year}"
    load_and_prepare_ir_image(ir_img_path, top_julian_day, top_year, axes["B"])

    axes["C"].spines[:].set_visible(False)
    
    bottom_row_year = bottom_year
    bottom_row_day = bottom_julian_day
    
    vis_img_path_bottom = pathlib.Path.cwd() / f"../data/lake_extent/visible_images_for_plot/{bottom_row_year}"
    load_and_prepare_visible_image(vis_img_path_bottom, bottom_row_day, bottom_row_year, axes["E"])

    ir_img_path_bottom = pathlib.Path.cwd() / f"../data/lake_extent/ir_images_for_plot/{bottom_row_year}"
    load_and_prepare_ir_image(ir_img_path_bottom, bottom_row_day, bottom_row_year, axes["F"])

    axes["G"].spines[:].set_visible(False)

    date = datetime(top_year, 1, 1) + td(days=top_julian_day - 1)
    second_date = datetime(bottom_year, 1, 1) + td(days=bottom_julian_day - 1)
    
    try:
        lake_extents = prepare_lake_extent_data()
        outliers_df = lake_extents[
            lake_extents["Outlier"].astype(str).str.lower().str.startswith("true")
        ]
        non_outliers_df = lake_extents[
            lake_extents["Outlier"].astype(str).str.lower().str.startswith("false")
        ]
        non_outliers_df = non_outliers_df[
            pd.notna(non_outliers_df["Lake Area (sq meters)"])
        ]
        
        second_date_data = plot_lake_extent_timeseries(
            axes["D"], lake_extents, non_outliers_df, outliers_df, date, second_date
        )
        save_updated_data(lake_extents)

        max_y = axes['D'].get_ylim()[1]
        con = patches.ConnectionPatch(
            xyA=(1, 0.5),
            coordsA=axes['B'].transAxes,
            xyB=(mdates.date2num(date), max_y),
            coordsB=axes['D'].transData,
            arrowstyle="->",
            linestyle="-",
            color="black",
            linewidth=1,
            mutation_scale=15
        )
        fig.add_artist(con)
        
        if not second_date_data.empty:
            target_x = mdates.date2num(second_date)
            target_y = second_date_data["Lake Area (sq meters)"].iloc[0]
            
            source_display_coords = axes['F'].transAxes.transform((1, 0.5))
            target_display_coords = axes['D'].transData.transform((target_x, target_y))
            
            vector = target_display_coords - source_display_coords
            norm_vector = vector / np.linalg.norm(vector)
            
            gap_pixels = 5
            
            new_endpoint_display = target_display_coords - gap_pixels * norm_vector
            new_endpoint_data = axes['D'].transData.inverted().transform(new_endpoint_display)
            
            con2 = patches.ConnectionPatch(
                xyA=(1, 0.5),
                coordsA=axes['F'].transAxes,
                xyB=new_endpoint_data,
                coordsB=axes['D'].transData,
                arrowstyle="->",
                linestyle="-",
                color="black",
                linewidth=1,
                mutation_scale=15
            )
            fig.add_artist(con2)

    except FileNotFoundError as e:
        print(f"Error: Could not find lake extent data or image files: {e}")
    except Exception as e:
        print(f"An unexpected error occurred during plotting: {e}")

    try:
        fig.savefig(output_filename, dpi=500, bbox_inches='tight')
        print(f"Plot saved successfully to {os.path.abspath(output_filename)}")
    except Exception as e:
        print(f"Error saving plot: {e}")

    plt.close(fig)
    return fig, axes

if __name__ == "__main__":
    plot_lake_extent_with_grayed_out_outliers(
        top_julian_day=39,
        top_year=2025,
        bottom_julian_day=52,
        bottom_year=2025,
        output_filename='lake_extent_overview.png'
    )