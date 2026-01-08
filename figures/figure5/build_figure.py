"""
This script produces Figure 5 of the manuscript:



:author:
    Qazi T. Ashikin

"""

import argparse
import pathlib
from datetime import datetime as dt

import cv2
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline


plt.style.use("../basic.mplstyle")


def crop_image(image: np.ndarray, x: int, y: int, a: int, b: int) -> np.ndarray:
    """
    Crop an image to a specific region.

    The cropped image will be the area enclosed by the points:

    (x, y - b)                     (x + a, y - b)
             .<-------- a -------->.
             ^
             |
             b
             |
             v
             .                     .
        (x, y)                     (x + a, y)

    Parameters
    ----------
    x:
        The horizontal pixel coordinate of the bottom-left point in cropped image.
    y:
        The vertical pixel coordinate of the bottom-left point in cropped image.
    a:
        The width (number of columns) of the cropped region.
    b:
        The height (number of rows) of the cropped region.

    Returns
    -------
     :
        The cropped image.

    """

    height, width, *_ = image.shape

    top_left_x = max(0, x)
    top_left_y = max(0, y - b)
    bottom_right_x = min(width, x + a)
    bottom_right_y = min(height, y)

    cropped_img = image[top_left_y:bottom_right_y, top_left_x:bottom_right_x]

    return cropped_img


def resize_mask(mask: np.ndarray) -> np.ndarray:
    """Pad the mask to be the same dimensions as the base image (640 x 480)."""

    top, bottom, left, right = 220, 110, 0, 640 - 360

    padded_mask = cv2.copyMakeBorder(
        mask, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0]
    )

    return padded_mask


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


def detect_outliers_zscore(
    data: pd.Series, window_days: int = 11, z_threshold: float = 3.5
):
    """Detect outliers using a rolling Z-score."""

    data = pd.Series(data)

    window_days = window_days if window_days % 2 else window_days + 1

    outliers = np.zeros(len(data), dtype=bool)

    rolling_median = data.rolling(window=window_days, center=True).median()
    mad = (
        (data - rolling_median).abs().rolling(window=window_days, center=True).median()
    )

    # rolling_mean = data.rolling(window=window_days, center=True).mean()
    # rolling_std = data.rolling(window=window_days, center=True).std()
    # z_scores = np.abs((data - rolling_mean) / rolling_std)
    # outliers = z_scores > threshold

    robust_z = (data - rolling_median) / (1.4826 * mad.replace(0, np.nan))
    outliers = robust_z.abs() > z_threshold

    return outliers.fillna(False).values


def prepare_lake_extent_data():
    """Load and prepare lake extent data, evaluating True/False strings."""

    base_path = pathlib.Path.cwd().parent / "data/lake_extent"
    lake_extents = pd.concat(
        [
            pd.read_csv(base_path / f"lake_extent_estimates_{year}.csv")
            for year in [2024, 2025]
        ]
    ).reset_index(drop=True)
    lake_extents["Date"] = pd.to_datetime(lake_extents["Date"])
    lake_extents = lake_extents.sort_values(by=["Date"]).reset_index(drop=True)

    lake_extents["Lake Area (sq meters)"] = pd.to_numeric(
        lake_extents["Lake Area (sq meters)"], errors="coerce"
    )
    pixels = lake_extents["White Pixel Count"]  # .values

    # Choose which outlier detection method to use:
    # 1. Original method (comment out to use Z-score)
    # detected_outliers = detect_outliers(pixels)

    # 2. Z-score method (uncomment to use Z-score, comment out the original method call above)
    detected_outliers = detect_outliers_zscore(lake_extents["Lake Area (sq meters)"])

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


def plot_lake_extent_timeseries(ax, lake_extents, non_outliers_df, outliers_df, date):
    """
    Plot the lake extent timeseries with outliers marked and vertical lines for the
    selected dates.
    
    """

    non_outliers_df = non_outliers_df.copy()

    non_outliers_df["Date"] = pd.to_datetime(non_outliers_df["Date"])

    non_outliers_df.loc[:, "Lake Area (sq meters)"] = pd.to_numeric(
        non_outliers_df["Lake Area (sq meters)"], errors="coerce"
    )
    non_outliers_df = non_outliers_df.dropna(subset=["Lake Area (sq meters)"])

    non_outliers_df.loc[:, "Averages"] = non_outliers_df.rolling(
        "10D", on="Date", center=True
    ).mean(numeric_only=True)["Lake Area (sq meters)"]

    spline = CubicSpline(
        [
            pd.to_datetime(date_val).timestamp()
            for date_val in non_outliers_df["Date"].values
        ],
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
        color="#02818a",
        alpha=0.7,
        linewidth=1.5,
        label="Adjusted Running Average",
    )

    # Plot vertical line for the top date
    ax.axvline(date, color="#e7298a", linestyle="-", linewidth=1)

    ax.set_xlim([lake_extents["Date"].min(), lake_extents["Date"].max()])
    ax.set_ylim([0, lake_extents["Lake Area (sq meters)"].max() * 1.1])
    ax.set_ylabel("Lake Area (m$^2$)")
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")


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
        "../data/lake_extent/lake_extent_estimates_2024.csv",
        index=False,
        columns=columns_order,
    )
    lake_extents_2025_updated.to_csv(
        "../data/lake_extent/lake_extent_estimates_2025.csv",
        index=False,
        columns=columns_order,
    )


def get_image_for_date(date: dt, image_path: pathlib.Path) -> pathlib.Path:
    """Return the pathlib.Path to the image for a given date."""

    pattern = f"{date.year}/{date.timetuple().tm_yday:03d}_*.png"
    matches = list(image_path.glob(pattern))

    if not matches:
        raise FileNotFoundError(f"No image found for date {date.strftime('%Y-%j')}")
    if len(matches) > 1:
        print(f"Multiple matches found for {date.strftime('%Y-%j')}, using first.")

    return matches[0]


def add_visible_image(date: dt, ax: plt.Axes) -> None:
    """
    Add a visible image to a specified Axes object.

    The image is resampled, cropped, and shifted to align with the IR images.

    Parameters
    ----------
    date:
        Date of visible image to be visualised.
    ax:
        Axes on which to plot the image.

    """

    image_path = pathlib.Path.cwd().parent / "data/lake_extent/visible_images_for_plot"
    try:
        image_file = get_image_for_date(date, image_path)
    except FileNotFoundError:
        return

    img = cv2.imread(image_file, cv2.IMREAD_COLOR)
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
    crop = [0, abs(ty) + 17, 0, abs(tx) + 10]  # Top, Bottom, Left, Right

    ax.imshow(stretched_img[crop[0] : height - crop[1], crop[2] : width - crop[3]])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal", adjustable="box")


def add_infrared_image(date: dt, ax: plt.Axes) -> None:
    """
    Add an infrared image to a specified Axes object.

    The image is resampled, cropped, and shifted.

    Parameters
    ----------
    date:
        Date of IR image to be visualised.
    ax:
        Axes on which to plot the image.

    """

    image_path = pathlib.Path.cwd().parent / "data/lake_extent/ir_images_for_plot"
    try:
        image_file = get_image_for_date(date, image_path)
    except FileNotFoundError:
        return

    mask = (
        image_path.parent
        / f"results/{date.year}/{date.timetuple().tm_yday:03d}_lake_extent.png"
    )
    if not mask.exists():
        print(f"No mask found for {date}.")
        return

    img = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)
    base_img = cv2.cvtColor((img * 0.9).astype(np.uint8), cv2.COLOR_GRAY2BGR)

    mask = cv2.imread(mask, cv2.IMREAD_GRAYSCALE)
    mask = resize_mask(mask)
    masked_region = mask > 0

    coloured_mask = np.zeros_like(base_img)
    coloured_mask[:] = (127, 255, 212)

    base_img[masked_region] = (
        0.05 * base_img[masked_region] + 0.95 * coloured_mask[masked_region]
    ).astype(np.uint8)

    ax.imshow(base_img[12:435, :620])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal", adjustable="box")


def build_plot(date: dt) -> None:
    """
    Visualise the lake extent time series and show a visible/IR image pair for the
    requested date.

    Parameters
    ----------
    date:
        Date for which to show highest visibility visible/IR image pair.

    """

    fig = plt.figure(figsize=(17.5 / 2.54, 12.5 / 2.54), constrained_layout=True)

    axes = fig.subplot_mosaic(
        """
        AABB
        CCCC
        """
    )

    add_visible_image(date, axes["A"])
    add_infrared_image(date, axes["B"])

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

        # Plotting the timeseries with vertical lines for both dates
        plot_lake_extent_timeseries(
            axes["C"], lake_extents, non_outliers_df, outliers_df, date
        )
        save_updated_data(lake_extents)

    except FileNotFoundError as e:
        print(f"Error: Could not find lake extent data or image files: {e}")
    except Exception as e:
        print(f"An unexpected error occurred during plotting: {e}")

    try:
        fig.savefig("figure5.png", dpi=400)
        print(f"Plot saved successfully to figure5.png")
    except Exception as e:
        print(f"Error saving plot: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--date",
        help="Specify the date to visualise, in format YYYY-JJJ.",
        required=True,
    )
    args = parser.parse_args()

    try:
        date = dt.strptime(args.date, "%Y-%j")
        build_plot(date)
    except ValueError:
        print("Date must be in format 'YYYY-JJJ', e.g., 2025-030.")
