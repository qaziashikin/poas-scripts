# Poas Volcano Infrared Image Analysis

This repository contains scripts for monitoring and analyzing volcanic activity at Poás Volcano, Costa Rica, using infrared (IR) and visible imagery from the Volcano Photo Monitoring Internet (VPMI) system. The project focuses on three main areas of interest:

1. **IR Image Classification** - Automated classification of IR images into categories such as plume presence, fumaroles, fences, and image quality assessment.
2. **Laguna Caliente Lake Extent Analysis** - Generation of lake extent masks, statistical analysis, and visualization of lake area changes over time.
3. **Figure Generation** - Scripts to create figures for manuscripts.

## Project Structure

```
poas-scripts/
├── classifiers/              # Image classification modules
│   ├── canny_feature_detector.py
│   ├── contour_feature_detector.py
│   ├── degraded_classifier.py
│   └── utils.py
├── configs/                  # Configuration files
│   └── config.toml
├── figures/                  # Figure generation scripts
│   ├── data/                 # Figure data and outputs
│   ├── figure2/
│   ├── figure3/
│   ├── figure4/
│   └── figure5/
├── scripts/                  # Main executable scripts
│   └── poas_ir_classifier.py
├── generate_lake_extent_masks.py
├── generate_lake_extent_stats.py
├── generate_plot.py
├── geometric_corrector.py
├── run_lake_extent_scripts.py
├── utilities.py
└── README.md
```

## Dependencies

- Python 3.8+
- OpenCV (cv2)
- NumPy
- Matplotlib
- Pandas
- SciPy
- scikit-learn
- Pillow (PIL)

## Data Requirements

The scripts expect data to be organized in the following structure:

```
/data/vulcand/archive/
├── imagery/
│   ├── infrared/345040/{year}/VPMI/still/{day:03}/*.jpg  # IR images
│   └── visible/345040/{year}/VPMI/still/{day:03}/*.jpg   # Visible images
└── products/
    └── plots/poas-lake-extent/{year}/
```

## Components

### 1. IR Image Classification

**Script:** `scripts/poas_ir_classifier.py`

Classifies IR images from the VPMI system into multiple categories using computer vision techniques:

- **Fence**: Detection of volcanic fence structures
- **Fumaroles**: Identification of fumarolic activity
- **Plume**: Volcanic plume detection
- **Low_Visibility**: Poor visibility conditions
- **Degraded**: Low-quality or corrupted images
- **Obscured**: Objects obscuring the view
- **Cloud cover**: Cloud interference

**Usage:**

```bash
python scripts/poas_ir_classifier.py <image_path> -c configs/config.toml
```

**Example Output:**

```
Assignments for Image 345040.VPMI.2024.274_041001-0000.jpg: {'Fence', 'Plume', 'Fumaroles'}
```

The classification uses:

- Canny edge detection for feature extraction
- Contour analysis for shape-based detection
- Configurable thresholds defined in `configs/config.toml`

### 2. Lake Extent Analysis

#### Lake Extent Mask Generation

**Script:** `generate_lake_extent_masks.py` or `run_lake_extent_scripts.py`

Generates binary masks representing lake extent from IR imagery using a voting-based approach:

1. Classifies IR images to filter out plume-containing and degraded images
2. Applies Gaussian blur and thresholding to detect white areas (lake surface)
3. Uses a voting system across multiple images per day to create robust masks
4. Dynamically adjusts thresholds based on plume frequency

**Usage:**

```bash
python generate_lake_extent_masks.py <year> <start_day> [<end_day>]
```

**Example:**

```bash
python generate_lake_extent_masks.py 2024 67
```

**Output:**

```
Overlaying white areas for lake_extent/images/2024/67_lake_extent, saving to outputs/lake_extent_masks/2024/67_lake_extent.png
Overlay created using a threshold of 183 votes
```

#### Lake Extent Statistics

**Script:** `generate_lake_extent_stats.py`

Computes statistics from generated masks:

- White pixel counts
- Real-world area calculations using geometric correction
- CSV output with date, pixel count, area, and outlier flags

**Usage:**

```bash
python generate_lake_extent_stats.py
```

**Output:** `outputs/data/lake_extent_estimates_{year}.csv`

#### Lake Extent Plot Generation

**Script:** `generate_plot.py`

Creates publication-quality plots showing:

- Lake extent time series
- IR and visible image overlays
- Outlier detection and graying
- Cubic spline interpolation

**Usage:** Typically called through `run_lake_extent_scripts.py`

### 3. Unified Lake Extent Pipeline

**Script:** `run_lake_extent_scripts.py`

Orchestrates the complete lake extent analysis pipeline:

1. Generates masks for specified date range
2. Computes statistics
3. Creates plots

**Usage:**

```bash
python run_lake_extent_scripts.py <year> <start_day> [<end_day>]
```

**Example:**

```bash
python run_lake_extent_scripts.py 2024 60 90
```

### 4. Geometric Correction

**Module:** `geometric_corrector.py`

Converts pixel-based measurements to real-world coordinates using:

- Known geographic control points
- Haversine distance calculations
- Local scaling based on k-nearest neighbors
- Accurate area calculations in square meters

### 5. Figure Builders

Located in `figures/figure[2-5]/build_figure.py`

Scripts for generating specific figures in research manuscripts, including:

- Annotated images
- Statistical plots
- Comparative visualizations

## Configuration

Classifier thresholds and parameters are defined in `configs/config.toml`:

- Canny edge detection parameters
- Contour detection settings
- Threshold values for each classifier
- Image preprocessing options

## Output Structure

```
outputs/
├── lake_extent_masks/{year}/     # Binary mask images
├── data/                         # CSV statistics and plume data
│   ├── lake_extent_estimates_{year}.csv
│   └── {year}_plume_percent.txt
└── plots/                        # Generated figures
```

## Notes

- Images are expected to be 640x480 resolution from VPMI system
- Lake extent detection relies on thermal contrast in IR imagery
- Geometric correction uses fixed control points specific to Poás crater
- Processing requires significant IR image archives for robust results
