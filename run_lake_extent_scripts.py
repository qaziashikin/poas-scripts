import sys
from pathlib import Path
from generate_lake_extent_masks import LakeExtentMaskGenerator
from generate_lake_extent_stats import LakeExtentStatisticGenerator
from generate_plot import PlotGenerator

# Validate and parse command-line arguments
if len(sys.argv) < 3 or len(sys.argv) > 4:
    print("Usage: python run_lake_extent_scripts.py <year> <start_day> [<end_day>]")
    sys.exit(1)

year = int(sys.argv[1])
start_day = int(sys.argv[2])
end_day = int(sys.argv[3]) if len(sys.argv) == 4 else start_day

# Initialize generators
lake_extent_mask_generator = LakeExtentMaskGenerator()
lake_extent_stats_generator = LakeExtentStatisticGenerator()
plot_generator = PlotGenerator()

# Generate lake extent masks
for day in range(start_day, end_day + 1):
    ir_images_path = Path(f"/data/vulcand/archive/imagery/infrared/345040/{year}/VPMI/still") / f"{day:03}"
    lake_extent_mask_file_name = f"{day}_lake_extent.png"
    lake_extent_mask_path_name = f"lake_extent_masks/{year}"

    lake_extent_mask_generator.overlay_white_areas(
        ir_images_path, lake_extent_mask_path_name, lake_extent_mask_file_name, year
    )
print(f"Lake extent masks generated.")

# Generate lake extent statistics
root_output_dir = "outputs"
path_to_lake_extent_masks = f"{root_output_dir}/lake_extent_masks/{year}"
lake_extent_stats_generator.generate_lake_extent_statistics(root_output_dir, path_to_lake_extent_masks)
print(f"Lake extent statistics generated.")

# Generate plots for each day
for julian_day in range(start_day, end_day + 1):
    csv_path = Path(f"{root_output_dir}/data/lake_extent_estimates_{year}.csv")
    if not csv_path.exists():
        continue

    fig, axes = plot_generator.plot_lake_extent_with_grayed_out_outliers(
        path_to_csvs=f"{root_output_dir}/data",
        vis_imgs_for_plot_folder_path="/data/vulcand/archive/imagery/visible/345040",
        ir_imgs_for_plot_folder_path="/data/vulcand/archive/imagery/infrared/345040",
        julian_day=julian_day,
        year=year
    )
    
    final_plot_output_dir = Path(f"/data/vulcand/products/plots/poas-lake-extent/{year}")
    final_plot_output_dir.mkdir(parents=True, exist_ok=True)
    path = final_plot_output_dir / f"poas-lake-extent_{year}-{julian_day:03}.png"
    fig.savefig(path, dpi=400)
    print(f"Saved plot to {path}")