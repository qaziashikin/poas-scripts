"""
This script produces Figure 4 of the manuscript:



:author:
    Qazi T. Ashikin

"""

import pathlib

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont


BASEPATH = pathlib.Path.cwd()
AUXILIARIES_PATH = BASEPATH.parent / "data/auxiliaries/figure4"
FONT_DIR = BASEPATH.parent / "data" / "fonts"


def annotate_image(
    image_path: pathlib.Path,
    annotations_data: list[dict],
    font_path: str = None,
) -> Image.Image | None:
    """
    Adds annotations to the images, corresponding to the regions of interest in the
    analysis and the resultant label.

    Parameters
    ----------
    image_path:
        Path to image file to annotate.
    annotations_data:
        List of dictionaries containing annotation information.
    font_path:
        Path to font to use for annotations.

    """

    try:
        img = Image.open(image_path).convert("RGB")
    except FileNotFoundError:
        print(f"Error: Could not open or find the image at {image_path}")
        return None
    except Exception as e:
        print(f"Error opening image: {e}")
        return None

    draw = ImageDraw.Draw(img)

    try:
        label_font_size = 16
        if font_path:
            label_font = ImageFont.truetype(font_path, label_font_size)
        else:
            # Try to load from project fonts directory first
            font_file = FONT_DIR / "helvetica-bold.ttf"
            if font_file.exists():
                label_font = ImageFont.truetype(str(font_file), label_font_size)
            else:
                # Fall back to system Helvetica
                label_font = ImageFont.truetype("Helvetica-Bold", label_font_size)
    except IOError:
        print(
            f"Warning: Could not load Helvetica-Bold font from {FONT_DIR} or system. Using default Pillow font."
        )
        label_font = ImageFont.load_default()
        label_font_size = 10

    text_color = (255, 255, 255)
    box_thickness = 3

    for annotation in annotations_data:
        x_start, y_start, x_end, y_end = annotation["coords"]
        label = annotation["label"]
        box_color = annotation["color"]

        x1 = min(x_start, x_end)
        y1 = min(y_start, y_end)
        x2 = max(x_start, x_end)
        y2 = max(y_end, y_start)

        draw.rectangle([(x1, y1), (x2, y2)], outline=box_color, width=box_thickness)

        text_bbox = draw.textbbox((0, 0), label, font=label_font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]

        text_x = x1 + 5
        text_y = y1 - text_height - 7

        if text_y < 0:
            text_y = y1 + 5

        draw.text((text_x, text_y), label, font=label_font, fill=text_color)

    border_color = (0, 0, 0)
    border_thickness = 3
    img_width, img_height = img.size
    draw.rectangle(
        [(0, 0), (img_width - 1, img_height - 1)],
        outline=border_color,
        width=border_thickness,
    )

    output_path = image_path.parent / f"{image_path.stem}_annotated.png"
    try:
        img.save(output_path)
        print(f"Annotated image saved to {output_path}")
        return img
    except Exception as e:
        print(f"Error saving image: {e}")
        return None


def main():
    # Image 1 - good visibility
    img1 = AUXILIARIES_PATH / "example1.png"
    img1_annotations = [
        {"label": "Plume", "coords": (100, 370, 340, 250), "color": (0, 200, 0)},
        {"label": "Fumaroles", "coords": (540, 130, 635, 200), "color": (0, 200, 0)},
        {"label": "Fence", "coords": (10, 470, 630, 380), "color": (0, 200, 0)},
    ]
    img1_pil = annotate_image(img1, img1_annotations)

    # Image 2 - bad visibility
    img2 = AUXILIARIES_PATH / "example2.png"
    img2_annotations = [
        {"label": "Plume", "coords": (100, 370, 340, 250), "color": (230, 0, 0)},
        {"label": "Fumaroles", "coords": (540, 130, 635, 200), "color": (230, 0, 0)},
        {"label": "Fence", "coords": (10, 470, 630, 380), "color": (0, 200, 0)},
    ]
    img2_pil = annotate_image(img2, img2_annotations)

    fig, axes = plt.subplots(
        1, 2, figsize=(17.5 / 2.54, 6.7 / 2.54), constrained_layout=True
    )

    for ax, img, label in zip(axes, [img1_pil, img2_pil], "ab"):
        ax.imshow(np.asarray(img))
        ax.axis("off")
        ax.text(
            0.02,
            0.95,
            label,
            ha="left",
            va="center",
            transform=ax.transAxes,
            fontweight="bold",
            c="white",
        )

    final_output_path = BASEPATH / "classification-examples.png"
    fig.savefig(final_output_path, dpi=600)
    print(f"Combined figure saved to {final_output_path}")


if __name__ == "__main__":
    main()
