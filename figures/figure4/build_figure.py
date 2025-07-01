from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import numpy as np
import os

def annotate_image(image_path, output_path, append_letter, annotations_data, font_path="data/fonts/helvetica.ttf"):
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
        label_font = ImageFont.truetype(font_path, label_font_size)
    except IOError:
        print(f"Warning: Could not load font from {font_path}. Using default Pillow font.")
        label_font = ImageFont.load_default()
        label_font_size = 10 

    try:
        append_letter_font_size = 18
        append_letter_font = ImageFont.truetype(font_path, append_letter_font_size)
    except IOError:
        append_letter_font = ImageFont.load_default()
        append_letter_font_size = 10

    text_color = (255, 255, 255)
    box_thickness = 1

    for annotation in annotations_data:
        x_start, y_start, x_end, y_end = annotation["coords"]
        label = annotation["label"]
        box_color = annotation["color"]

        x1 = min(x_start, x_end)
        y1 = min(y_start, y_end)
        x2 = max(x_start, x_end)
        y2 = max(y_end, y_start)

        draw.rectangle([(x1, y1), (x2, y2)], outline=box_color, width=box_thickness)

        text_bbox = draw.textbbox((0,0), label, font=label_font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]

        text_x = x1 + 10
        text_y = y1 - text_height - 7 

        if text_y < 0:
            text_y = y1 + 5

        draw.text((text_x, text_y), label, font=label_font, fill=text_color)

    append_letter_text_bbox = draw.textbbox((0,0), append_letter, font=append_letter_font)
    append_letter_text_width = append_letter_text_bbox[2] - append_letter_text_bbox[0]
    append_letter_text_height = append_letter_text_bbox[3] - append_letter_text_bbox[1]

    append_letter_x = img.width - append_letter_text_width - 12
    append_letter_y = append_letter_text_height + 5
    draw.text((append_letter_x, append_letter_y), append_letter, font=append_letter_font, fill=(255, 255, 255))
    
    border_color = (0, 0, 0) 
    border_thickness = 2
    img_width, img_height = img.size
    draw.rectangle([(0, 0), (img_width - 1, img_height - 1)], outline=border_color, width=border_thickness)

    try:
        img.save(output_path)
        print(f"Annotated image saved to {output_path}")
        return img
    except Exception as e:
        print(f"Error saving image: {e}")
        return None

def main():
    good_vis_img = "../data/auxiliaries/figure4/example1.png"
    bad_vis_img = "../data/auxiliaries/figure4/example2.png"

    annotations_good_visibility = [
        {"label": "Plume", "coords": (100, 370, 340, 250), "color": (0, 200, 0)},
        {"label": "Fumaroles", "coords": (540, 130, 635, 200), "color": (0, 200, 0)},
        {"label": "Fence", "coords": (10, 470, 630, 380), "color": (0, 200, 0)}
    ]

    annotations_poor_visibility = [
        {"label": "Plume", "coords": (100, 370, 340, 250), "color": (230, 0, 0)},
        {"label": "Fumaroles", "coords": (540, 130, 635, 200), "color": (230, 0, 0)},
        {"label": "Fence", "coords": (10, 470, 630, 380), "color": (0, 200, 0)}
    ]

    output_good_img = "../data/auxiliaries/figure4/example1_annotated.png"
    output_bad_img = "../data/auxiliaries/figure4/example2_annotated.png"

    img1_pil = annotate_image(good_vis_img, output_good_img, "a", annotations_good_visibility)
    img2_pil = annotate_image(bad_vis_img, output_bad_img, "b", annotations_poor_visibility)

    if img1_pil and img2_pil:
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))

        axes[0].imshow(np.array(img1_pil))
        axes[0].axis('off')

        axes[1].imshow(np.array(img2_pil))
        axes[1].axis('off')

        plt.tight_layout()
        final_output_path = "classification-examples.png"
        plt.savefig(final_output_path, dpi=600, bbox_inches='tight')
        plt.close(fig)
        print(f"Combined figure saved to {os.path.abspath(final_output_path)}")
    else:
        print("Could not generate one or both annotated images. Skipping combined figure creation.")

if __name__ == "__main__":
    main()
