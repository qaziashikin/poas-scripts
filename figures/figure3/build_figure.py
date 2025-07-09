import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import os

def generate_and_save_plot():
    fig, ax = plt.subplots(figsize=(8.5 / 2.54, 3))   
    ax.axis('off')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    fontsize = 7

    cont_w = 0.46
    space_between = 0.03
    total_width = 2 * cont_w + space_between

    par_x = (1 - total_width) / 2
    seq_x = par_x + cont_w + space_between

    box_h = 0.07

    seq_cont_y = 0.08
    seq_cont_h = 0.82

    par_ys_initial = [0.75, 0.60, 0.45]
    margin_int = 0.10

    ax.add_patch(Rectangle((seq_x, seq_cont_y), cont_w, seq_cont_h,
                           linewidth=1, edgecolor='black', facecolor='none'))
    ax.text(seq_x + cont_w/2, seq_cont_y + seq_cont_h + 0.02,
            'Run In-Sequence', ha='center', va='bottom',
            fontsize=fontsize, fontweight='bold')
    ax.text(seq_x + 0.01, seq_cont_y + seq_cont_h - 0.02,
            'b', ha='left', va='top',
            fontsize=fontsize, fontweight='bold')

    narrow_w = (cont_w - 0.04) / 3
    wide_w = cont_w - 0.08

    seq_labels = [
        'Plume Classifier', 'Low Visibility Classifier',
        'Obscured Classifier', 'Cloud Cover Classifier', 'End'
    ]
    seq_ys = [0.75, 0.60, 0.45, 0.30, 0.15]

    for y, label in zip(seq_ys, seq_labels):
        w = narrow_w if label == 'End' else wide_w
        x = seq_x + (cont_w - w) / 2
        ax.add_patch(Rectangle((x, y), w, box_h,
                               linewidth=1, edgecolor='black', facecolor='white'))
        ax.text(x + w/2, y + box_h/2, label,
                ha='center', va='center', fontsize=fontsize)

    for y0, y1 in zip(seq_ys, seq_ys[1:]):
        ax.annotate('', xy=(seq_x + cont_w/2, y1 + box_h),
                    xytext=(seq_x + cont_w/2, y0),
                    arrowprops=dict(arrowstyle='->', lw=1))

    top_orig = max(par_ys_initial) + box_h + margin_int
    bottom_orig = min(par_ys_initial) - margin_int
    height_par = top_orig - bottom_orig - 0.03

    env_bottom = 0.5 - height_par / 2
    env_top = env_bottom + height_par

    ax.add_patch(Rectangle((par_x, env_bottom), cont_w, height_par,
                           linewidth=1, edgecolor='black', facecolor='none'))
    ax.text(par_x + cont_w/2, env_top + 0.02,
            'Run In-Parallel', ha='center', va='bottom',
            fontsize=fontsize, fontweight='bold')
    ax.text(par_x + 0.01, env_top - 0.02,
            'a', ha='left', va='top',
            fontsize=fontsize, fontweight='bold')

    inner_w = cont_w - 0.10
    right_center = par_x + (cont_w - inner_w) / 2

    y_offset = env_bottom - bottom_orig
    new_par_ys = [y + y_offset for y in par_ys_initial]

    par_labels = [
        'Fence Detection', 'Fumarole Detection', 'Degraded Classifier'
    ]

    for y, label in zip(new_par_ys, par_labels):
        ax.add_patch(Rectangle((right_center, y), inner_w, box_h,
                               linewidth=1, edgecolor='black', facecolor='white'))
        ax.text(right_center + inner_w/2, y + box_h/2,
                label, ha='center', va='center', fontsize=fontsize)

    plt.tight_layout()

    output_path = 'classification-workflow-overview.png'

    try:
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved successfully to {os.path.abspath(output_path)}")
    except Exception as e:
        print(f"Error saving plot: {e}")

    plt.close(fig)

if __name__ == "__main__":
    generate_and_save_plot()