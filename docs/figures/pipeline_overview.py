"""Schematic overview of the dcnum pipeline"""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

COLOR_DATA = "#dfe7f2"
COLOR_STEP = "#f4f4f4"
COLOR_BG = "#fbe3c8"
COLOR_THRESH = "#fbe3c8"
COLOR_UNET = "#d5ecd4"
EDGE = "#444444"


def box(ax, xc, yc, text, w=34, h=7.5, fc=COLOR_STEP, bold=False):
    patch = FancyBboxPatch((xc - w / 2, yc - h / 2), w, h,
                           boxstyle="round,pad=0.4,rounding_size=1.2",
                           fc=fc, ec=EDGE, lw=1)
    ax.add_patch(patch)
    ax.text(xc, yc, text, ha="center", va="center", fontsize=8.5,
            fontweight="bold" if bold else "normal", linespacing=1.3)


def arrow(ax, xy_from, xy_to, label=None, ls="-", color=EDGE,
          label_pos=0.5, label_kw=None, connectionstyle="arc3"):
    ax.add_patch(FancyArrowPatch(xy_from, xy_to, arrowstyle="-|>",
                                 mutation_scale=11, lw=1.1, color=color,
                                 ls=ls, connectionstyle=connectionstyle,
                                 shrinkA=0, shrinkB=0))
    if label:
        x = xy_from[0] + (xy_to[0] - xy_from[0]) * label_pos
        y = xy_from[1] + (xy_to[1] - xy_from[1]) * label_pos
        kw = {"fontsize": 7.5, "ha": "left", "va": "center",
              "color": color, "style": "italic"}
        kw.update(label_kw or {})
        x = kw.pop("x", x)
        ax.text(x, y, label, **kw)


def polyline(ax, points, ls="-", color=EDGE):
    """Draw a line through `points` with an arrow head at the end"""
    xs, ys = zip(*points[:-1])
    ax.plot(xs, ys, ls=ls, color=color, lw=1.1)
    arrow(ax, points[-2], points[-1], ls=ls, color=color)


fig, ax = plt.subplots(figsize=(7.5, 9.6))
ax.set_xlim(0, 100)
ax.set_ylim(-8, 104)
ax.axis("off")

# segmentation area (two alternative lanes)
ax.add_patch(FancyBboxPatch((9, 39), 89, 38,
                            boxstyle="round,pad=0,rounding_size=2",
                            fc="none", ec="#999999", ls="--", lw=1))
ax.text(96, 74.5, "Segmentation (one segmenter per job)", fontsize=8.5,
        color="#555555", fontweight="bold", va="center", ha="right")
ax.text(30, 41, "segmenter_code = \"thresh\"", fontsize=8,
        ha="center", family="monospace", color="#8a4b00")
ax.text(75, 40.6, "\"torchmpo\", \"torchsto\", \"torchuni\"",
        fontsize=8, ha="center", family="monospace", color="#2c6b2a")
ax.text(75, 43, "no background correction required", ha="center",
        va="center", fontsize=8, color="#2c6b2a", fontweight="bold")

# boxes
box(ax, 50, 98, "Input .rtdc file\n(raw \"image\" data and metadata)",
    w=46, fc=COLOR_DATA, bold=True)
box(ax, 30, 84, "Background computation\n(\"sparsemed\", \"rollmed\", "
    "\"copy\")\n→ \"image_bg\", \"bg_off\"", w=32, h=9.5, fc=COLOR_BG)
box(ax, 30, 61, "Background correction\nimage − image_bg − bg_off",
    w=32, fc=COLOR_THRESH)
box(ax, 30, 49, "Thresholding\n(e.g. darker than −6) → mask",
    w=32, fc=COLOR_THRESH)
box(ax, 75, 61, "Pre-processing of the raw image\n(normalization, "
    "crop/pad)", w=36, fc=COLOR_UNET)
box(ax, 75, 49, "U-Net inference (PyTorch)\nprobability ≥ 0.5 → mask",
    w=36, fc=COLOR_UNET)
box(ax, 50, 30, "Labeling and mask post-processing\n(clear border, "
    "fill holes, smoothing)", w=46)
box(ax, 50, 17.5, "Feature extraction per event\n(contour & moments, "
    "brightness, texture, volume)", w=46)
box(ax, 50, 5, "Gating\n(minimum mask size, optional online gates)", w=46)
box(ax, 50, -4, "Output .rtdc file (features, PPID, logs, basins)",
    w=46, h=5, fc=COLOR_DATA, bold=True)

# input → background computation
arrow(ax, (38, 93.9), (32, 89.1))
# input → U-Net lane (raw image, no background data involved)
arrow(ax, (62, 93.9), (75, 65.1), label="raw image", label_pos=0.45)
# input → background correction (image)
polyline(ax, [(26.6, 98), (4, 98), (4, 61), (13.6, 61)])
ax.text(3, 92, "image", fontsize=7.5, color=EDGE, style="italic",
        ha="right", rotation=90, va="top")
# background → background correction
arrow(ax, (30, 78.9), (30, 65.1), color="#b25f00",
      label="image_bg, bg_off", label_pos=0.45, label_kw={"x": 31})
# within lanes
arrow(ax, (30, 56.9), (30, 53.1))
arrow(ax, (75, 56.9), (75, 53.1))
# lanes → labeling
arrow(ax, (30, 39), (38, 34.1))
arrow(ax, (75, 39), (62, 34.1))
# downstream steps
arrow(ax, (50, 25.9), (50, 21.6), label="labels (one per event)",
      label_kw={"x": 51.5})
arrow(ax, (50, 13.4), (50, 9.1))
arrow(ax, (50, 0.9), (50, -1.1))
# background data are also used for feature extraction
polyline(ax, [(13.6, 82), (7, 82), (7, 17.5), (26.6, 17.5)], ls="--",
         color="#b25f00")
ax.text(8, 24, "image_bg, bg_off\n(corrected brightness\nand texture)",
        fontsize=7, color="#b25f00", style="italic", va="center",
        ha="left")

fig.tight_layout()
plt.show()
