"""Intermediate results of the "thresh" segmenter for one example frame"""
import pathlib
import tempfile
import zipfile

import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage

from dcnum.read import HDF5Data
from dcnum.segm.segm_thresh import SegmentThresh

# example dataset shipped with the dcnum test suite
path_zip = pathlib.Path("../../tests/data/"
                        "fmt-hdf5_cytoshot_full-features_2024.zip")
idx = 4
with tempfile.TemporaryDirectory(prefix="dcnum_docs_") as tmp_dir:
    with zipfile.ZipFile(path_zip) as zf:
        path_rtdc = zf.extract("dcnum_thresh_example.rtdc", tmp_dir)
    with HDF5Data(path_rtdc) as hd:
        image = hd.image[idx]
        image_bg = hd.image_bg[idx]
        bg_off = hd["bg_off"][idx]

# background correction (as done in the pipeline)
image_corr = np.asarray(image, dtype=np.int16) - image_bg - bg_off
# thresholding (default threshold is -6)
mask_raw = SegmentThresh.segment_algorithm(image_corr[np.newaxis])[0]
# mask post-processing with the default settings of the "thresh" segmenter
mask = SegmentThresh.process_labels(mask_raw.copy(),
                                    **SegmentThresh.mask_default_kwargs) > 0
# gating: only objects with more than 10 pixels are events (default)
labels, _ = ndimage.label(mask)
sizes = np.bincount(labels.ravel())
sizes[0] = 0
mask = sizes[labels] > 10

panels = [
    ("1. raw image (\"image\")", image, "gray", (0, 255)),
    ("2. background image (\"image_bg\")", image_bg, "gray", (0, 255)),
    ("3. background-corrected image (image − image_bg − bg_off; "
     "red is darker than the background)",
     image_corr, "RdBu", (-40, 40)),
    ("4. thresholding (pixels darker than −6)", mask_raw, "gray_r",
     (0, 1)),
    ("5. mask post-processing and gating (event contour)", image, "gray",
     (0, 255)),
]

fig, axes = plt.subplots(len(panels), 1, figsize=(7, 8.2))
for ax, (title, data, cmap, (vmin, vmax)) in zip(axes, panels):
    ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="none")
    ax.set_title(title, fontsize=9, loc="left")
    ax.set_xticks([])
    ax.set_yticks([])
axes[-1].contour(mask, levels=[0.5], colors="#e66100", linewidths=1.5)

fig.tight_layout()
plt.show()
