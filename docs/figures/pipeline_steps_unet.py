"""Input and output of a U-Net segmenter for example frames

The masks were computed with the U-Net test model of the dcnum test suite
(``segm-torch-model_unet-dcnum-test_g1_910c2.zip``) and are stored next to
the images in ``segm-torch-test-data_unet-dcnum-test_g1_910c2.zip``, so
building the documentation does not require PyTorch.
"""
import pathlib
import zipfile

import cv2
import matplotlib.pyplot as plt
import numpy as np

from dcnum.segm.segm_torch import SegmentTorchMPO

path_zip = pathlib.Path("../../tests/data/"
                        "segm-torch-test-data_unet-dcnum-test_g1_910c2.zip")


def read_png(zf, name):
    data = np.frombuffer(zf.read(name), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)


stems = ["cell_image_1", "cell_image_3"]

fig, axes = plt.subplots(len(stems), 3, figsize=(9, 2.2))
with zipfile.ZipFile(path_zip) as zf:
    for row, stem in zip(axes, stems):
        image = read_png(zf, f"{stem}.png")
        # binary U-Net output (probability >= 0.5)
        mask_unet = read_png(zf, f"{stem}_mask_segm.png") > 0
        # mask post-processing with the defaults of the torch segmenters
        mask = SegmentTorchMPO.process_labels(
            mask_unet.copy(), **SegmentTorchMPO.mask_default_kwargs) > 0

        row[0].imshow(image, cmap="gray", vmin=0, vmax=255)
        row[1].imshow(mask_unet, cmap="gray_r", vmin=0, vmax=1)
        row[2].imshow(image, cmap="gray", vmin=0, vmax=255)
        row[2].contour(mask, levels=[0.5], colors="#1a7f37",
                       linewidths=1.5)

titles = ["1. raw image (model input)",
          "2. U-Net output (binary mask)",
          "3. post-processed mask and contour"]
for ax, title in zip(axes[0], titles):
    ax.set_title(title, fontsize=9, loc="left")
for ax in axes.flat:
    ax.set_xticks([])
    ax.set_yticks([])

fig.tight_layout()
plt.show()
