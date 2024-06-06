from typing import Tuple

from ..segmenter import Segmenter

import numpy as np
from scipy import ndimage as ndi


def postprocess_masks(masks,
                      original_image_shape: Tuple[int, int]):
    """Postprocess mask images from ML segmenters

    The transformation includes:
    - Revert the cropping and padding operations done in
      :func:`.preprocess_images` by padding with zeros and cropping.
    - If the original image shape is larger than the mask image shape,
      also clear borders in an intermediate step
      (maks postprocessing using :func:`Segmenter.process_mask`).

    Parameters
    ----------
    masks: 3d or 4d ndarray
        Mask data in shape (batch_size, 1, imagex_size, imagey_size)
        or (batch_size, imagex_size, imagey_size).
    original_image_shape: tuple of (int, int)
        The required output mask shape for one event. This required for
        doing the inverse of what is done in :func:`.preprocess_images`.

    Returns
    -------
    labels_proc: np.ndarray
        An integer array with the same dimensions as the original image
        data passed to :func:`.preprocess_images`. The shape of this array
        is (batch_size, original_image_shape[0], original_image_shape[1]).
    """
    # If output of model is 4d, remove channel axis
    if len(masks.shape) == 4:
        masks = masks[:, 0, :, :]

    # Label the mask image
    labels = np.empty(masks.shape, dtype=np.uint16)
    label_struct = ndi.generate_binary_structure(2, 2)
    for ii in range(masks.shape[0]):
        ndi.label(
            input=masks[ii],
            output=labels[ii],
            structure=label_struct)

    batch_size = labels.shape[0]

    # Revert padding and cropping from preprocessing
    mask_shape_ret = labels.shape[1:]
    # height
    s0diff = original_image_shape[0] - mask_shape_ret[0]
    s0t = abs(s0diff) // 2
    s0b = abs(s0diff) - s0t
    # width
    s1diff = original_image_shape[1] - mask_shape_ret[1]
    s1l = abs(s1diff) // 2
    s1r = abs(s1diff) - s1l

    if s0diff > 0 or s1diff > 0:
        # The masks that we have must be padded. Before we do that, we have
        # to remove events on the edges, otherwise we will have half-segmented
        # cell events in the output array.
        for ii in range(batch_size):
            labels[ii] = Segmenter.process_mask(labels[ii],
                                                clear_border=True,
                                                fill_holes=False,
                                                closing_disk=0)

    # Crop first, only then pad.
    if s1diff > 0:
        labels_pad = np.zeros((batch_size,
                              labels.shape[1],
                              original_image_shape[1]),
                              dtype=np.uint16)
        labels_pad[:, :, s1l:-s1r] = labels
        labels = labels_pad
    elif s1diff < 0:
        labels = labels[:, :, s1l:-s1r]

    if s0diff > 0:
        labels_pad = np.zeros((batch_size,
                              original_image_shape[0],
                              original_image_shape[1]),
                              dtype=np.uint16)
        labels_pad[:, s0t:-s0b, :] = labels
        labels = labels_pad
    elif s0diff < 0:
        labels = labels[:, s0t:-s0b, :]

    return labels
