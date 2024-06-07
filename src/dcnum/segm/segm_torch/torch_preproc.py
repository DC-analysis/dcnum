from typing import Tuple

import numpy as np


def preprocess_images(images: np.ndarray,
                      norm_mean: float | None,
                      norm_std: float | None,
                      image_shape: Tuple[int, int] = None,
                      ):
    """Transform image data to something torch models expect

    The transformation includes:
    - normalization (division by 255, subtraction of mean, division by std)
    - cropping and padding of the input images to `image_shape`. For padding,
      the median of each *individual* image is used.
    - casting the input images to four dimensions
      (batch_size, 1, height, width) where the second axis is "channels"

    Parameters
    ----------
    images:
        Input image array (batch_size, height_in, width_in). If this is a
        2D image, it will be reshaped to a 3D image with a batch_size of 1.
    norm_mean:
        Mean value used for standard score data normalization, i.e.
        `normalized = `(images / 255 - norm_mean) / norm_std`; Set
        to None to disable normalization.
    norm_std:
        Standard deviation used for standard score data normalization;
        Set to None to disable normalization (see above).
    image_shape
        Image shape for which the model was created (height, width).
        If the image shape does not match the input image shape, then
        the input images are padded/cropped to fit the image shape of
        the model.

    Returns
    -------
    image_proc:
        3D array with preprocessed image data of shape
        (batch_size, 1, height, width)
    """
    if len(images.shape) == 2:
        # Insert indexing axis (batch dimension)
        images = images[np.newaxis, :, :]

    batch_size = images.shape[0]

    # crop and pad the images based on what the model expects
    image_shape_act = images.shape[1:]
    if image_shape is None:
        # model fits perfectly to input data
        image_shape = image_shape_act

    # height
    hdiff = image_shape_act[0] - image_shape[0]
    ht = abs(hdiff) // 2
    hb = abs(hdiff) - ht
    # width
    wdiff = image_shape_act[1] - image_shape[1]
    wl = abs(wdiff) // 2
    wr = abs(wdiff) - wl
    # helper variables
    wpad = wdiff < 0
    wcrp = wdiff > 0
    hpad = hdiff < 0
    hcrp = hdiff > 0

    # The easy part is the cropping
    if hcrp or wcrp:
        # define slices for width and height
        slice_hc = slice(ht, -hb) if hcrp else slice(None, None)
        slice_wc = slice(wl, -wr) if wcrp else slice(None, None)
        img_proc = images[:, slice_hc, slice_wc]
    else:
        img_proc = images

    # The hard part is the padding
    if hpad or wpad:
        # compute median for each original input image
        img_med = np.median(images, axis=(1, 2))
        # broadcast the median array from 1D to 3D
        img_med = img_med[:, None, None]

        # define slices for width and height
        slice_hp = slice(ht, -hb) if hpad else slice(None, None)
        slice_wp = slice(wl, -wr) if wpad else slice(None, None)

        # empty padding image stack with the shape required for the model
        img_pad = np.empty(shape=(batch_size, image_shape[0], image_shape[1]),
                           dtype=np.float32)
        # fill in original data
        img_pad[:, slice_hp, slice_wp] = img_proc
        # fill in background data for height
        if hpad:
            img_pad[:, :ht, :] = img_med
            img_pad[:, -hb:, :] = img_med
        # fill in background data for width
        if wpad:
            img_pad[:, :, :wl] = img_med
            img_pad[:, :, -wr:] = img_med
        # Replace img_norm
        img_proc = img_pad

    if norm_mean is None or norm_std is None:
        # convert to float32
        img_norm = img_proc.astype(np.float32)
    else:
        # normalize images
        img_norm = (img_proc.astype(np.float32) / 255 - norm_mean) / norm_std

    # Add a "channels" axis for the ML models.
    return img_norm[:, np.newaxis, :, :]
