"""Write chopped image data ("h5datasetchop" internal basins)

Chopped image data store only the region of each event image that is
related to the event. The format is the same as the one written by
``dclab-compress --crop-event-images`` (see :mod:`dcnum.read.chopped`
for a description). The algorithms in this module (event geometry,
clustering of the event images by shape, and cropping) are the same
as in dclab, such that dcnum and dclab write identical data.
"""
from __future__ import annotations

import json
import hashlib
from typing import Literal
import warnings

import numpy as np

from ..common import LazyLoader, h5py

cv2 = LazyLoader("cv2")


#: Chunk size of the datasets in bytes (as in dclab)
CHUNK_SIZE_BYTES = 1024 ** 2


class NegativeFrameCountWarning(UserWarning):
    """Used when the "frame" feature is not monotonically increasing"""


def disk(radius: int) -> np.ndarray:
    """Return a flat, disk-shaped footprint for binary dilation

    A pixel is within the neighborhood if the Euclidean distance between
    it and the origin is no greater than `radius` (as in
    ``skimage.morphology.disk``).
    """
    ll = np.arange(-radius, radius + 1)
    xx, yy = np.meshgrid(ll, ll)
    return (xx ** 2 + yy ** 2) <= radius ** 2


def get_chunk_size(item_shape: tuple[int, ...],
                   item_dtype: np.dtype | type) -> int:
    """Return the number of events in one chunk of chopped data

    The chunk size is approximately :const:`CHUNK_SIZE_BYTES`, but it
    contains at least 10 events (as in dclab).
    """
    event_size = np.prod(item_shape) * np.dtype(item_dtype).itemsize
    return max(10, int(np.floor(CHUNK_SIZE_BYTES / event_size)))


def get_event_shapes(size_x: np.ndarray,
                     size_y: np.ndarray,
                     pixel_size: float,
                     roi_shape: tuple[int, int],
                     pad_um: float = 1.5,
                     ) -> np.ndarray:
    """Return the shapes of the chopped event images

    Parameters
    ----------
    size_x, size_y:
        the "size_x" and "size_y" features [µm]
    pixel_size:
        pixel size [µm]
    roi_shape:
        shape of the full frames
    pad_um:
        padding added to each side of the event [µm]

    Returns
    -------
    shapes:
        2D array with the size along y (first column) and along x
        (second column) of the chopped event images
    """
    shapes = np.zeros((len(size_x), 2), dtype=np.uint16)
    # Use np.ceil to avoid cropping too much and np.clip to keep the
    # shapes within the ROI.
    for col, size, max_size in [(0, size_y, roi_shape[0]),
                                (1, size_x, roi_shape[1])]:
        shapes[:, col] = np.clip(
            np.asarray(np.ceil((np.asarray(size) + 2 * pad_um) / pixel_size),
                       dtype=np.uint64),
            a_min=1,
            a_max=max_size)
    return shapes


def get_event_offsets(mask: np.ndarray,
                      shapes: np.ndarray,
                      pixel_size: float,
                      pad_um: float = 1.5,
                      ) -> np.ndarray:
    """Return the offsets of the chopped event images in the frames

    Parameters
    ----------
    mask:
        3D boolean array of event masks
    shapes:
        the shapes of the chopped event images for `mask` (see
        :func:`get_event_shapes`)
    pixel_size:
        pixel size [µm]
    pad_um:
        padding added to each side of the event [µm]

    Returns
    -------
    offsets:
        2D array with the offset along y (first column) and along x
        (second column) of the chopped event images
    """
    shape_y, shape_x = mask.shape[1:]
    offsets = np.zeros((len(mask), 2), dtype=np.uint16)
    pad_px = np.round(pad_um / pixel_size)
    # first non-zero value along y
    start_y = np.argmax(np.count_nonzero(mask, axis=2) > 0, axis=1)
    offsets[:, 0] = np.clip(start_y - pad_px, a_min=0, a_max=shape_y)
    # first non-zero value along x
    start_x = np.argmax(np.count_nonzero(mask, axis=1) > 0, axis=1)
    offsets[:, 1] = np.clip(start_x - pad_px, a_min=0, a_max=shape_x)
    # Move events that would exceed the frame
    too_high = offsets[:, 0] + shapes[:, 0] > shape_y
    offsets[too_high, 0] = shape_y - shapes[too_high, 0]
    too_wide = offsets[:, 1] + shapes[:, 1] > shape_x
    offsets[too_wide, 1] = shape_x - shapes[too_wide, 1]
    return offsets


def get_clusters(shapes: np.ndarray,
                 dtype: np.dtype | type,
                 ) -> tuple[np.ndarray, int]:
    """Assign the events to datasets in which they are stored

    Events with similar shapes are stored in one dataset (whose image
    shape is the maximum shape of its events). Shapes are joined until
    there are enough events for one chunk in a dataset.

    Parameters
    ----------
    shapes:
        the shapes of the chopped event images
        (see :func:`get_event_shapes`)
    dtype:
        data type of the chopped images

    Returns
    -------
    clusters:
        cluster label (starting at 1) for each event
    num_clusters:
        number of clusters
    """
    clusters = np.zeros(len(shapes), dtype=np.uint64)
    indices = np.arange(len(clusters))
    cluster_id = 1

    # create a copy of the shapes (we will modify this array)
    shapes = np.array(shapes, copy=True)

    join_shapes = False
    join_x = False

    while not np.all(clusters):
        # indices that are not assigned to a cluster yet
        indices_todo = indices[clusters == 0]
        # shapes that need assignment
        shapes_todo = shapes[indices_todo]

        # identify unique shapes
        if join_x:
            shapes_to_unique = shapes_todo[:, ::-1]
        else:
            shapes_to_unique = shapes_todo

        uniques, inverse, counts = np.unique(shapes_to_unique,
                                             axis=0,
                                             return_counts=True,
                                             return_inverse=True,
                                             )
        # numpy 2.0.x returns a 2D inverse for `axis=0`
        inverse = inverse.reshape(-1)

        if join_x:
            uniques = uniques[:, ::-1]

        if np.sum(counts) < 100 or len(uniques) == 1:
            clusters[indices_todo] = cluster_id
            cluster_id += 1
            break

        if join_shapes:
            # join adjacent shapes
            join_x = not join_x

            # extend for odd number of shapes
            if len(counts) % 2:
                counts = np.pad(counts, (0, 1))
                uniques = np.pad(uniques, ((0, 1), (0, 0)))
                uniques[-1] = uniques[-2]

            # update shapes
            for ii in range(0, len(uniques), 2):
                shape_match = indices_todo[inverse == ii]
                shapes[shape_match] = uniques[ii + 1]

            # combine adjacent shapes
            uniques = uniques[1::2]
            inverse = inverse // 2
            counts = np.sum(counts.reshape(-1, 2), axis=1)

        for idx, (shape, size) in enumerate(zip(uniques, counts)):
            if size >= get_chunk_size(item_shape=shape, item_dtype=dtype):
                # Assign a common cluster to these shapes
                clusters[indices_todo[inverse == idx]] = cluster_id
                cluster_id += 1

        join_shapes = True

    return clusters, cluster_id - 1


def get_sibling_links(frame: np.ndarray) -> np.ndarray:
    """Return the links to the other events in the same frame

    Events in the same frame are linked in a circle (each event links
    to the previous event in the frame, the first event links to the
    last event). Events without siblings link to themselves.
    """
    links = np.arange(len(frame))
    frame = repair_frame(frame)
    _, fr_index, fr_counts = np.unique(frame,
                                       return_index=True,
                                       return_counts=True,
                                       )
    # We are only interested in frames with multiple events
    fr_relevant = fr_counts > 1
    for start, count in zip(fr_index[fr_relevant], fr_counts[fr_relevant]):
        # events with identical frames are consecutive
        same = np.arange(start, start + count)
        links[same] = np.roll(same, 1)
    return links


def repair_frame(frame: np.ndarray) -> np.ndarray:
    """Return a version of the "frame" feature that never decreases

    A faulty camera may return invalid frame numbers (e.g. all-zero
    frame numbers from some point on or a frame counter that restarts).
    Starting with the first event whose frame number is smaller than
    that of the previous event, all events are assigned consecutive
    frame numbers (as in dclab). No data are lost, but after such an
    invalid event, there is only one event per reassembled frame.

    Returns
    -------
    frame_repaired:
        64-bit integer copy of `frame` that is monotonically increasing
    """
    # int64, not uint, because `np.diff` would wrap around for uint.
    frame = np.array(frame, dtype=np.int64, copy=True)
    frames_invalid = np.diff(frame) < 0
    if np.any(frames_invalid):
        warnings.warn("Detected negative frame count",
                      NegativeFrameCountWarning)
        ii = np.argmax(frames_invalid)
        frame[ii + 1:] = frame[ii] + 1 + np.arange(frame.size - ii - 1)
    return frame


def chop_events(data: np.ndarray,
                shapes: np.ndarray,
                offsets: np.ndarray,
                data_bg: np.ndarray | None = None,
                mask: np.ndarray | None = None,
                dilation_structure: np.ndarray | None = None,
                ) -> list[np.ndarray]:
    """Crop the event images

    Parameters
    ----------
    data:
        3D array of event images
    shapes, offsets:
        shapes and offsets of the chopped images
    data_bg:
        background images that are subtracted from `data` (modulo 256
        for uint8 data)
    mask:
        event masks; required for `dilation_structure`
    dilation_structure:
        if given, pixels outside the binary dilation of the mask with
        this structure are set to zero (crop method "dilation")

    Returns
    -------
    chopped:
        list of chopped event images
    """
    if dilation_structure is not None and mask is None:
        raise ValueError("The mask is required for binary dilation")
    chopped = []
    for ii in range(len(data)):
        shy, shx = shapes[ii]
        offy, offx = offsets[ii]
        sl = (slice(offy, offy + shy), slice(offx, offx + shx))
        chop = np.array(data[ii][sl], copy=True)
        if data_bg is not None:
            chop -= data_bg[ii][sl]
        if dilation_structure is not None and mask is not None:
            # Binary dilation with OpenCV is much faster than with
            # `scipy.ndimage.binary_dilation` (used in dclab) and yields
            # the same result (symmetric structure, zero border).
            chop *= cv2.dilate(np.asarray(mask[ii][sl], dtype=bool).astype(
                np.uint8),
                kernel=dilation_structure.astype(np.uint8),
                borderType=cv2.BORDER_CONSTANT,
                borderValue=0)
        chopped.append(chop)
    return chopped


def get_dilation_structure(crop_method: Literal["box", "dilation"],
                           pixel_size: float,
                           pad_um: float = 1.5) -> np.ndarray | None:
    """Return the structure for binary dilation (None for "box")"""
    if crop_method == "dilation":
        # The radius of the disk is equal to the padding size.
        return disk(int(np.ceil(pad_um / pixel_size)))
    elif crop_method == "box":
        return None
    else:
        raise ValueError(f"Invalid crop method '{crop_method}', expected "
                         f"'box' or 'dilation'")


def store_chopped_basin(h5: h5py.File,  # type: ignore
                        feat: str,
                        ) -> str:
    """Write the basin definition for chopped data in `basin_events/feat`

    Returns the name of the basin dataset in the "basins" group.
    """
    b_data = {
        "description": None,
        "features": [feat],
        "format": "h5datasetchop",
        "identifier": None,
        "mapping": "same",
        "name": f"chopped-{feat}",
        "paths": ["basin_events"],
        "perishable": False,
        "type": "internal",
    }
    bstring = json.dumps(b_data, indent=2, sort_keys=True)
    key = hashlib.md5(bstring.encode("utf-8")).hexdigest()
    basins = h5.require_group("basins")
    if key not in basins:
        blines = bstring.split("\n")
        basins.create_dataset(
            name=key,
            data=blines,
            dtype=f"S{max([len(b) for b in blines])}")
    return key
