"""Access to chopped image data ("h5datasetchop" internal basins)

Chopped image data store only the region of each event image that is
related to the event (bounding box plus padding). This format is written
e.g. by ``dclab-compress --crop-event-images`` and is described in the
dclab documentation (internal chopped image basin).

An "h5datasetchop" basin defines a group (e.g. ``basin_events/image``)
that contains

- an ``index`` dataset with one row per event: the name of the dataset
  that contains the event image (``"0"``, ``"1"``, ...), the index of the
  image in that dataset, the y and x offsets of the image in the frame,
  and the index of another event in the same frame (events of one frame
  are linked in a circle; an event without siblings links to itself),
- datasets ``"0"``, ``"1"``, ... with the event images, grouped by shape.

If the group has the attribute ``inverse_background_feature`` (e.g.
``"image_bg"``), the stored images are background-corrected (computed
modulo 256 for uint8 data) and the background image must be added again.
If the group has the attribute ``is_boolean``, the data are boolean
(e.g. ``"mask"``) and stored as uint8.
"""
from __future__ import annotations

import numbers

import numpy as np

from ..common import h5py


class ChoppedImageDataset:
    def __init__(self,
                 h5group: h5py.Group,  # type: ignore
                 roi_shape: tuple[int, int],
                 inverse_background=None):
        """Read-only access to chopped image data as full frames

        The object behaves like an image stack ``h5py.Dataset`` of shape
        ``(N, roi_y, roi_x)``: it can be indexed with integers, slices,
        and integer arrays, and it can be used for
        :class:`dcnum.read.cache.HDF5ImageCache` and
        :class:`dcnum.read.mapped.MappedHDF5Dataset`.

        Parameters
        ----------
        h5group: h5py.Group
            the group of the chopped feature (e.g. ``basin_events/image``)
        roi_shape: tuple of int
            shape of the full frames (``imaging:roi size y``,
            ``imaging:roi size x``)
        inverse_background:
            background image data with the same event indices as this
            dataset (e.g. an ``h5py.Dataset`` or a
            :class:`dcnum.read.mapped.MappedHDF5Dataset` of
            "image_bg"); required if the group defines the attribute
            ``inverse_background_feature``
        """
        self.h5group = h5group
        self.roi_shape = tuple(int(sh) for sh in roi_shape)
        self.index = np.asarray(h5group["index"][:], dtype=np.int64)
        """index array (dataset, sub-index, offset y, offset x, other)"""
        self.is_boolean = bool(h5group.attrs.get("is_boolean", False))
        self.inverse_background_feature = h5group.attrs.get(
            "inverse_background_feature")
        if self.inverse_background_feature and inverse_background is None:
            raise ValueError(
                f"The chopped data in '{h5group.name}' require the "
                f"background feature '{self.inverse_background_feature}'")
        self.inverse_background = inverse_background
        self.shape = (len(self.index),) + self.roi_shape
        self._datasets = {}

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, idx):
        if isinstance(idx, tuple):
            # Only the first index refers to events, the remaining
            # indices are applied to the event data (e.g. image ROI).
            idx, idx_data = idx[0], idx[1:]
        else:
            idx_data = ()
        if isinstance(idx, numbers.Integral):
            frames = self.get_frames(np.array([idx], dtype=np.int64))[0]
        else:
            indices = np.arange(len(self), dtype=np.int64)[idx]
            frames = self.get_frames(indices)
            idx_data = (slice(None),) + idx_data if idx_data else ()
        return frames[idx_data] if idx_data else frames

    @property
    def dtype(self):
        """Data type of the reassembled frames"""
        if self.is_boolean:
            return np.dtype(bool)
        else:
            return self._get_dataset(0).dtype

    @property
    def file(self):
        """HDF5 file of the chopped data"""
        return self.h5group.file

    @property
    def name(self):
        """Name of the HDF5 group of the chopped data"""
        return self.h5group.name

    def _get_dataset(self, ds_index):
        if ds_index not in self._datasets:
            self._datasets[ds_index] = self.h5group[str(ds_index)]
        return self._datasets[ds_index]

    def _get_siblings(self, index):
        """Return the indices of the other events in the frame of `index`

        The order is the order in which the event images are placed
        into the frame (the same order as in dclab).
        """
        siblings = []
        seen = [index]
        other = int(self.index[index, 4])
        # The links are circular; `seen` prevents endless loops for
        # malformed index data.
        while other not in seen:
            seen.append(other)
            siblings.append(other)
            other = int(self.index[other, 4])
        return siblings

    def _read_event_images(self, indices):
        """Return a dictionary with the stored images of the events"""
        images = {}
        rows = self.index[indices]
        for ds_index in np.unique(rows[:, 0]):
            sel = rows[:, 0] == ds_index
            ev_indices = indices[sel]
            sub_indices = rows[sel, 1]
            # Read all images of this dataset in one slice (the
            # sub-indices of consecutive events are increasing).
            start = int(np.min(sub_indices))
            stop = int(np.max(sub_indices)) + 1
            data = self._get_dataset(int(ds_index))[start:stop]
            for ev_idx, sub_idx in zip(ev_indices, sub_indices):
                images[int(ev_idx)] = data[sub_idx - start]
        return images

    def _place_image(self, frame, image, offy, offx, only_nonzero):
        """Place the event `image` in `frame` at the given offsets"""
        # crop the event image to the frame
        image = image[:self.roi_shape[0] - offy, :self.roi_shape[1] - offx]
        sy, sx = image.shape
        if only_nonzero:
            region = frame[offy:offy + sy, offx:offx + sx]
            support = image != 0
            region[support] = image[support]
        else:
            frame[offy:offy + sy, offx:offx + sx] = image

    def get_frames(self, indices):
        """Return the reassembled frames of the events with `indices`"""
        indices = np.asarray(indices, dtype=np.int64)
        siblings = {int(idx): self._get_siblings(int(idx))
                    for idx in indices}
        needed = set(siblings)
        for sibs in siblings.values():
            needed.update(sibs)
        images = self._read_event_images(
            np.array(sorted(needed), dtype=np.int64))

        frames = np.zeros((len(indices),) + self.roi_shape,
                          dtype=self._get_dataset(0).dtype)
        for ii, idx in enumerate(indices):
            idx = int(idx)
            _, _, offy, offx, _ = self.index[idx]
            self._place_image(frames[ii], images[idx], offy, offx,
                              only_nonzero=False)
            # Add the image data of the other events in the same frame
            for other in siblings[idx]:
                _, _, offy, offx, _ = self.index[other]
                self._place_image(frames[ii], images[other], offy, offx,
                                  only_nonzero=True)

        if self.inverse_background_feature:
            # The stored data are background-corrected (modulo 256 for
            # uint8), so adding the background restores the image.
            frames += get_data(self.inverse_background, indices)

        if self.is_boolean:
            frames = np.asarray(frames, dtype=bool)
        return frames


def get_data(obj, indices):
    """Return the data of `obj` for `indices` (reading a single slice)"""
    if len(indices) == 0:
        return obj[0:0]
    start = int(np.min(indices))
    stop = int(np.max(indices)) + 1
    data = obj[start:stop]
    return data[indices - start]
