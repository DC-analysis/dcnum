"""Shared-memory slot holding one chunk of input data"""
import multiprocessing as mp

import numpy as np

from ..segm import get_segmenters

from .chunk_slot_data import ChunkSlotData


mp_spawn = mp.get_context("spawn")


class ChunkSlot(ChunkSlotData):
    _instance_counter = 0

    def __init__(self, job, data, is_remainder=False):
        """A slot holding one chunk of image data in shared memory

        A chunk slot holds the image data (and, if available,
        background image, background-corrected image, and brightness
        offset data) of one chunk of the input dataset, together with
        the mask and label arrays computed from it. All arrays live in
        shared memory, so that multiple worker processes can operate
        on the same chunk. Chunk slots are created and managed by a
        :class:`~dcnum.logic.slot_register.SlotRegister`, and workers
        move them through the states defined in
        :attr:`~dcnum.logic.chunk_slot_data.ChunkSlotData.state`
        (loading, segmentation, labeling, label processing, feature
        extraction). Once all tasks are done, a slot is reused for
        the next chunk.

        Parameters
        ----------
        job: dcnum.logic.job.DCNumPipelineJob
            pipeline job; ``job["segmenter_code"]`` defines the
            segmenter class
        data: dcnum.read.hdf5_data.HDF5Data
            input data from which chunks are loaded
        is_remainder: bool
            whether this slot is used only for the last chunk of the
            input data, whose size may be smaller than the regular
            chunk size
        """
        ChunkSlot._instance_counter += 1
        self.index = ChunkSlot._instance_counter

        self.job = job
        """Job information object"""

        self.data = data
        """Input data object"""

        self.is_remainder = is_remainder
        """Whether this slot only applies to the last chunk"""

        self.seg_cls = get_segmenters()[self.job["segmenter_code"]]
        """Segmentation class"""

        if self.is_remainder:
            length = self.data.image.get_chunk_size(
                    chunk_index=self.data.image.num_chunks - 1)
        else:
            length = self.data.image.chunk_size

        super().__init__(
            shape=(length,) + self.data.image.shape[1:],
            available_features=self.data.keys(),
        )

    def __repr__(self):
        return (f"<dcnum ChunkSlot {self.index} (state {self.state}) "
                f"with chunk {self.chunk} at {hex(id(self))}>")

    def load(self, idx):
        """Load chunk ``idx`` into shared memory and return numpy views

        Parameters
        ----------
        idx: int
            index of the chunk in the input data

        Returns
        -------
        image: np.ndarray
            image data
        image_bg: np.ndarray | None
            background image data (None if not available)
        image_corr: np.ndarray | None
            background-corrected image data, ``image - image_bg``
            (None if no background is available)
        bg_off: np.ndarray | None
            brightness offset data (None if not available)
        """
        # create views on image arrays
        image = self.image
        image[:] = self.data.image.get_chunk(idx)

        if self.mp_image_bg is not None:
            image_bg = self.image_bg
            image_bg[:] = self.data.image_bg.get_chunk(idx)
            image_corr = self.image_corr
            image_corr[:] = np.asarray(image, dtype=np.int16) - image_bg
        else:
            image_bg = None
            image_corr = None

        if self.mp_bg_off is not None:
            bg_off = self.bg_off
            chunk_slice = self.data.image.get_chunk_slice(idx)
            bg_off[:] = self.data["bg_off"][chunk_slice]
        else:
            bg_off = None

        # TODO: Check for duplicate, consecutive images while loading data
        #  and store that information in a boolean array. This can speed-up
        #  segmentation and feature extraction.

        self.chunk = idx
        return image, image_bg, image_corr, bg_off
