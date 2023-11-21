import logging
import multiprocessing as mp
import time
import threading

import numpy as np

from ..read.cache import HDF5ImageCache, ImageCorrCache

from .segmenter import Segmenter
from .segmenter_cpu import CPUSegmenter


class SegmenterManagerThread(threading.Thread):
    def __init__(self,
                 segmenter: Segmenter,
                 image_data: HDF5ImageCache | ImageCorrCache,
                 slot_states: mp.Array,
                 slot_chunks: mp.Array,
                 debug: bool = False,
                 *args, **kwargs):
        """Manage the segmentation of image data

        Parameters
        ----------
        segmenter:
            The segmenter instance to use.
        image_data:
            The image data to use. This can be background-corrected
            or not (hence the type hint), depending on `segmenter`
        slot_states:
            This is an utf-8 shared array whose length defines how many slots
            are available. The segmenter will only ever perform the
            segmentation for a free slot. A free slot means a value of "s"
            (for "task is with segmenter"). After the segmenter has done
            its job for a slot, the slot value will be set to "e" (for
            "task is with feature extractor").
        slot_chunks:
            For each slot in `slot_states`, this shared array defines
            on which chunk in `image_data` the segmentation took place.
        debug:
            Whether to run in debugging mode (more verbose messages and
            CPU-based segmentation is done in one single thread instead
            of in multiple subprocesses).

        Notes
        -----
        This manager keeps a list `labels_list` which enumerates the
        slots just like `slot_states` and `slot_chunks` do. For each
        slot, this list contains the labeled image data (integer-valued)
        for the input `image_data` chunks.

        The working principle of this `SegmenterManagerThread` allows
        the user to define a fixed number of slots on which the segmenter
        can work on. For instance, if the segmenter is a CPU-based segmenter,
        it does not make sense to have more than one slot (because feature
        extraction should not take place at the same time). But if the
        segmenter is a GPU-based segmenter, then it makes sense to have
        more than one slot, so CPU and GPU can work in parallel.
        """
        super(SegmenterManagerThread, self).__init__(
              name="SegmenterManager", *args, **kwargs)
        self.logger = logging.getLogger("dcnum.segm.SegmenterManagerThread")
        #: Segmenter instance
        self.segmenter = segmenter
        #: Image data which is being segmented
        self.image_data = image_data
        #: Slot states
        self.slot_states = slot_states
        #: Current slot chunk index for the slot states
        self.slot_chunks = slot_chunks
        #: List containing the segmented labels of each slot
        self.labels_list = [None] * len(self.slot_states)
        #: Time counter for segmentation
        self.t_count = 0
        #: Whether running in debugging mode
        self.debug = debug

    def run(self):
        # We iterate over all the chunks of the image data.
        for chunk in self.image_data.iter_chunks():
            num_slots = len(self.slot_states)
            cur_slot = 0
            empty_slots = 0
            # Wait for a free slot to perform segmentation (compute labels)
            while True:
                # - "e" there is data from the segmenter (the extractor
                #   can take it and process it)
                # - "s" the extractor processed the data and is waiting
                #   for the segmenter
                if self.slot_states[cur_slot] != "e":
                    break
                else:
                    empty_slots += 1
                    cur_slot = (cur_slot + 1) % num_slots
                if empty_slots >= num_slots:
                    # There is nothing to do, try to avoid 100% CPU
                    empty_slots = 0
                    time.sleep(.01)

            t1 = time.monotonic()

            # We have a free slot to compute the segmentation
            labels = self.segmenter.segment_chunk(
                image_data=self.image_data,
                chunk=chunk)

            # TODO: make this more memory efficient (pre-shared mp.Arrays?)
            # Store labels in a list accessible by the main thread
            self.labels_list[cur_slot] = np.copy(labels)
            # Remember the chunk index for this slot
            self.slot_chunks[cur_slot] = chunk
            # This must be done last: Let the extractor know that this
            # slot is ready for processing.
            self.slot_states[cur_slot] = "e"
            self.logger.debug(f"Segmented one chunk: {chunk}")

            self.t_count += time.monotonic() - t1

        # Cleanup
        if isinstance(self.segmenter, CPUSegmenter):
            # Join the segmentation workers.
            self.segmenter.join_workers()

        self.logger.info(f"Segmentation time: {self.t_count:.1f}s")
