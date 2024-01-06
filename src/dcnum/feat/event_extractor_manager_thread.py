"""Feature computation: managing event extraction threads"""
import collections
import logging
import multiprocessing as mp
import threading
import time
from typing import Dict, List

import numpy as np

from .queue_event_extractor import EventExtractorThread, EventExtractorProcess


class EventExtractorManagerThread(threading.Thread):
    def __init__(self,
                 slot_states: mp.Array,
                 slot_chunks: mp.Array,
                 labels_list: List,
                 fe_kwargs: Dict,
                 num_workers: int,
                 writer_dq: collections.deque,
                 debug: bool = False,
                 *args, **kwargs):
        """Manage event extraction threads or precesses

        Parameters
        ----------
        slot_states:
            This is an utf-8 shared array whose length defines how many slots
            are available. The extractor will only ever extract features
            from a labeled image for a slot with segmented data. A slot
            with slot segmented data means a value of "e" (for "task is
            with extractor"). After the extractor has finished feature
            extraction, the slot value will be set to "s" (for "task is
            with segmenter"), so that the segmenter can compute a new
            chunk of labels.
        slot_chunks:
            For each slot in `slot_states`, this shared array defines
            on which chunk in `image_data` the segmentation took place.
        fe_kwargs:
            Feature extraction keyword arguments. See
            :func:`.EventExtractor.get_init_kwargs` for more information.
        num_workers:
            Number of child threads or worker processes to use.
        writer_dq:
            The queue the writer uses. We monitor this queue. If it
            fills up, we take a break.
        debug:
            Whether to run in debugging mode which means more log
            messages and only one thread (`num_workers` has no effect).
        """
        super(EventExtractorManagerThread, self).__init__(
              name="EventExtractorManager", *args, **kwargs)
        self.logger = logging.getLogger(
            "dcnum.feat.EventExtractorManagerThread")
        #: Keyword arguments for class:`.EventExtractor`
        self.fe_kwargs = fe_kwargs
        #: Data instance
        self.data = fe_kwargs["data"]
        #: States of the segmenter-extractor pipeline slots
        self.slot_states = slot_states
        #: Chunks indices corresponding to `slot_states`
        self.slot_chunks = slot_chunks
        #: Number of workers
        self.num_workers = 1 if debug else num_workers
        #: Queue for sending chunks and label indices to the workers
        self.raw_queue = self.fe_kwargs["raw_queue"]
        #: List of chunk labels corresponding to `slot_states`
        self.labels_list = labels_list
        #: Shared labeling array
        self.label_array = np.ctypeslib.as_array(
            self.fe_kwargs["label_array"]).reshape(
            self.data.image.chunk_shape)
        #: Writer deque to monitor
        self.writer_dq = writer_dq
        #: Time counter for feature extraction
        self.t_count = 0
        #: Whether debugging is enabled
        self.debug = debug

    def run(self):
        # Initialize all workers
        if self.debug:
            worker_cls = EventExtractorThread
        else:
            worker_cls = EventExtractorProcess
        workers = [worker_cls(*list(self.fe_kwargs.values()), worker_index=ii)
                   for ii in range(self.num_workers)]
        [w.start() for w in workers]
        worker_monitor = self.fe_kwargs["worker_monitor"]

        num_slots = len(self.slot_states)
        chunks_processed = 0
        frames_processed = 0
        while True:
            # If the writer_dq starts filling up, then this could lead to
            # an oom-kill signal. Stall for the writer to prevent this.
            ldq = len(self.writer_dq)
            if ldq > 100:
                stallsec = ldq / 100
                self.logger.warning(
                    f"Stalling {stallsec:.1f}s for slow writer")
                time.sleep(stallsec)

            cur_slot = 0
            unavailable_slots = 0
            # Check all slots for segmented labels
            while True:
                # - "e" there is data from the segmenter (the extractor
                #   can take it and process it)
                # - "s" the extractor processed the data and is waiting
                #   for the segmenter
                if self.slot_states[cur_slot] == "e":
                    # The segmenter has something for us in this slot.
                    break
                else:
                    # Try another slot.
                    unavailable_slots += 1
                    cur_slot = (cur_slot + 1) % num_slots
                if unavailable_slots >= num_slots:
                    # There is nothing to do, try to avoid 100% CPU
                    unavailable_slots = 0
                    time.sleep(.1)

            t1 = time.monotonic()

            # We have a chunk, process it!
            chunk = self.slot_chunks[cur_slot]
            # Populate the labeling array for the workers
            new_labels = self.labels_list[cur_slot]
            if len(new_labels) == self.label_array.shape[0]:
                self.label_array[:] = new_labels
            elif len(new_labels) < self.label_array.shape[0]:
                self.label_array[:len(new_labels)] = new_labels
                self.label_array[len(new_labels):] = 0
            else:
                raise ValueError("labels_list contains bad size data!")

            # Let the workers know there is work
            chunk_size = self.data.image.get_chunk_size(chunk)
            [self.raw_queue.put((chunk, ii)) for ii in range(chunk_size)]

            # Make sure the entire chunk has been processed.
            while np.sum(worker_monitor) != frames_processed + chunk_size:
                time.sleep(.1)

            # We are done here. The segmenter may continue its deed.
            self.slot_states[cur_slot] = "w"

            self.logger.debug(f"Extracted one chunk: {chunk}")
            self.t_count += time.monotonic() - t1

            chunks_processed += 1
            frames_processed += chunk_size

            if chunks_processed == self.data.image.num_chunks:
                break

        inv_masks = self.fe_kwargs["invalid_mask_counter"].value
        if inv_masks:
            self.logger.info(f"Encountered {inv_masks} invalid masks")
            inv_frac = inv_masks / len(self.data)
            if inv_frac > 0.005:  # warn above one half percent
                self.logger.warning(f"Discarded {inv_frac:.1%} of the masks, "
                                    f"please check segmenter applicability")

        self.logger.debug("Requesting extraction workers to join")
        self.fe_kwargs["finalize_extraction"].value = True
        [w.join() for w in workers]

        self.logger.debug("Finished extraction")
        self.logger.info(f"Extraction time: {self.t_count:.1f}s")
