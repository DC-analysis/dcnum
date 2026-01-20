import logging
import time
import threading
import traceback

from .segmenter import Segmenter
from .segmenter_mpo import MPOSegmenter


class SegmenterManagerThread(threading.Thread):
    def __init__(self,
                 segmenter: Segmenter,
                 slot_register: "SlotRegister",  # noqa: F821
                 *args, **kwargs):
        """Manage the segmentation of image data

        Parameters
        ----------
        segmenter:
            The segmenter instance to use.
        slot_register:
            Manages a list of `ChunkSlots`, shared arrays on which
            to operate

        Notes
        -----
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

        self.segmenter = segmenter
        """Segmenter instance"""

        self.slot_register = slot_register
        """Slot manager"""

        self.t_segm = 0
        """Segmentation time counter"""

        self.t_wait = 0
        """Waiting time counter"""

    def run(self):
        try:
            self.segmenter.log_info(self.logger)
        except BaseException:
            self.logger.error("Failed to log device information")
            self.logger.info(traceback.format_exc())

        # We iterate over all the chunks of the image data.
        for chunk in range(self.slot_register.num_chunks):
            t0 = time.perf_counter()

            while True:
                # Find the slot that has the `chunk`
                # (preloaded from disk by UniversalWorker)
                state_warden = self.slot_register.reserve_slot_for_task(
                    current_state="s",
                    next_state="e"
                )
                if state_warden is None or state_warden.batch_size == 0:
                    time.sleep(.01)
                else:
                    break

            # We have a free slot to compute the segmentation
            t1 = time.perf_counter()
            self.t_wait += t1 - t0

            with state_warden as (cs, _):
                if state_warden.batch_size != cs.length:
                    raise ValueError(f"Batch size must match chunk size "
                                     f"({state_warden.batch_size=} vs. "
                                     f"{cs.length=})")

                # `segment_chunk` populates the `cs.labels` array.
                self.segmenter.segment_chunk(cs.chunk,
                                             self.slot_register.slots)
            self.logger.debug(f"Segmented chunk {chunk} in slot {cs}")

            self.t_segm += time.perf_counter() - t1

        # Cleanup
        if isinstance(self.segmenter, MPOSegmenter):
            self.segmenter.close()

        self.logger.info(f"Segmentation time: {self.t_segm:.1f}s")
        self.logger.info(f"Waiting time: {self.t_wait:.1f}s")
