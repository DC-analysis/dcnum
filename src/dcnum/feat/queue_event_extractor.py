"""Feature Extraction: event extractor worker"""
import collections
import logging
from logging.handlers import QueueHandler
import multiprocessing as mp
import os
import queue
import threading
import traceback

import numpy as np

from ..os_env_st import RequestSingleThreaded, confirm_single_threaded
from ..meta.ppid import kwargs_to_ppid, ppid_to_kwargs

from .feat_brightness import brightness_features
from .feat_contour import moments_based_features, volume_from_contours
from .feat_texture import haralick_texture_features
from .gate import Gate


# All subprocesses should use 'spawn' to avoid issues with threads
# and 'fork' on POSIX systems.
mp_spawn = mp.get_context("spawn")


class QueueEventExtractor:
    def __init__(self,
                 slot_register: "SlotRegister",  # noqa: F821
                 pixel_size: float,
                 gate: Gate,
                 raw_queue: mp.Queue,
                 event_queue: mp.Queue,
                 log_queue: mp.Queue,
                 feat_nevents: mp.Array,
                 finalize_extraction: mp.Value,
                 invalid_mask_counter: mp.Value,
                 worker_monitor: mp.RawArray,
                 log_level: int = None,
                 extract_kwargs: dict = None,
                 worker_index: int = None,
                 *args, **kwargs):
        """Base class for event extraction from label images

        This class is meant to be subclassed to run either in a
        :class:`threading.Thread` or a :class:`multiprocessing.Process`.

        Parameters
        ----------
        slot_register: .logic.slot_register.SlotRegister
            Chunk slot register
        pixel_size:
            Imaging pixel size
        gate: .gate.Gate
            Gating rules.
        raw_queue:
            Queue from which the worker obtains the chunks and
            indices of the labels to work on.
        event_queue:
            Queue in which the worker puts the extracted event feature
            data.
        log_queue:
            Logging queue, used for sending messages to the main Process.
        feat_nevents:
            Shared array of same length as data into which the number of
            events per input frame is written. This array must be initialized
            with -1 (all values minus one).
        finalize_extraction:
            Shared value indicating whether this worker should stop as
            soon as the `raw_queue` is empty.
        invalid_mask_counter:
            Counts masks labeled as invalid by the feature extractor
        worker_monitor:
            Monitors the frames each worker has processed. Only the
            value in `worker_monitor[worker_index]` is modified.
        log_level:
            Logging level to use
        extract_kwargs:
            Keyword arguments for the extraction process. See the
            keyword-only arguments in
            :func:`QueueEventExtractor.get_events_from_masks`.
        worker_index:
            The index to increment values in `worker_monitor`
        """
        super(QueueEventExtractor, self).__init__(*args, **kwargs)

        self.worker_index = worker_index or 0
        """Worker index for populating"""

        self.slot_register = slot_register
        """Chunk slot register"""

        self.pixel_size = pixel_size
        """Imaging pixel size"""

        self.gate = gate
        """Gating information"""

        self.raw_queue = raw_queue
        """queue containing sub-indices for the labels"""

        self.event_queue = event_queue
        """queue with event-wise feature dictionaries"""

        self.log_queue = log_queue
        """queue for logging"""

        self.invalid_mask_counter = invalid_mask_counter
        """invalid mask counter"""

        self.worker_monitor = worker_monitor
        """worker busy counter"""

        # Logging needs to be set up after `start` is called, otherwise
        # it looks like we have the same PID as the parent process. We
        # are setting up logging in `run`.
        self.logger = None
        self.log_level = log_level or logging.getLogger("dcnum").level

        self.feat_nevents = feat_nevents
        """Number of events per frame
        Shared array of length `len(data)` into which the number of
        events per frame is written.
        """

        self.finalize_extraction = finalize_extraction
        """Set to True to let worker join when `raw_queue` is empty."""

        # Keyword arguments for data extraction
        if extract_kwargs is None:
            extract_kwargs = {}
        extract_kwargs.setdefault("brightness", True)
        extract_kwargs.setdefault("haralick", True)

        self.extract_kwargs = extract_kwargs
        """Feature extraction keyword arguments."""

    @staticmethod
    def get_init_kwargs(slot_register: "SlotRegister",  # noqa: F821
                        pixel_size: float,
                        gate: Gate,
                        num_extractors: int,
                        log_queue: mp.Queue,
                        log_level: int = None,
                        ):
        """Get initialization arguments for :class:`.QueueEventExtractor`

        This method was created for convenience reasons:

        - It makes sure that the order of arguments is correct, since it
          is implemented in the same class.
        - It simplifies testing.

        Parameters
        ----------
        slot_register: .logic.slot_register.SlotRegister
            Chunk slot register
        pixel_size:
            Imaging pixel size
        gate: .gate.Gate
            Gating class to use
        num_extractors: int
            Number of extractors that will be used
        log_queue: mp.Queue
            Queue the worker uses for sending log messages
        log_level: int
            Logging level to use in the worker process

        Returns
        -------
        args: dict
            You can pass `*args.values()` directly to `__init__`
        """
        # queue with the raw (unsegmented) image data
        raw_queue = mp_spawn.Queue()
        # queue with event-wise feature dictionaries
        event_queue = mp_spawn.Queue()

        num_frames = slot_register.num_frames

        # Note that the order must be identical to  __init__
        args = collections.OrderedDict()
        args["slot_register"] = slot_register
        args["pixel_size"] = pixel_size
        args["gate"] = gate
        args["raw_queue"] = raw_queue
        args["event_queue"] = event_queue
        args["log_queue"] = log_queue
        args["feat_nevents"] = mp_spawn.Array("l", num_frames)
        args["feat_nevents"][:] = np.full(num_frames, -1)
        args["finalize_extraction"] = mp_spawn.Value("b", False)
        args["invalid_mask_counter"] = mp_spawn.Value("Q", 0)
        args["worker_monitor"] = mp_spawn.RawArray("I", num_extractors)
        args["log_level"] = log_level or logging.getLogger("dcnum").level
        return args

    def get_events_from_masks(self,
                              masks,
                              chunk_slot,
                              sub_index,
                              *,
                              brightness: bool = True,
                              haralick: bool = True,
                              volume: bool = True,
                              ):
        """Get events dictionary, performing event-based gating"""
        events = {"mask": masks}
        image = chunk_slot.image[sub_index][np.newaxis]
        image_bg = chunk_slot.image_bg[sub_index][np.newaxis]
        image_corr = chunk_slot.image_corr[sub_index][np.newaxis]
        if chunk_slot.bg_off is not None:
            bg_off = chunk_slot.bg_off[sub_index]
        else:
            bg_off = None

        events.update(
            moments_based_features(
                masks,
                pixel_size=self.pixel_size,
                ret_contour=volume,
                ))

        if brightness:
            events.update(brightness_features(
                mask=masks,
                image=image,
                image_bg=image_bg,
                bg_off=bg_off,
                image_corr=image_corr
            ))
        if haralick:
            events.update(haralick_texture_features(
                mask=masks,
                image=image,
                image_corr=image_corr,
            ))

        if volume:
            events.update(volume_from_contours(
                contour=events.pop("contour"),  # remove contour from events!
                pos_x=events["pos_x"],
                pos_y=events["pos_y"],
                pixel_size=self.pixel_size,
            ))

        # gating on feature arrays
        if self.gate.box_gates:
            valid = self.gate.gate_events(events)
            gated_events = {}
            for key in events:
                gated_events[key] = events[key][valid]
        else:
            gated_events = events

        # removing events with invalid features
        valid_events = {}
        # the valid key-value pair was added in `moments_based_features` and
        # its only purpose is to mark events with invalid contours as such,
        # so they can be removed here. Resolves issue #9.
        valid = gated_events.pop("valid")
        invalid = ~valid
        # The following might lead to a computational overhead, if only a few
        # events are invalid, because then all 2d-features need to be copied
        # over from gated_events to valid_events. According to our experience
        # invalid events happen rarely though.
        if np.any(invalid):
            self.invalid_mask_counter.value += np.sum(invalid)
            for key in gated_events:
                valid_events[key] = gated_events[key][valid]
        else:
            valid_events = gated_events

        return valid_events

    def get_masks_from_label(self, label):
        """Get masks, performing mask-based gating"""
        # Using np.unique is a little slower than iterating over lmax
        # unu = np.unique(label)  # background is 0
        lmax = np.max(label)
        masks = []
        for jj in range(1, lmax+1):  # first item is 0
            mask_jj = label == jj
            mask_sum = np.sum(mask_jj)
            if mask_sum and self.gate.gate_mask(mask_jj, mask_sum=mask_sum):
                masks.append(mask_jj)
        return np.array(masks)

    def get_ppid(self):
        """Return a unique feature extractor pipeline identifier

        The pipeline identifier is universally applicable and must
        be backwards-compatible (future versions of dcnum will
        correctly acknowledge the ID).

        The feature extractor pipeline ID is defined as::

            KEY:KW_APPROACH

        Where KEY is e.g. "legacy", and KW_APPROACH is a
        list of keyword-only arguments for `get_events_from_masks`,
        e.g.::

            brightness=True^haralick=True

        which may be abbreviated to::

            b=1^h=1
        """
        return self.get_ppid_from_ppkw(self.extract_kwargs)

    @classmethod
    def get_ppid_code(cls):
        return "legacy"

    @classmethod
    def get_ppid_from_ppkw(cls, kwargs):
        """Return the pipeline ID for this event extractor"""
        code = cls.get_ppid_code()
        cback = kwargs_to_ppid(cls, "get_events_from_masks", kwargs)
        return ":".join([code, cback])

    @staticmethod
    def get_ppkw_from_ppid(extr_ppid):
        code, pp_extr_kwargs = extr_ppid.split(":")
        if code != QueueEventExtractor.get_ppid_code():
            raise ValueError(
                f"Could not find extraction method '{code}'!")
        kwargs = ppid_to_kwargs(cls=QueueEventExtractor,
                                method="get_events_from_masks",
                                ppid=pp_extr_kwargs)
        return kwargs

    def process_label(self, index):
        """Process one label image, extracting masks and features"""
        chunk = index // self.slot_register.chunk_size
        sub_index = index % self.slot_register.chunk_size

        # Fetch the chunk slot we are supposed to be working on
        for chunk_slot in self.slot_register:
            if chunk_slot.chunk == chunk:
                break
        else:
            raise ValueError(f"Could not find slot for {chunk=}")

        images = chunk_slot.image

        # Check for duplicates
        # TODO: Check for duplicate images when loading data in ChunkSlot,
        #  and make that information available as a boolean array.
        if sub_index == 0:
            # We have to check whether the last image from the previous
            # chunk matches the current image.
            data = self.slot_register.data
            if (chunk != 0
                    and np.all(images[sub_index] == data.image[index - 1])):
                # skip duplicate events that have been analyzed already
                return None
        else:
            if np.all(images[sub_index] == images[sub_index - 1]):
                # skip duplicate events that have been analyzed already
                return None

        masks = self.get_masks_from_label(chunk_slot.labels[sub_index])
        if masks.size:
            events = self.get_events_from_masks(
                masks=masks,
                chunk_slot=chunk_slot,
                sub_index=sub_index,
                **self.extract_kwargs)
        else:
            events = None
        return events

    def run(self):
        """Main loop of worker process"""
        confirm_single_threaded()
        self.worker_monitor[self.worker_index] = 0
        # Don't wait for the raw queue when joining workers
        self.raw_queue.cancel_join_thread()

        self.logger = logging.getLogger(
            f"dcnum.feat.EventExtractor.{os.getpid()}")
        """logger that sends all logs to `self.log_queue`"""

        self.logger.setLevel(self.log_level)
        # Clear any handlers that might be set for this logger. This is
        # important for the case when we are an instance of
        # EventExtractorThread, because then all handlers from the main
        # thread are inherited (as opposed to no handlers in the case
        # of EventExtractorProcess).
        self.logger.handlers.clear()
        queue_handler = QueueHandler(self.log_queue)
        queue_handler.setLevel(self.log_level)
        self.logger.addHandler(queue_handler)
        self.logger.info("Ready")

        # only close queues when we have created them ourselves.
        close_queues = isinstance(self, EventExtractorProcess)

        while True:
            try:
                chunk_index, label_index = self.raw_queue.get(timeout=.03)
                index = (chunk_index * self.slot_register.chunk_size
                         + label_index)
            except queue.Empty:
                if self.finalize_extraction.value:
                    # The manager told us that there is nothing more coming.
                    self.logger.debug(
                        f"Finalizing worker {self} with PID {os.getpid()}")
                    break
            else:
                try:
                    events = self.process_label(index=index)
                except BaseException:
                    self.logger.error(traceback.format_exc())
                else:
                    if events:
                        key0 = list(events.keys())[0]
                        self.feat_nevents[index] = len(events[key0])
                    else:
                        self.feat_nevents[index] = 0
                    self.event_queue.put((index, events))
                self.worker_monitor[self.worker_index] += 1

        self.logger.debug(f"Finalizing `run` for PID {os.getpid()}, {self}")
        if close_queues:
            # Explicitly close the event queue and join it
            self.event_queue.close()
            self.event_queue.join_thread()
            self.logger.debug(f"End of `run` for PID {os.getpid()}, {self}")

        # Make sure everything gets written to the queue.
        queue_handler.flush()
        queue_handler.close()

        if close_queues:
            # Also close the logging queue. Note that not all messages might
            # arrive in the logging queue, since we called `cancel_join_thread`
            # earlier.
            self.log_queue.close()
            self.log_queue.join_thread()

        if close_queues:
            self.raw_queue.close()


class EventExtractorProcess(QueueEventExtractor, mp_spawn.Process):
    """Multiprocessing worker for regular segmentation and extraction"""
    def __init__(self, *args, **kwargs):
        super(EventExtractorProcess, self).__init__(
            name="EventExtractorProcess", *args, **kwargs)

    def start(self):
        # Set all relevant os environment variables such libraries in the
        # new process only use single-threaded computation.
        with RequestSingleThreaded():
            mp_spawn.Process.start(self)


class EventExtractorThread(QueueEventExtractor, threading.Thread):
    """Threading worker for debugging (only one single thread)"""
    def __init__(self, *args, **kwargs):
        super(EventExtractorThread, self).__init__(
            name="EventExtractorThread", *args, **kwargs)
