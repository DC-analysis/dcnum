import logging
from collections import deque
import queue
import multiprocessing as mp
import threading
import time
from typing import List

import numpy as np

from ..read import HDF5Data


class EventStash:
    def __init__(self,
                 index_offset: int,
                 feat_nevents: List[int]):
        """Sort events into predefined arrays for bulk access

        Parameters
        ----------
        index_offset:
            This is the index offset at which we are working on.
            Normally, `feat_nevents` is just a slice of a larger
            array and `index_offset` defines at which position
            it is taken.
        feat_nevents:
            List that defines how many events there are for each input
            frame. If summed up, this defines `self.size`.
        """
        #: Dictionary containing the event arrays
        self.events = {}
        #: List containing the number of events per input frame
        self.feat_nevents = feat_nevents
        #: Cumulative sum of `feat_nevents` for determining sorting offsets
        self.nev_idx = np.cumsum(feat_nevents)
        #: Number of events in this stash
        self.size = int(np.sum(feat_nevents))
        #: Number of frames in this stash
        self.num_frames = len(feat_nevents)
        #: Global offset compared to the original data instance.
        self.index_offset = index_offset
        #: Array containing the indices in the original data instance
        #: that correspond to the events in `events`.
        self.indices_for_data = np.zeros(self.size, dtype=np.uint32)
        # Private array that tracks the progress.
        self._tracker = np.zeros(self.num_frames, dtype=bool)

    def is_complete(self):
        """Determine whether the event stash is complete (all events added)"""
        return np.all(self._tracker)

    def add_events(self, index, events):
        """Add events to this stash

        Parameters
        ----------
        index: int
            Global index (from input dataset)
        events: dict
            Event dictionary
        """
        idx_loc = index - self.index_offset
        idx_stop = self.nev_idx[idx_loc]
        self._tracker[idx_loc] = True

        if events:
            slice_loc = None
            for feat in events:
                dev = events[feat]
                if dev.size:
                    darr = self.require_feature(feat=feat,
                                                sample_data=dev[0])
                    slice_loc = (slice(idx_stop - dev.shape[0], idx_stop))
                    darr[slice_loc] = dev
            if slice_loc:
                self.indices_for_data[slice_loc] = index

    def require_feature(self, feat, sample_data):
        """Create a new empty feature array in `self.events` and return it

        Parameters
        ----------
        feat:
            Feature name
        sample_data:
            Sample data for one event of the feature (used to determine
            shape and dtype of the feature array)
        """
        sample_data = np.array(sample_data)
        event_shape = sample_data.shape
        dtype = sample_data.dtype
        if feat not in self.events:
            darr = np.zeros((self.size,) + tuple(event_shape),
                            dtype=dtype)
            self.events[feat] = darr
        return self.events[feat]


class QueueCollectorThread(threading.Thread):
    def __init__(self,
                 data: HDF5Data,
                 event_queue: mp.Queue,
                 writer_dq: deque,
                 feat_nevents: mp.Array,
                 write_threshold: int = 500,
                 *args, **kwargs
                 ):
        """Convenience class for events from queues

        Events coming from a queue cannot be guaranteed to be in order.
        The :class:`.QueueCollectorThread` uses a :class:`.EventStash`
        to sort events into the correct order before sending them to
        the :class:`DequeWriterThread` for storage.

        Parameters
        ----------
        data:
            Data source object. This is used for appending additional
            information
        event_queue:
            A queue object to which other processes or threads write
            events as tuples `(frame_index, events_dict)`.
        writer_dq:
            A :class:`DequeWriterThread` should be attached to the
            other end of this :class:`collections.deque`.
        feat_nevents:
            This 1D array contains the number of events for each frame
            in the input data. This serves two purposes: (1) it allows
            us to determine how many events we are writing when we are
            writing data from `write_threshold` frames, and (2) it
            allows us to keep track how many frames have actually been
            processed (and thus we can expect entries in `event_queue`
            for). If an entry in this array is -1, this means that there
            is no event in `event_queue`. See `write_threshold` below.
        write_threshold:
            This integer defines how many frames should be collected at
            once and put into `writer_dq`. For instance, with a value of
            500, at least 500 items are taken from the `event_queue`
            (they should match the expected frame index, frame indices
            that do not match are kept in a :class:`.EventStash`). Then,
            for each frame, we may have multiple or None events, so the
            output size could be 513 which is computed via
            `np.sum(feat_nevents[idx:idx+write_threshold])`.
        """
        super(QueueCollectorThread, self).__init__(
              name="QueueCollector", *args, **kwargs)
        self.logger = logging.getLogger("dcnum.write.QueueCollector")
        #: HDF5 data instance
        self.data = data
        #: Event queue from which to collect event data
        self.event_queue = event_queue
        #: Writer deque to which event arrays are appended
        self.writer_dq = writer_dq
        #: Buffer deque to which events are appended that do not belong
        #: to the current chunk (chunk defined by `write_threshold`).
        self.buffer_dq = deque()
        #: A shared array containing the number of events for each
        #: frame in `data`.
        self.feat_nevents = feat_nevents
        #: Number of frames to send to `writer_dq` at a time.
        self.write_threshold = write_threshold
        #: Number of events sent to `writer_dq`
        self.written_events = 0
        #: Number of frames from `data` written to `writer_dq`
        self.written_frames = 0

    def run(self):
        # We are not writing to `event_queue` so we can safely cancel
        # our queue thread if we are told to stop.
        self.event_queue.cancel_join_thread()
        # Indexes the current frame in `self.data`.
        last_idx = 0
        self.logger.debug("Started collector thread")
        while True:
            # Slice of the shared nevents array. If it contains -1 values,
            # this means that some of the frames have not yet been processed.
            cur_nevents = self.feat_nevents[
                          last_idx:last_idx + self.write_threshold]
            if np.any(np.array(cur_nevents) < 0):
                # We are not yet ready to write any new data to the queue.
                time.sleep(.01)
                continue

            if len(cur_nevents) == 0:
                self.logger.info(
                    "Reached dataset end (frame "
                    # `last_idx` is the size of the dataset in the end,
                    # because `len(cur_nevents)` is always added to it.
                    f"{last_idx} of {len(self.feat_nevents)})")
                break

            # We have reached the writer threshold. This means the extractor
            # has analyzed at least `write_threshold` frames (not events).
            self.logger.debug(f"Current frame: {last_idx}")

            # Create an event stash
            stash = EventStash(
                index_offset=last_idx,
                feat_nevents=cur_nevents
            )

            # First check whether there is a matching event from the buffer
            # that we possibly populated earlier.
            for ii in range(len(self.buffer_dq)):
                idx, events = self.buffer_dq.popleft()
                if last_idx <= idx < last_idx + self.write_threshold:
                    stash.add_events(index=idx, events=events)
                else:
                    # Put it back into the buffer (this should not happen
                    # more than once unless you have many workers adding
                    # or some of the workers being slower/faster).
                    self.buffer_dq.append((idx, events))

            if not stash.is_complete():
                # Now, get the data from the queue until we have everything
                # that belongs to our chunk (this might also populate
                # buffer_dq).
                while True:
                    try:
                        idx, events = self.event_queue.get(timeout=.3)
                    except queue.Empty:
                        # No time.sleep here, because we are already using
                        # a timeout in event_queue.get.
                        continue
                    if last_idx <= idx < last_idx + self.write_threshold:
                        stash.add_events(index=idx, events=events)
                    else:
                        # Goes onto the buffer stack (might happen if a
                        # segmentation process was fast and got an event
                        # from the next slice (context: write_threshold))
                        self.buffer_dq.append((idx, events))
                    if stash.is_complete():
                        break

            # Send the data from the stash to the writer. The stash has
            # already put everything into the correct order.
            for feat in stash.events:
                self.writer_dq.append((feat, stash.events[feat]))

            # Now we also would like to add all the other information
            # that were not in the events dictionaries.

            # This array contains indices for `data` corresponding to
            # the events that we just saved.
            indices = stash.indices_for_data

            # Write all the scalar features.
            for feat in self.data.features_scalar_frame:
                self.writer_dq.append((feat, self.data[feat][indices]))

            # Write the image and background data.
            imdat = np.zeros((stash.size,) + self.data.image.image_shape,
                             dtype=np.uint8)
            bgdat = np.zeros((stash.size,) + self.data.image.image_shape,
                             dtype=np.uint8)
            for ii, idx in enumerate(indices):
                imdat[ii] = self.data.image[idx]
                bgdat[ii] = self.data.image_bg[idx]
            self.writer_dq.append(("image", imdat))
            self.writer_dq.append(("image_bg", bgdat))

            # Write the number of events.
            self.writer_dq.append(("nevents",
                                   # Get nevents for each event from the
                                   # frame-based cur_nevents array.
                                   np.array(stash.feat_nevents)[
                                       indices - stash.index_offset]
                                   ))
            # Update events/frames written (used for monitoring)
            self.written_events += stash.size
            self.written_frames += stash.num_frames

            # Increment current frame index.
            last_idx += len(cur_nevents)

        self.logger.info(f"Counted {self.written_events} events")
        self.logger.debug(f"Counted {self.written_frames} frames")
