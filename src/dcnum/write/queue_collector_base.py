import logging
from collections import deque
import queue
import multiprocessing as mp
import time

import numpy as np


from .event_stash import EventStash


class QueueCollectorBase:
    def __init__(self,
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
        the :class:`ChunkWriter` for storage.

        Parameters
        ----------
        event_queue:
            A queue object to which other processes or threads write
            events as tuples `(frame_index, events_dict)`.
        writer_dq:
            A :class:`ChunkWriter` should be attached to the
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
        # Must call super init, otherwise Thread or Process are not initialized
        super(QueueCollectorBase, self).__init__()

        self.logger = logging.getLogger("dcnum.write.QueueCollector")

        self.event_queue = event_queue
        """Event queue from which to collect event data"""

        self.writer_dq = writer_dq
        """Writer deque to which event arrays are appended"""

        self.buffer_dq = deque()
        """Buffer deque
        Events that do not not belong to the current chunk
        (chunk defined by `write_threshold`) go here.
        """

        self.feat_nevents = feat_nevents
        """shared array containing the number of events
        for each frame in `data`."""

        self.write_threshold = write_threshold
        """Number of frames to send to `writer_dq` at a time."""

        self.written_events = 0
        """Number of events sent to `writer_dq`"""

        self.written_frames = 0
        """Number of frames from `data` written to `writer_dq`"""

    def run(self):
        # We are not writing to `event_queue` so we can safely cancel
        # our queue thread if we are told to stop.
        self.event_queue.cancel_join_thread()
        # Indexes the current frame in the input HDF5Data instance.
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

            # This is the unmapped index from the input HDF5Data instance.
            # Unmapped means that this only enumerates HDF5Data, but since
            # HDF5Data can be mapped, the index does not necessarily enumerate
            # the underlying HDF5 file. Later on, we will have to convert this
            # to the correct "basinmap0" feature
            # (see `DCNumJobRunner.task_enforce_basin_strategy`)
            self.writer_dq.append(("index_unmapped",
                                   np.array(indices, dtype=np.uint32)))

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
