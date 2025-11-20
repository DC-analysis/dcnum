from typing import List

import numpy as np


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
        self.events = {}
        """Dictionary containing the event arrays"""

        self.feat_nevents = feat_nevents
        """List containing the number of events per input frame"""

        self.nev_idx = np.cumsum(feat_nevents)
        """Cumulative sum of `feat_nevents` for determining sorting offsets"""

        self.size = int(np.sum(feat_nevents))
        """Number of events in this stash"""

        self.num_frames = len(feat_nevents)
        """Number of frames in this stash"""

        self.index_offset = index_offset
        """Global offset compared to the original data instance."""

        self.indices_for_data = np.zeros(self.size, dtype=np.uint32)
        """Array containing the indices in the original data instance.
        These indices correspond to the events in `events`.
        """

        self._tracker = np.zeros(self.num_frames, dtype=bool)
        """Private array that tracks the progress."""

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

        if events:
            slice_loc = None
            idx_stop = self.nev_idx[idx_loc]
            for feat in events:
                dev = events[feat]
                if dev.size:
                    darr = self.require_feature(feat=feat,
                                                sample_data=dev[0])
                    slice_loc = (slice(idx_stop - dev.shape[0], idx_stop))
                    darr[slice_loc] = dev
            if slice_loc:
                self.indices_for_data[slice_loc] = index

        self._tracker[idx_loc] = True

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
        if feat not in self.events:
            sample_data = np.array(sample_data)
            event_shape = sample_data.shape
            dtype = sample_data.dtype
            darr = np.zeros((self.size,) + tuple(event_shape),
                            dtype=dtype)
            self.events[feat] = darr
        return self.events[feat]
