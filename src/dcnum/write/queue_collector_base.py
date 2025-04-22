from typing import List
import numpy as np

class EventStash:
    def __init__(self,
                 index_offset: int,
                 feat_nevents: List[int]):
        """Sortiert Ereignisse in vordefinierte Arrays für den Bulk-Zugriff

        Parameters
        ----------
        index_offset:
            Der Index-Offset, an dem gearbeitet wird.
            Normalerweise ist `feat_nevents` ein Slice eines größeren
            Arrays, und `index_offset` definiert die Position.
        feat_nevents:
            Liste, die angibt, wie viele Ereignisse es pro Eingabeframe gibt.
            Die Summe dieser Werte definiert `self.size`.
        """
        self.events = {}
        self.feat_nevents = feat_nevents
        self.nev_idx = np.cumsum(feat_nevents)
        self.size = int(np.sum(feat_nevents))
        self.num_frames = len(feat_nevents)
        self.index_offset = index_offset
        self.indices_for_data = np.zeros(self.size, dtype=np.uint32)
        self._tracker = np.zeros(self.num_frames, dtype=bool)

    def is_complete(self):
        """Bestimmt, ob der EventStash vollständig ist (alle Ereignisse hinzugefügt)"""
        return np.all(self._tracker)

    def add_events(self, index, events):
        """Fügt Ereignisse zu diesem Stash hinzu

        Parameters
        ----------
        index: int
            Globaler Index (aus dem Eingabedatensatz)
        events: dict
            Ereignis-Dictionary
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
        """Erstellt ein neues leeres Feature-Array in `self.events` und gibt es zurück

        Parameters
        ----------
        feat:
            Feature-Name
        sample_data:
            Beispieldaten für ein Ereignis des Features (zur Bestimmung
            der Form und des Datentyps des Feature-Arrays)
        """
        if feat not in self.events:
            sample_data = np.array(sample_data)
            event_shape = sample_data.shape
            dtype = sample_data.dtype
            darr = np.zeros((self.size,) + tuple(event_shape),
                            dtype=dtype)
            self.events[feat] = darr
        return self.events[feat]