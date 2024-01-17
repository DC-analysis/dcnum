"""Feature gating"""
import copy
import numbers
import warnings

import numpy as np

from ..meta.ppid import kwargs_to_ppid, ppid_to_kwargs


class Gate:
    #: the default value for `size_thresh_mask` if not given as kwarg
    _default_size_thresh_mask = 10

    def __init__(self, data, *,
                 online_gates: bool = False,
                 size_thresh_mask: int = None):
        """Gate feature data

        Parameters
        ----------
        data: .HDF5Data
            dcevent data instance
        online_gates: bool
            set to True to enable gating with "online" gates stored
            in the input file; online gates are applied in real-time
            deformability cytometry before writing data to disk during
            a measurement
        size_thresh_mask: int
            Only masks with more pixels than `size_thresh_mask` are
            considered to be a valid event; Originally, the
            `bin area min`/`trig_thresh` value defaulted to 200 which is
            too large; defaults to 10 or the original value in case
            `online_gates` is set.
        """
        #: box gating (value range for each feature)
        self.box_gates = {}

        if online_gates:
            # Deal with online gates.
            # First, compute the box gates.
            self.box_gates.update(self._extract_online_gates(data))
            # If the user did not specify a threshold, attempt to extract
            # it from the metadata.
            if size_thresh_mask is None:
                size_thresh_mask = data.meta_nest.get(
                    "online_contour", {}).get("bin area min")

        #: gating keyword arguments
        self.kwargs = {
            "online_gates": online_gates,
            # Set the size threshold, defaulting to `_default_size_thresh_mask`
            "size_thresh_mask":
                size_thresh_mask or self._default_size_thresh_mask
        }

    def _extract_online_gates(self, data):
        ogates = {}
        # Extract online filters from the dataset
        source_meta = data.meta_nest.get("online_filter", {})
        for key in source_meta:
            if key.endswith("polygon points"):
                raise NotImplementedError("Polygon gating not implemented!")
            elif (key.endswith("soft limit")
                    or key.startswith("target")):
                # we only want hard gates
                continue
            else:
                try:
                    feat, lim = key.rsplit(' ', 1)
                    lim_idx = ["min", "max"].index(lim)
                except ValueError:
                    warnings.warn(f"Unexpected online gate '{key}'")
                else:
                    # make sure we are not dealing with a soft limit
                    if not source_meta.get(f"{feat} soft limit", True):
                        ogates.setdefault(feat, [None, None])
                        ogates[feat][lim_idx] = source_meta[key]

        # This is somehow hard-coded in Shape-In (minimum size is 3px)
        px_size = data.pixel_size
        ogates["size_x"] = [
            max(ogates.get("size_x min", 0), 3 * px_size), None]
        ogates["size_y"] = [
            max(ogates.get("size_y min", 0), 3 * px_size), None]

        return ogates

    @property
    def features(self):
        """Sorted list of feature gates defined"""
        return sorted(self.box_gates.keys())

    def get_ppid(self):
        """Return a unique gating pipeline identifier

        The pipeline identifier is universally applicable and must
        be backwards-compatible (future versions of dcevent will
        correctly acknowledge the ID).

        The gating pipeline ID is defined as::

            KEY:KW_GATE

        Where KEY is e.g. "online_gates", and KW_GATE is
        the corresponding value, e.g.::

            online_gates=True^size_thresh_mask=5
        """
        return self.get_ppid_from_ppkw(self.kwargs)

    @classmethod
    def get_ppid_code(cls):
        return "norm"

    @classmethod
    def get_ppid_from_ppkw(cls, kwargs):
        """return full pipeline identifier from the given keywords"""
        # TODO:
        #  If polygon filters are used, the MD5sum should be used and
        #  they should be placed as a log to the output .rtdc file.
        kwargs = copy.deepcopy(kwargs)
        if kwargs.get("size_thresh_mask") is None:
            # Set the default described in init
            kwargs["size_thresh_mask"] = cls._default_size_thresh_mask
        key = cls.get_ppid_code()
        cback = kwargs_to_ppid(cls, "__init__", kwargs)

        return ":".join([key, cback])

    @staticmethod
    def get_ppkw_from_ppid(gate_ppid):
        code, pp_gate_kwargs = gate_ppid.split(":")
        if code != Gate.get_ppid_code():
            raise ValueError(
                f"Could not find gating method '{code}'!")
        kwargs = ppid_to_kwargs(cls=Gate,
                                method="__init__",
                                ppid=pp_gate_kwargs)
        return kwargs

    def gate_event(self, event):
        """Return None if the event should not be used, else `event`"""
        if self.box_gates and event:
            # Only use those events that are within the limits of the
            # online filters.
            for feat in self.features:
                if not self.gate_feature(feat, event[feat]):
                    return
        return event

    def gate_events(self, events):
        """Return boolean array with events that should be used"""
        if self.box_gates and bool(events):
            key0 = list(events.keys())[0]
            size = len(events[key0])
            valid = np.ones(size, dtype=bool)
            for feat in self.features:
                valid = np.logical_and(valid,
                                       self.gate_feature(feat, events[feat])
                                       )
        else:
            raise ValueError("Empty events provided!")
        return valid

    def gate_feature(self,
                     feat: str,
                     data: numbers.Number | np.ndarray):
        """Return boolean indicating whether `data` value is in box gate

        `data` may be a number or an array. If no box filter is defined
        for `feat`, `True` is always returned. Otherwise, either a boolean
        or a boolean array is returned, depending on the type of `data`.
        Not that `np.logical_and` can deal with mixed argument types
        (scalar and array).
        """
        bound_lo, bound_up = self.box_gates[feat]
        valid_lo = data >= bound_lo if bound_lo is not None else True
        valid_up = data <= bound_up if bound_up is not None else True
        return np.logical_and(valid_lo, valid_up)

    def gate_mask(self, mask, mask_sum=None):
        """Gate the mask, return False if the mask should not be used

        Parameters
        ----------
        mask: 2d ndarray
            The boolean mask image for the event.
        mask_sum: int
            The sum of the mask (if not specified, it is computed)
        """
        if mask_sum is None:
            mask_sum = np.sum(mask)
        return mask_sum > self.kwargs["size_thresh_mask"]
