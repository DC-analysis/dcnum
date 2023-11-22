"""Feature computation: gating after feature extraction"""
import copy
import warnings

import numpy as np

from ..meta.ppid import kwargs_to_ppid, ppid_to_kwargs


class Gate:
    #: the default value for `size_thresh_mask` if not given as kwarg
    _default_size_thresh_mask = 10

    def __init__(self, data, *, online_gates: bool = False,
                 size_thresh_mask: int = None):
        """Gate feature data

        Parameters
        ----------
        data: .HDF5Data
            dcevent data instance
        online_gates: bool
            set to True to enable gating with `online_filters`
        size_thresh_mask: int
            Only masks with more pixels than `size_thresh_mask` are
            considered to be a valid event; Originally, this
            `trig_thresh` value defaulted to 200, but this seemed to
            be a little too large, defaults to 10.
        """
        #: dcevent .HDF5Data instance
        self.data = data

        #: gating keyword arguments
        self.kwargs = {}

        if online_gates:
            self.box_gates = self._compute_online_gates()
            # Set triggering threshold to value from source dataset
            self._set_kwarg("size_thresh_mask", "online_contour",
                            "bin area min", size_thresh_mask)
            # If the user did not provide a value and there is nothing in the
            # original file, then set the default value.
            if self.kwargs.get("size_thresh_mask") is None:
                self.kwargs["size_thresh_mask"] = \
                    self._default_size_thresh_mask
        else:
            self.box_gates = {}
            # If the user did not provide a size_thresh_mask, use the default.
            if size_thresh_mask is None:
                size_thresh_mask = self._default_size_thresh_mask
            self.kwargs["size_thresh_mask"] = size_thresh_mask

        self.kwargs["online_gates"] = online_gates

    def _set_kwarg(self, name, sec, key, user_value):
        if user_value is None:
            value = self.data.meta_nest.get(sec, {}).get(key)
        else:
            value = user_value
        if value is not None:
            self.kwargs[name] = value

    def _compute_online_gates(self):
        all_online_filters = {}
        # Extract online filters from the dataset
        of = self.data.meta_nest.get("online_filter", {})
        for key in of:
            if key.endswith("polygon points"):
                raise NotImplementedError("Polygon gating not implemented!")
            elif (key.endswith("soft limit")
                    or key.startswith("target")):
                # we only want hard gates
                continue
            else:
                # only add the filter if it is not a soft limit
                sl = of.get(f"{key.rsplit(' ', 1)[0]} soft limit", True)
                if not sl:
                    all_online_filters[key] = of[key]

        # This is somehow hard-coded in Shape-In (minimum size is 3px)
        px_size = self.data.pixel_size
        all_online_filters["size_x min"] = max(
            all_online_filters.get("size_x min", 0), 3 * px_size)
        all_online_filters["size_y min"] = max(
            all_online_filters.get("size_y min", 0), 3 * px_size)

        return all_online_filters

    def gate_feature(self, feat, data):
        valid_left = True
        valid_right = True
        if f"{feat} min" in self.box_gates:
            valid_left = data > self.box_gates[f"{feat} min"]
        if f"{feat} max" in self.box_gates:
            valid_right = data < self.box_gates[f"{feat} max"]
        return np.logical_and(valid_left, valid_right)

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

    @property
    def features(self):
        return [kk.split()[0] for kk in list(self.box_gates.keys())]

    @classmethod
    def get_ppid_from_kwargs(cls, kwargs):
        warnings.warn(
            "Please use get_ppid_from_ppkw instead of get_ppid_from_kwargs.",
            DeprecationWarning)
        return cls.get_ppid_from_ppkw(kwargs)
