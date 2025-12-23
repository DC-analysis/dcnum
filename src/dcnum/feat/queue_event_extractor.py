"""Feature Extraction: event extractor worker"""
import logging
import multiprocessing as mp

import numpy as np

from ..common import LazyLoader
from ..meta.ppid import kwargs_to_ppid, ppid_to_kwargs

from .gate import Gate


feat_brightness = LazyLoader("feat_brightness", __name__)
feat_contour = LazyLoader("feat_contour", __name__)
feat_texture = LazyLoader("feat_texture", __name__)


class QueueEventExtractor:
    def __init__(self,
                 slot_register: "SlotRegister",  # noqa: F821
                 pixel_size: float,
                 gate: Gate,
                 event_queue: "mp.Queue",
                 extract_kwargs: dict = None,
                 logger: logging.Logger = None):
        """Event extraction from label images

        This class is used for extracting events from label images.
        Events are appended to the `event_queue` for the writer.

        Parameters
        ----------
        slot_register: .logic.slot_register.SlotRegister
            Chunk slot register
        pixel_size:
            Imaging pixel size
        gate: .gate.Gate
            Gating rules.
        event_queue:
            Queue in which the worker puts the extracted event feature
            data.
        extract_kwargs:
            Keyword arguments for the extraction process. See the
            keyword-only arguments in
            :func:`QueueEventExtractor.get_events_from_masks`.
        logger:
            Logger to use
        """
        self.slot_register = slot_register
        """Chunk slot register"""

        self.pixel_size = pixel_size
        """Imaging pixel size"""

        self.gate = gate
        """Gating information"""

        self.event_queue = event_queue
        """queue with event-wise feature dictionaries"""

        self.logger = logger or logging.getLogger(__name__)

        # Keyword arguments for data extraction
        if extract_kwargs is None:
            extract_kwargs = {}
        extract_kwargs.setdefault("brightness", True)
        extract_kwargs.setdefault("haralick", True)

        self.extract_kwargs = extract_kwargs
        """Feature extraction keyword arguments."""

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
            feat_contour.moments_based_features(
                masks,
                pixel_size=self.pixel_size,
                ret_contour=volume,
                ))

        if brightness:
            events.update(feat_brightness.brightness_features(
                mask=masks,
                image=image,
                image_bg=image_bg,
                bg_off=bg_off,
                image_corr=image_corr
            ))
        if haralick:
            events.update(feat_texture.haralick_texture_features(
                mask=masks,
                image=image,
                image_corr=image_corr,
            ))

        if volume:
            events.update(feat_contour.volume_from_contours(
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
        # over from gated_events to valid_events. In our experience, and
        # especially with U-Net-based segmentation, invalid events happen
        # rarely.
        if np.any(invalid):
            with self.slot_register.get_counter_lock("masks_dropped"):
                self.slot_register.masks_dropped += np.sum(invalid)
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
