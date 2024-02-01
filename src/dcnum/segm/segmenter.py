import abc
import copy
import functools
import inspect
import logging
from typing import Dict

import cv2
import numpy as np
import scipy.ndimage as ndi
from skimage import morphology

from ..meta.ppid import kwargs_to_ppid, ppid_to_kwargs


class Segmenter(abc.ABC):
    #: Required hardware ("cpu" or "gpu") defined in first-level subclass.
    hardware_processor = "none"
    #: Whether to enable mask post-processing. If disabled, you should
    #: make sure that your mask is properly defined and cleaned or you
    #: have to call `process_mask` in your `segment_approach` implementation.
    mask_postprocessing = False
    #: Default keyword arguments for mask post-processing. See `process_mask`
    #: for available options.
    mask_default_kwargs = {}
    #: If the segmenter requires a background-corrected image, set this to True
    requires_background_correction = False

    def __init__(self,
                 *,
                 kwargs_mask: Dict = None,
                 debug: bool = False,
                 **kwargs):
        """Base segmenter

        Parameters
        ----------
        kwargs_mask: dict
            Keyword arguments for mask post-processing (see `process_mask`)
        debug: bool
            Debugging parameters
        kwargs:
            Additional, optional keyword arguments for `segment_batch`.
        """
        self.debug = debug
        self.logger = logging.getLogger(__name__).getChild(
            self.__class__.__name__)
        spec = inspect.getfullargspec(self.segment_approach)
        #: custom keyword arguments for the subclassing segmenter
        self.kwargs = spec.kwonlydefaults or {}
        self.kwargs.update(kwargs)

        #: keyword arguments for mask post-processing
        self.kwargs_mask = {}
        if self.mask_postprocessing:
            spec_mask = inspect.getfullargspec(self.process_mask)
            self.kwargs_mask.update(spec_mask.kwonlydefaults or {})
            self.kwargs_mask.update(self.mask_default_kwargs)
            if kwargs_mask is not None:
                self.kwargs_mask.update(kwargs_mask)
        elif kwargs_mask:
            raise ValueError(
                "`kwargs_mask` has been specified, but mask post-processing "
                f"is disabled for segmenter {self.__class__}")

    @staticmethod
    @functools.cache
    def get_border(shape):
        """Cached boolean image with outer pixels set to True"""
        border = np.zeros(shape, dtype=bool)
        border[0, :] = True
        border[-1, :] = True
        border[:, 0] = True
        border[:, -1] = True
        return border

    @staticmethod
    @functools.cache
    def get_disk(radius):
        """Cached `skimage.morphology.disk(radius)`"""
        return morphology.disk(radius)

    def get_ppid(self):
        """Return a unique segmentation pipeline identifier

        The pipeline identifier is universally applicable and must
        be backwards-compatible (future versions of dcevent will
        correctly acknowledge the ID).

        The segmenter pipeline ID is defined as::

            KEY:KW_APPROACH:KW_MASK

        Where KEY is e.g. "legacy" or "watershed", and KW_APPROACH is a
        list of keyword arguments for `segment_approach`, e.g.::

            thresh=-6^blur=0

        which may be abbreviated to::

            t=-6^b=0

        KW_MASK represents keyword arguments for `process_mask`.
        """
        return self.get_ppid_from_ppkw(self.kwargs, self.kwargs_mask)

    @classmethod
    def get_ppid_code(cls):
        """The unique code/name of this segmenter class"""
        code = cls.__name__.lower()
        if code.startswith("segment"):
            code = code[7:]
        return code

    @classmethod
    def get_ppid_from_ppkw(cls, kwargs, kwargs_mask=None):
        """Return the pipeline ID from given keyword arguments

        See Also
        --------
        get_ppid: Same method for class instances
        """
        kwargs = copy.deepcopy(kwargs)
        if kwargs_mask is None and kwargs.get("kwargs_mask", None) is None:
            raise KeyError("`kwargs_mask` must be either specified as "
                           "keyword argument to this method or as a key "
                           "in `kwargs`!")
        if kwargs_mask is None:
            # see check above (kwargs_mask may also be {})
            kwargs_mask = kwargs.pop("kwargs_mask")
        # Start with the default mask kwargs defined for this subclass
        kwargs_mask_used = copy.deepcopy(cls.mask_default_kwargs)
        kwargs_mask_used.update(kwargs_mask)
        code = cls.get_ppid_code()
        csegm = kwargs_to_ppid(cls, "segment_approach", kwargs)
        cmask = kwargs_to_ppid(cls, "process_mask", kwargs_mask_used)
        return ":".join([code, csegm, cmask])

    @staticmethod
    def get_ppkw_from_ppid(segm_ppid):
        """Return keyword arguments for this pipeline identifier"""
        code, pp_kwargs, pp_kwargs_mask = segm_ppid.split(":")
        for cls_code in get_available_segmenters():
            if cls_code == code:
                cls = get_available_segmenters()[cls_code]
                break
        else:
            raise ValueError(
                f"Could not find segmenter '{code}'!")
        kwargs = ppid_to_kwargs(cls=cls,
                                method="segment_approach",
                                ppid=pp_kwargs)
        kwargs["kwargs_mask"] = ppid_to_kwargs(cls=cls,
                                               method="process_mask",
                                               ppid=pp_kwargs_mask)
        return kwargs

    @staticmethod
    def process_mask(labels, *,
                     clear_border: bool = True,
                     fill_holes: bool = True,
                     closing_disk: int = 5):
        """Post-process retrieved mask image

        This is an optional convenience method that is called for each
        subclass individually. To enable mask post-processing, set
        `mask_postprocessing=True` in the subclass and specify default
        `mask_default_kwargs`.

        Parameters
        ----------
        labels: 2d integer ndarray
            Labeled input (contains blobs with same number)
        clear_border: bool
            clear the image boarder using
            :func:`skimage.segmentation.clear_border`
        fill_holes: bool
            binary-fill-holes in the binary mask image using
            :func:`scipy.ndimage.binary_fill_holes`
        closing_disk: int or None
            if > 0, perform a binary closing with a disk
            of that radius in pixels
        """
        if clear_border:
            #
            # from skimage import segmentation
            # segmentation.clear_border(mask, out=mask)
            #
            if (labels[0, :].sum() or labels[-1, :].sum()
                    or labels[:, 0].sum() or labels[:, -1].sum()):
                border = Segmenter.get_border(labels.shape)
                indices = sorted(np.unique(labels[border]))
                for li in indices:
                    if li == 0:
                        # ignore background values
                        continue
                    labels[labels == li] = 0

        # scikit-image is too slow for us here. So we use OpenCV.
        # https://github.com/scikit-image/scikit-image/issues/1190

        if closing_disk:
            #
            # from skimage import morphology
            # morphology.binary_closing(
            #    mask,
            #    footprint=morphology.disk(closing_disk),
            #    out=mask)
            #
            element = Segmenter.get_disk(closing_disk)
            labels_uint8 = np.array(labels, dtype=np.uint8)
            labels_dilated = cv2.dilate(labels_uint8, element)
            labels_eroded = cv2.erode(labels_dilated, element)
            labels, _ = ndi.label(
                input=labels_eroded > 0,
                structure=ndi.generate_binary_structure(2, 2))

        if fill_holes:
            # Floodfill only works with uint8 (too small) or int32
            if labels.dtype != np.int32:
                labels = np.array(labels, dtype=np.int32)
            #
            # from scipy import ndimage
            # mask_old = ndimage.binary_fill_holes(mask)
            #
            # Floodfill algorithm fills the background image and
            # the resulting inversion is the image with holes filled.
            # This will destroy labels (adding 2,147,483,647 to background)
            cv2.floodFill(labels, None, (0, 0), 2147483647)
            mask = labels != 2147483647
            labels, _ = ndi.label(
                input=mask,
                structure=ndi.generate_binary_structure(2, 2))

        return labels

    def segment_chunk(self, image_data, chunk):
        """Return the integer labels for one `image_data` chunk"""
        data = image_data.get_chunk(chunk)
        return self.segment_batch(data)

    def segment_frame(self, image):
        """Return the integer label image for `index`"""
        segm_wrap = self.segment_frame_wrapper()
        # obtain mask or label
        mol = segm_wrap(image)
        if mol.dtype == bool:
            # convert mask to label
            labels, _ = ndi.label(
                input=mol,
                structure=ndi.generate_binary_structure(2, 2))
        else:
            labels = mol
        # optional postprocessing
        if self.mask_postprocessing:
            labels = self.process_mask(labels, **self.kwargs_mask)
        return labels

    @functools.cache
    def segment_frame_wrapper(self):
        if self.kwargs:
            # For segmenters that accept keyword arguments.
            segm_wrap = functools.partial(self.segment_approach,
                                          **self.kwargs)
        else:
            # For segmenters that don't accept keyword arguments.
            segm_wrap = self.segment_approach
        return segm_wrap

    @staticmethod
    @abc.abstractmethod
    def segment_approach(image):
        """Perform segmentation and return integer label or binary mask image

        This is the approach the subclasses implement.
        """

    @abc.abstractmethod
    def segment_batch(self, data, start=None, stop=None):
        """Return the integer labels for an entire batch"""


@functools.cache
def get_available_segmenters():
    """Return dictionary of available segmenters"""
    segmenters = {}
    for scls in Segmenter.__subclasses__():
        for cls in scls.__subclasses__():
            segmenters[cls.get_ppid_code()] = cls
    return segmenters
