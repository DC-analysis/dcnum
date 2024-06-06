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


class SegmenterNotApplicableError(BaseException):
    """Used to indicate when a dataset cannot be segmented with a segmenter"""
    def __init__(self, segmenter_class, reasons_list):
        super(SegmenterNotApplicableError, self).__init__(
            f"The dataset cannot be segmented with the "
            f"'{segmenter_class.get_ppid_code()}' segmenter: "
            f"{', '.join(reasons_list)}"
        )
        self.reasons_list = reasons_list
        self.segmenter_class = segmenter_class


class Segmenter(abc.ABC):
    #: Required hardware ("cpu" or "gpu") defined in first-level subclass.
    hardware_processor = "none"
    #: Whether to enable mask post-processing. If disabled, you should
    #: make sure that your mask is properly defined and cleaned or you
    #: have to call `process_mask` in your `segment_algorithm` implementation.
    mask_postprocessing = True
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
        """Base segmenter class

        This is the base segmenter class for the multiprocessing operation
        segmenter :class:`.MPOSegmenter` (multiple subprocesses are spawned
        and each of them works on a queue of images) and the single-threaded
        operation segmenter :class:`.STOSegmenter` (e.g. for batch
        segmentation on a GPU).

        Parameters
        ----------
        kwargs_mask: dict
            Keyword arguments for mask post-processing (see `process_mask`)
        debug: bool
            Enable debugging mode (e.g. CPU segmenter runs in one thread)
        kwargs:
            Additional, optional keyword arguments for `segment_batch`.
        """
        self.debug = debug
        self.logger = logging.getLogger(__name__).getChild(
            self.__class__.__name__)
        spec = inspect.getfullargspec(self.segment_algorithm)
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
        border[[0, -1], :] = True
        border[:, [0, -1]] = True
        return border

    @staticmethod
    @functools.cache
    def get_disk(radius):
        """Cached `skimage.morphology.disk(radius)`"""
        return morphology.disk(radius)

    def get_ppid(self):
        """Return a unique segmentation pipeline identifier

        The pipeline identifier is universally applicable and must
        be backwards-compatible (future versions of dcnum will
        correctly acknowledge the ID).

        The segmenter pipeline ID is defined as::

            KEY:KW_APPROACH:KW_MASK

        Where KEY is e.g. "legacy" or "watershed", and KW_APPROACH is a
        list of keyword arguments for `segment_algorithm`, e.g.::

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
        if cls.mask_postprocessing:
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
        elif kwargs_mask:
            raise ValueError(f"The segmenter '{cls.__name__}' does not "
                             f"support mask postprocessing, but 'kwargs_mask' "
                             f"was provided: {kwargs_mask}")

        ppid_parts = [
            cls.get_ppid_code(),
            kwargs_to_ppid(cls, "segment_algorithm", kwargs),
            ]

        if cls.mask_postprocessing:
            ppid_parts.append(
                kwargs_to_ppid(cls, "process_mask", kwargs_mask_used))

        return ":".join(ppid_parts)

    @staticmethod
    def get_ppkw_from_ppid(segm_ppid):
        """Return keyword arguments for this pipeline identifier"""
        ppid_parts = segm_ppid.split(":")
        code = ppid_parts[0]
        pp_kwargs = ppid_parts[1]

        for cls_code in get_available_segmenters():
            if cls_code == code:
                cls = get_available_segmenters()[cls_code]
                break
        else:
            raise ValueError(
                f"Could not find segmenter '{code}'!")
        kwargs = ppid_to_kwargs(cls=cls,
                                method="segment_algorithm",
                                ppid=pp_kwargs)
        if cls.mask_postprocessing:
            pp_kwargs_mask = ppid_parts[2]
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
        labels: 2d integer or boolean ndarray
            Labeled input (contains blobs consisting of unique numbers)
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
        if labels.dtype == bool:
            # Convert mask image to labels
            labels, _ = ndi.label(
                input=labels,
                structure=ndi.generate_binary_structure(2, 2))

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
            # Since floodfill will use the upper left corner of the image as
            # a seed, we have to make sure it is set to background. We set
            # a line of pixels in the upper channel wall to zero to be sure.
            labels[0, :] = 0
            # ...and a 4x4 pixel region in the top left corner.
            labels[1, :2] = 0
            cv2.floodFill(labels, None, (0, 0), 2147483647)
            mask = labels != 2147483647
            labels, _ = ndi.label(
                input=mask,
                structure=ndi.generate_binary_structure(2, 2))

        if closing_disk:
            # scikit-image is too slow for us here. So we use OpenCV.
            # https://github.com/scikit-image/scikit-image/issues/1190
            #
            # from skimage import morphology
            # morphology.binary_closing(
            #    mask,
            #    footprint=morphology.disk(closing_disk),
            #    out=mask)
            #
            element = Segmenter.get_disk(closing_disk)
            # Note: erode/dilate not implemented for int32
            labels_uint8 = np.array(labels, dtype=np.uint8)
            # Historically, we would like to do a closing (dilation followed
            # by erosion) on the image data where lower brightness values
            # meant "we have an event". However, since we are now working
            # with labels instead of image data (0 is background and labels
            # are enumerated with integers), high "brightness" values are
            # actually the event. Thus, we have to perform an opening
            # (erosion followed by dilation) of the label image.
            labels_eroded = cv2.erode(labels_uint8, element)
            labels_dilated = cv2.dilate(labels_eroded, element)
            labels, _ = ndi.label(
                input=labels_dilated > 0,
                structure=ndi.generate_binary_structure(2, 2))

        return labels

    @staticmethod
    @abc.abstractmethod
    def segment_algorithm(image):
        """The segmentation algorithm implemented in the subclass

        Perform segmentation and return integer label or binary mask image
        """

    @functools.cache
    def segment_algorithm_wrapper(self):
        """Wraps `self.segment_algorithm` to only accept an image

        The static method `self.segment_algorithm` may optionally accept
        keyword arguments `self.kwargs`. This wrapper returns the
        wrapped method that only accepts the image as an argument. This
        makes sense if you want to unify
        """
        if self.kwargs:
            # For segmenters that accept keyword arguments.
            segm_wrap = functools.partial(self.segment_algorithm,
                                          **self.kwargs)
        else:
            # For segmenters that don't accept keyword arguments.
            segm_wrap = self.segment_algorithm
        return segm_wrap

    @abc.abstractmethod
    def segment_batch(self, images, start=None, stop=None, bg_off=None):
        """Return the integer labels for an entire batch

        This is implemented in the MPO and STO segmenters.
        """

    def segment_chunk(self, image_data, chunk, bg_off=None):
        """Return the integer labels for one `image_data` chunk

        This is a wrapper for `segment_batch`.

        Parameters
        ----------
        image_data:
            Instance of dcnum's :class:`.BaseImageChunkCache` with
            the methods `get_chunk` and `get_chunk_slice`.
        chunk: int
            Integer identifying the chunk in `image_data` to segment
        bg_off: ndarray
            Optional 1D array with same length as `image_data` that holds
            additional background offset values that should be subtracted
            from the image data before segmentation. Should only be
            used in combination with segmenters that have
            `requires_background_correction` set to True.
        """
        images = image_data.get_chunk(chunk)
        if bg_off is not None:
            bg_off_chunk = bg_off[image_data.get_chunk_slice(chunk)]
        else:
            bg_off_chunk = None
        return self.segment_batch(images, bg_off=bg_off_chunk)

    @abc.abstractmethod
    def segment_single(self, image):
        """Return the integer label for one image

        This is implemented in the MPO and STO segmenters.
        """

    @classmethod
    def validate_applicability(cls,
                               segmenter_kwargs: Dict,
                               meta: Dict = None,
                               logs: Dict = None):
        """Validate the applicability of this segmenter for a dataset

        Parameters
        ----------
        segmenter_kwargs: dict
            Keyword arguments for the segmenter
        meta: dict
            Dictionary of metadata from an :class:`HDF5Data` instance
        logs: dict
            Dictionary of logs from an :class:`HDF5Data` instance

        Returns
        -------
        applicable: bool
            True if the segmenter is applicable to the dataset

        Raises
        ------
        SegmenterNotApplicableError
            If the segmenter is not applicable to the dataset
        """
        return True


@functools.cache
def get_available_segmenters():
    """Return dictionary of available segmenters"""
    segmenters = {}
    for scls in Segmenter.__subclasses__():
        for cls in scls.__subclasses__():
            segmenters[cls.get_ppid_code()] = cls
    return segmenters
