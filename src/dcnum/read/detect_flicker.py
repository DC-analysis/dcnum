import numpy as np

from .hdf5_data import HDF5Data


def detect_flickering(image_data: np.ndarray | HDF5Data,
                      roi_height: int = 10,
                      brightness_threshold: float = 2.5,
                      count_threshold: int = 5,
                      max_frames: int = 1000):
    """Determine whether an image series experiences flickering

    Flickering is an unwelcome phenomenon due to a faulty data
    acquisition device. For instance, if there is random voltage noise in
    the electronics managing the LED power, then the brightness of the
    LED will vary randomly when the noise signal overlaps with the flash
    triggering signal.

    If flickering is detected, you should use the "sparsemed" background
    computation with `offset_correction` set to True.

    Parameters
    ----------
    image_data:
        sliceable object (e.g. numpy array or HDF5Data) containing
        image data.
    roi_height: int
        height of the ROI in pixels for which to search for flickering;
        the entire width of the image is used
    brightness_threshold: float
        brightness difference between individual ROIs median and median
        of all ROI medians leading to a positive flickering event
    count_threshold: int
        minimum number of flickering events that would lead to a positive
        flickering decision
    max_frames: int
        maximum number of frames to include in the flickering analysis
    """
    # slice event axis first in case we have and HDF5Data instance
    roi_data = image_data[:max_frames][:, :roi_height, :]
    roi_median = np.median(roi_data, axis=(1, 2))
    roi_offset = roi_median - np.median(roi_median)
    flickering_events = np.sum(np.abs(roi_offset) >= abs(brightness_threshold))
    return flickering_events >= count_threshold
