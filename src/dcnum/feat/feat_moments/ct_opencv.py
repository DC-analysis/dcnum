import numpy as np
import cv2


def contour_single_opencv(mask):
    """Return the contour for a boolean mask image containg *one* blob"""

    cv_im = np.array(mask * 255, dtype=np.uint8)
    # determine the contour information using opencv
    conts, _ = cv2.findContours(cv_im,
                                cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_NONE)

    # raw contour
    cont_raw = np.squeeze(conts[0])

    return cont_raw
