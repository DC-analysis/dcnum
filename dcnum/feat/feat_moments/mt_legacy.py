import cv2
import numpy as np


from .ct_opencv import contour_single_opencv


def moments_based_features(mask, pixel_size):
    assert pixel_size is not None and pixel_size != 0

    size = mask.shape[0]

    empty = np.full(size, np.nan, dtype=np.float64)

    # features from raw contour
    feat_area_msd = np.copy(empty)
    feat_area_ratio = np.copy(empty)
    feat_area_um_raw = np.copy(empty)
    feat_aspect = np.copy(empty)
    feat_deform_raw = np.copy(empty)
    feat_eccentr_prnc = np.copy(empty)
    feat_inert_ratio_prnc = np.copy(empty)
    feat_inert_ratio_raw = np.copy(empty)
    feat_per_ratio = np.copy(empty)
    feat_per_um_raw = np.copy(empty)
    feat_size_x = np.copy(empty)
    feat_size_y = np.copy(empty)
    feat_tilt = np.copy(empty)

    # features from convex hull
    feat_area_um = np.copy(empty)
    feat_deform = np.copy(empty)
    feat_inert_ratio_cvx = np.copy(empty)
    feat_pos_x = np.copy(empty)
    feat_pos_y = np.copy(empty)

    # The following valid-array is not a real feature, but only
    # used to figure out which events need to be removed due
    # to invalid computed features, often due to invalid contours.
    valid = np.full(size, False)

    for ii in range(size):
        # raw contour
        cont_raw = contour_single_opencv(mask[ii])
        if len(cont_raw.shape) < 2:
            continue
        if cv2.contourArea(cont_raw) == 0:
            continue

        mu_raw = cv2.moments(cont_raw)
        arc_raw = np.float64(cv2.arcLength(cont_raw, True))
        area_raw = np.float64(mu_raw["m00"])

        # convex hull
        cont_cvx = np.squeeze(cv2.convexHull(cont_raw))
        mu_cvx = cv2.moments(cont_cvx)
        arc_cvx = np.float64(cv2.arcLength(cont_cvx, True))

        if mu_cvx["m00"] == 0 or mu_raw["m00"] == 0:
            # contour size too small
            continue

        # bounding box
        x, y, w, h = cv2.boundingRect(cont_raw)

        feat_area_msd[ii] = mu_raw["m00"]
        feat_area_ratio[ii] = mu_cvx["m00"] / mu_raw["m00"]
        feat_aspect[ii] = w / h
        feat_area_um[ii] = mu_cvx["m00"] * pixel_size**2
        feat_area_um_raw[ii] = area_raw * pixel_size**2
        feat_deform[ii] = 1 - 2 * np.sqrt(np.pi * mu_cvx["m00"]) / arc_cvx
        feat_deform_raw[ii] = 1 - 2 * np.sqrt(np.pi * area_raw) / arc_raw
        feat_per_ratio[ii] = arc_raw / arc_cvx
        feat_per_um_raw[ii] = arc_raw * pixel_size
        feat_pos_x[ii] = mu_cvx["m10"] / mu_cvx["m00"] * pixel_size
        feat_pos_y[ii] = mu_cvx["m01"] / mu_cvx["m00"] * pixel_size
        feat_size_x[ii] = w * pixel_size
        feat_size_y[ii] = h * pixel_size

        # inert_ratio_cvx
        if mu_cvx['mu02'] > 0:  # defaults to zero
            feat_inert_ratio_cvx[ii] = np.sqrt(mu_cvx['mu20'] / mu_cvx['mu02'])

        # moments of inertia of raw contour
        i_xx = np.float64(mu_raw["mu02"])
        i_yy = np.float64(mu_raw["mu20"])
        i_xy = np.float64(mu_raw["mu11"])

        # tilt
        feat_tilt[ii] = np.abs(0.5 * np.arctan2(-2 * i_xy, i_yy - i_xx))

        # inert_ratio_raw
        if i_xx > 0:  # defaults to zero
            feat_inert_ratio_raw[ii] = np.sqrt(i_yy / i_xx)

        # central moments in principal axes
        i_root_1 = (i_xx - i_yy) ** 2 + 4*(i_xy ** 2)
        i_root_2 = (i_xx - i_yy) ** 2 + 4*(i_xy ** 2)

        # inert_ratio_prnc and eccentr_prnc
        if i_root_1 >= 0 and i_root_2 >= 0:
            i_1 = 0.5 * (i_xx + i_yy + np.sqrt(i_root_1))
            i_2 = 0.5 * (i_xx + i_yy - np.sqrt(i_root_2))
            i_ratio = i_1 / i_2
            if i_ratio >= 0:
                feat_inert_ratio_prnc[ii] = np.sqrt(i_ratio)

            feat_eccentr_prnc[ii] = np.sqrt((i_1 - i_2) / i_1)

        # specify validity
        valid[ii] = True

    return {
        "area_msd": feat_area_msd,
        "area_ratio": feat_area_ratio,
        "area_um": feat_area_um,
        "area_um_raw": feat_area_um_raw,
        "aspect": feat_aspect,
        "deform": feat_deform,
        "deform_raw": feat_deform_raw,
        "eccentr_prnc": feat_eccentr_prnc,
        "inert_ratio_cvx": feat_inert_ratio_cvx,
        "inert_ratio_prnc": feat_inert_ratio_prnc,
        "inert_ratio_raw": feat_inert_ratio_raw,
        "per_ratio": feat_per_ratio,
        "per_um_raw": feat_per_um_raw,
        "pos_x": feat_pos_x,
        "pos_y": feat_pos_y,
        "size_x": feat_size_x,
        "size_y": feat_size_y,
        "tilt": feat_tilt,
        "valid": valid,
    }
