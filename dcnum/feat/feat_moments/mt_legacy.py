import cv2
import numpy as np


from .ct_opencv import contour_single_opencv


def moments_based_features(mask, pixel_size):
    assert pixel_size is not None and pixel_size != 0

    size = mask.shape[0]

    empty = np.full(size, np.nan, dtype=np.float64)

    # features from raw contour
    size_x = np.copy(empty)
    size_y = np.copy(empty)
    aspect = np.copy(empty)
    area_msd = np.copy(empty)
    area_um_raw = np.copy(empty)
    per_um_raw = np.copy(empty)
    deform_raw = np.copy(empty)
    area_ratio = np.copy(empty)
    per_ratio = np.copy(empty)
    inert_ratio_raw = np.copy(empty)
    inert_ratio_prnc = np.copy(empty)
    tilt = np.copy(empty)

    # features from convex hull
    pos_x = np.copy(empty)
    pos_y = np.copy(empty)
    area_um = np.copy(empty)
    deform = np.copy(empty)
    inert_ratio_cvx = np.copy(empty)
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
        event_mask = mask[ii]

        # convex hull
        cont_cvx = np.squeeze(cv2.convexHull(cont_raw))
        mu_cvx = cv2.moments(cont_cvx)
        arc_cvx = np.float64(cv2.arcLength(cont_cvx, True))
        # circ
        circ = 2 * np.sqrt(np.pi * mu_cvx["m00"]) / arc_cvx
        deform[ii] = 1 - circ

        if mu_cvx["m00"] == 0 or mu_raw["m00"] == 0:
            # contour size too small
            continue

        # bounding box
        x, y, w, h = cv2.boundingRect(cont_raw)

        # tilt angle in radians
        oii = 0.5 * np.arctan2(2 * mu_raw['mu11'],
                               mu_raw['mu02'] - mu_raw['mu20'])
        # +PI/2 because relative to channel axis
        tilti = oii + np.pi / 2
        # restrict to interval [0,PI/2]
        tilti = np.mod(tilti, np.pi)
        if tilti > np.pi / 2:
            tilti -= np.pi
        tilt[ii] = np.abs(tilti)

        size_x[ii] = w * pixel_size
        size_y[ii] = h * pixel_size
        pos_x[ii] = mu_cvx["m10"] / mu_cvx["m00"] * pixel_size
        pos_y[ii] = mu_cvx["m01"] / mu_cvx["m00"] * pixel_size
        area_msd[ii] = mu_raw["m00"]
        area_um_raw[ii] = area_raw * pixel_size**2
        area_ratio[ii] = mu_cvx["m00"] / mu_raw["m00"]
        area_um[ii] = mu_cvx["m00"] * pixel_size**2
        aspect[ii] = w / h
        per_um_raw[ii] = arc_raw * pixel_size
        deform_raw[ii] = 1 - 2 * np.sqrt(np.pi * area_raw) / arc_raw
        per_ratio[ii] = arc_raw / arc_cvx

        # inert_ratio_cvx
        if mu_cvx['mu02'] > 0:  # defaults to zero
            inert_ratio_cvx[ii] = np.sqrt(mu_cvx['mu20'] / mu_cvx['mu02'])

        # inert_ratio_raw
        if mu_raw['mu02'] > 0:  # defaults to zero
            inert_ratio_raw[ii] = np.sqrt(mu_raw['mu20'] / mu_raw['mu02'])

        # inert_ratio_prnc
        # rotate contour
        orient = 0.5 * np.arctan2(2 * mu_raw['mu11'],
                                  mu_raw['mu02'] - mu_raw['mu20'])
        cc = np.array(cont_raw, dtype=np.float32, copy=True)  # float32 [sic]
        rho = np.sqrt(cc[:, 0] ** 2 + cc[:, 1] ** 2)
        phi = np.arctan2(cc[:, 1], cc[:, 0]) + orient + np.pi / 2
        cc[:, 0] = rho * np.cos(phi)
        cc[:, 1] = rho * np.sin(phi)
        # compute inertia ratio of rotated contour
        mprnc = cv2.moments(cc)
        root_prnc = mprnc["mu20"] / mprnc["mu02"]
        if root_prnc > 0:  # defaults to zero
            inert_ratio_prnc[ii] = np.sqrt(root_prnc)

        # specify validity
        valid[ii] = True

    return {
        "deform": deform,
        "size_x": size_x,
        "size_y": size_y,
        "pos_x": pos_x,
        "pos_y": pos_y,
        "area_msd": area_msd,
        "area_ratio": area_ratio,
        "area_um": area_um,
        "aspect": aspect,
        "tilt": tilt,
        "inert_ratio_cvx": inert_ratio_cvx,
        "inert_ratio_raw": inert_ratio_raw,
        "inert_ratio_prnc": inert_ratio_prnc,
        "area_um_raw": area_um_raw,
        "per_um_raw": per_um_raw,
        "deform_raw": deform_raw,
        "per_ratio": per_ratio,
        "valid": valid,
    }
