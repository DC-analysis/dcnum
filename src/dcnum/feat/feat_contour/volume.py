from typing import List

import numpy as np


def volume_from_contours(
        contour: List[np.ndarray],
        pos_x: np.ndarray,
        pos_y: np.ndarray,
        pixel_size: float):
    """Calculate the volume of a polygon revolved around an axis

    The volume estimation assumes rotational symmetry.

    Parameters
    ----------
    contour: list of ndarrays of shape (N,2)
        One entry is a 2D array that holds the contour of an event
    pos_x: float ndarray of length N
        The x coordinate(s) of the centroid of the event(s) [µm]
    pos_y: float ndarray of length N
        The y coordinate(s) of the centroid of the event(s) [µm]
    pixel_size: float
        The detector pixel size in µm.

    Returns
    -------
    volume: float ndarray
        volume in um^3

    Notes
    -----
    The computation of the volume is based on a full rotation of the
    upper and the lower halves of the contour from which the
    average is then used.

    The volume is computed radially from the center position
    given by (`pos_x`, `pos_y`). For sufficiently smooth contours,
    such as densely sampled ellipses, the center position does not
    play an important role. For contours that are given on a coarse
    grid, as is the case for deformability cytometry, the center position
    must be given.

    References
    ----------
    - https://de.wikipedia.org/wiki/Kegelstumpf#Formeln
    - Yields identical results to the Matlab script by Geoff Olynyk
      <https://de.mathworks.com/matlabcentral/fileexchange/36525-volrevolve>`_
    """
    # results are stored in a separate array initialized with nans
    v_avg = np.zeros_like(pos_x, dtype=np.float64) * np.nan

    for ii in range(pos_x.shape[0]):
        # If the contour has less than 4 pixels, the computation will fail.
        # In that case, the value np.nan is already assigned.
        cc = contour[ii]
        if cc is not None and cc.shape[0] >= 4:
            # Center contour coordinates with given centroid
            contour_x = cc[:, 0] - pos_x[ii] / pixel_size
            contour_y = cc[:, 1] - pos_y[ii] / pixel_size
            # Switch to r and z to follow notation of vol_revolve
            # (In RT-DC the axis of rotation is x, but for vol_revolve
            # we need the axis vertically)
            contour_r = contour_y
            contour_z = contour_x

            # Compute right volume
            # Which points are at negative r-values (r<0)?
            inx_neg = np.where(contour_r < 0)
            # These points will be shifted up to r=0 directly on the z-axis
            contour_right = np.copy(contour_r)
            contour_right[inx_neg] = 0
            vol_right = vol_revolve(r=contour_right,
                                    z=contour_z,
                                    point_scale=pixel_size)

            # Compute left volume
            # Which points are at positive r-values? (r>0)?
            idx_pos = np.where(contour_r > 0)
            # These points will be shifted down to y=0 to build an x-axis
            contour_left = np.copy(contour_r)
            contour_left[idx_pos] = 0
            # Now we still have negative r values, but vol_revolve needs
            # positive values, so we flip the sign...
            contour_left[:] *= -1
            # ... but in doing so, we have switched to clockwise rotation,
            # and we need to pass the array in reverse order
            vol_left = vol_revolve(r=contour_left[::-1],
                                   z=contour_z[::-1],
                                   point_scale=pixel_size)

            # Compute the average
            v_avg[ii] = (vol_right + vol_left) / 2

    return {"volume": v_avg}


def vol_revolve(r, z, point_scale=1.):
    r"""Calculate the volume of a polygon revolved around the Z-axis

    This implementation yields the same results as the volRevolve
    Matlab function by Geoff Olynyk (from 2012-05-03)
    https://de.mathworks.com/matlabcentral/fileexchange/36525-volrevolve.

    The difference here is that the volume is computed using (a much
    more approachable) implementation using the volume of a truncated
    cone (https://de.wikipedia.org/wiki/Kegelstumpf).

    .. math::

      V = \frac{h \cdot \pi}{3} \cdot (R^2 + R \cdot r + r^2)

    Where :math:`h` is the height of the cone and :math:`r` and
    `R` are the smaller and larger radii of the truncated cone.

    Each line segment of the contour resembles one truncated cone. If
    the z-step is positive (counter-clockwise contour), then the
    truncated cone volume is added to the total volume. If the z-step
    is negative (e.g. inclusion), then the truncated cone volume is
    removed from the total volume.

    Parameters
    ----------
    r: 1d np.ndarray
        radial coordinates (perpendicular to the z axis)
    z: 1d np.ndarray
        coordinate along the axis of rotation
    point_scale: float
        point size in your preferred units; The volume is multiplied
        by a factor of `point_scale**3`.

    Notes
    -----
    The coordinates must be given in counter-clockwise order,
    otherwise the volume will be negative.
    """
    r = np.atleast_1d(r)
    z = np.atleast_1d(z)

    # make sure we have a closed contour
    if (r[-1] != r[0]) or (z[-1] != z[0]):
        # We have an open contour - close it.
        r = np.resize(r, len(r) + 1)
        z = np.resize(z, len(z) + 1)

    rp = r[:-1]

    # array of radii differences: R - r
    dr = np.diff(r)
    # array of height differences: h
    dz = np.diff(z)

    # If we expand the function in the doc string with
    # dr = R - r and dz = h, then we get three terms for the volume
    # (as opposed to four terms in Olynyk's script). Those three terms
    # all resemble area slices multiplied by the z-distance dz.
    a1 = 3 * rp ** 2
    a2 = 3 * rp * dr
    a3 = dr ** 2

    # Note that the formula for computing the volume is symmetric
    # with respect to r and R. This means that it does not matter
    # which sign dr has (R and r are always positive). Since this
    # algorithm assumes that the contour is ordered counter-clockwise,
    # positive dz means adding to the contour while negative dz means
    # subtracting from the contour (see test functions for more details).
    # Conveniently so, dz only appears one time in this formula, so
    # we can take the sign of dz as it is (Otherwise, we would have
    # to take the absolute value of every truncated cone volume and
    # multiply it by np.sign(dz)).
    v = np.pi / 3 * dz * np.abs(a1 + a2 + a3)
    vol = np.sum(v) * point_scale ** 3

    return vol
