""" Code copied from skimage.morphology

License: BSD-3-Clause

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:
1. Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.
3. Neither the name of the University nor the names of its contributors
   may be used to endorse or promote products derived from this software
   without specific prior written permission.
.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE HOLDERS OR
CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""
import numpy as np


def disk(radius, dtype=np.uint8, *, strict_radius=True, decomposition=None):
    """Generates a flat, disk-shaped footprint.

    A pixel is within the neighborhood if the Euclidean distance between
    it and the origin is no greater than radius (This is only approximately
    True, when `decomposition == 'sequence'`).

    Parameters
    ----------
    radius : int
        The radius of the disk-shaped footprint.

    Other Parameters
    ----------------
    dtype : data-type, optional
        The data type of the footprint.
    strict_radius : bool, optional
        If False, extend the radius by 0.5. This allows the circle to expand
        further within a cube that remains of size ``2 * radius + 1`` along
        each axis. This parameter is ignored if decomposition is not None.
    decomposition : {None, 'sequence', 'crosses'}, optional
        This method was copied from skimage.morphology and only supports
        `decomposition=None`.

    Returns
    -------
    footprint : ndarray
        The footprint where elements of the neighborhood are 1 and 0 otherwise.
    """
    if decomposition is None:
        L = np.arange(-radius, radius + 1)
        X, Y = np.meshgrid(L, L)
        if not strict_radius:
            radius += 0.5
        return np.array((X**2 + Y**2) <= radius**2, dtype=dtype)
    else:
        raise NotImplementedError("Please use `skimage.morphology.disk`")
