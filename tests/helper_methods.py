import pathlib
import tempfile
import zipfile

import numpy as np


class MockImageData:
    mask = np.array([
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 1, 1, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0, 0],  # filled, 1
        [0, 0, 1, 1, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 1, 1, 1],  # border, 2
        [0, 0, 0, 0, 0, 1, 1, 1],
        [0, 1, 1, 1, 0, 0, 0, 0],
        [0, 0, 1, 1, 1, 0, 0, 0],  # other, 3
        [0, 0, 1, 1, 1, 0, 0, 0],
        [0, 0, 1, 1, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
    ], dtype=bool)

    def get_chunk(self, chunk_index):
        image = np.array(-(10 + chunk_index) * self.mask, dtype=np.int16)
        chunk = np.stack([image] * 100, dtype=np.int16)
        return chunk

    def get_chunk_slice(self, chunk_index):
        return slice(0, 100)


def calltracker(func):
    """Decorator to track how many times a function is called"""

    def wrapped(*args, **kwargs):
        wrapped.calls += 1
        return func(*args, **kwargs)

    wrapped.calls = 0
    return wrapped


def extract_data(zip_file):
    """Extract zip file content from data directory, return directory"""
    zpath = pathlib.Path(__file__).resolve().parent / "data" / zip_file
    # unpack
    arc = zipfile.ZipFile(str(zpath))

    # extract all files to a temporary directory
    edest = tempfile.mkdtemp(prefix=zpath.name)
    arc.extractall(edest)
    return pathlib.Path(edest)


def find_data(path):
    """Find .avi and .rtdc data files in a directory"""
    path = pathlib.Path(path)
    avifiles = [r for r in path.rglob("*.avi") if r.is_file()]
    rtdcfiles = [r for r in path.rglob("*.rtdc") if r.is_file()]
    files = [pathlib.Path(ff) for ff in rtdcfiles + avifiles]
    return files


def find_model(path):
    """Find .ckp files in a directory"""
    path = pathlib.Path(path)
    jit_files = [r for r in path.rglob("*.dcnm") if r.is_file()]
    files = [pathlib.Path(ff) for ff in jit_files]
    return files


def retrieve_data(zip_file):
    """Extract contents of data zip file and return data files
    """
    # extract all files to a temporary directory
    edest = extract_data(zip_file)

    # Load RT-DC dataset
    # find tdms files
    datafiles = find_data(edest)

    if len(datafiles) == 1:
        datafiles = datafiles[0]

    return datafiles


def retrieve_model(zip_file):
    """Extract contents of model zip file and return model paths
    """
    # extract all files to a temporary directory
    edest = extract_data(zip_file)

    # find model checkpoint paths
    modelpaths = find_model(edest)

    if len(modelpaths) == 1:
        modelpaths = modelpaths[0]

    return modelpaths
