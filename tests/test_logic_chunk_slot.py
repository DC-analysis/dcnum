from dcnum import logic
from dcnum import read

import numpy as np

import pytest

from helper_methods import retrieve_data


@pytest.mark.parametrize("chunk_size", (32, 64, 1000))
def test_basic_chunk_slot(chunk_size):
    h5path = retrieve_data("fmt-hdf5_cytoshot_full-features_2023.zip")
    path = h5path.with_name("test.hdf5")
    with read.concatenated_hdf5_data(10 * [h5path], path_out=path):
        # This creates HDF5 chunks of size 32. Total length is 400.
        # There will be one "remainder" chunk of size `400 % 32 = 16`.
        pass

    job = logic.DCNumPipelineJob(path_in=path)
    data = read.HDF5Data(path, image_chunk_size=chunk_size)
    chunk_size_act = min(chunk_size, len(data.image))
    assert data.image_chunk_size == chunk_size
    assert data.image.chunk_size == chunk_size_act
    # Normal chunk
    cs = logic.ChunkSlot(job=job, data=data)
    # Remainder chunk
    csr = logic.ChunkSlot(job=job, data=data, is_remainder=True)
    assert cs.state == "i"
    for idx in range(data.image.num_chunks):
        if data.image.get_chunk_size(idx) == chunk_size_act:
            slot_chunk = cs.load(idx)[2]
            assert cs.state == "s"
        else:
            assert csr.length == 16
            slot_chunk = csr.load(idx)[2]
            assert csr.state == "s"
        assert np.all(slot_chunk == data.image_corr.get_chunk(idx))
        cs.state = "i"
