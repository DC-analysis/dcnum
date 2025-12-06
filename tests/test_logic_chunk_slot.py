from dcnum import logic
from dcnum.logic.slot_register import StateWarden
from dcnum import read

import h5py
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
            with StateWarden(cs, current_state="i", next_state="s"):
                slot_chunk = cs.load(idx)[2]
            assert cs.state == "s"
        else:
            assert csr.length == 16
            with StateWarden(csr, current_state="i", next_state="s"):
                slot_chunk = csr.load(idx)[2]
            assert csr.state == "s"
        assert np.all(slot_chunk == data.image_corr.get_chunk(idx))
        cs.state = "i"


def test_read_bg_off():
    h5path = retrieve_data("fmt-hdf5_cytoshot_full-features_2023.zip")
    path = h5path.with_name("test.hdf5")
    with read.concatenated_hdf5_data(101 * [h5path], path_out=path):
        # This creates HDF5 chunks of size 32. Total length is 400.
        # There will be one "remainder" chunk of size `400 % 32 = 16`.
        pass

    with h5py.File(path, "a") as h5:
        size = h5.attrs["experiment:event count"]
        assert size == 4040
        bg_off = np.linspace(-.23, .21, size)
        # Add zeros to make sure features are still computed
        bg_off[3:5] = 0
        h5["events/bg_off"] = bg_off
        image = h5["events/image"][:]

    job = logic.DCNumPipelineJob(path_in=path)
    data = read.HDF5Data(path)
    assert data.image_chunk_size == 1000
    assert data.image.chunk_size == 1000
    # Normal chunk
    cs = logic.ChunkSlot(job=job, data=data)
    # Remainder chunk
    csr = logic.ChunkSlot(job=job, data=data, is_remainder=True)

    assert cs.length == 1000
    assert csr.length == 40

    for chunk_index in range(4):
        cs.state = "i"
        c_image, _, _, c_bg_off = cs.load(chunk_index)
        assert np.all(cs.bg_off == c_bg_off)
        assert np.all(cs.bg_off
                      == bg_off[chunk_index*1000:(chunk_index+1)*1000])
        # Also test the image data, while we are at it.
        assert np.all(cs.image == c_image)
        assert np.all(cs.image
                      == image[chunk_index*1000:(chunk_index+1)*1000])

    cr_image, _, _, cr_bg_off = csr.load(4)
    assert np.all(csr.bg_off == cr_bg_off)
    assert np.all(csr.bg_off == bg_off[-40:])
    assert np.all(csr.image == cr_image)
    assert np.all(csr.image == image[-40:])
