import multiprocessing as mp

from dcnum.logic.chunk_slot_data import ChunkSlotData
from dcnum.read import HDF5Data
from dcnum.logic.slot_register import StateWarden, SlotRegister
from dcnum.logic.job import DCNumPipelineJob

import pytest

from helper_methods import retrieve_data


mp_spawn = mp.get_context("spawn")


def slot_register_reserve_slot_for_task():
    path = retrieve_data("fmt-hdf5_cytoshot_full-features_2023.zip")
    hd = HDF5Data(path)
    assert "image" in hd

    print("Setting up pipeline job")
    job = DCNumPipelineJob(path_in=path)
    slot_register = SlotRegister(job=job, data=hd, num_slots=1)

    warden = slot_register.reserve_slot_for_task(current_state="i",
                                                 next_state="s")
    with warden as cs:
        assert warden.locked
        assert cs.state == "i"
    assert cs.state == "s"

    # We only have one slot, this means requesting the same thing will
    # not work.
    warden2 = slot_register.reserve_slot_for_task(current_state="i",
                                                  next_state="s")
    assert warden2 is None

    warden3 = slot_register.reserve_slot_for_task(current_state="s",
                                                  next_state="e")
    assert warden3 is not None


def test_state_warden_changes_state():
    cs = ChunkSlotData((100, 80, 320))
    assert cs.state == "i"
    with StateWarden(cs, current_state="i", next_state="s"):
        assert not cs.task_lock.acquire(block=False)
    assert cs.state == "s"
    assert cs.task_lock.acquire(block=False)


def test_state_warden_changes_state_wrong_initial():
    cs = ChunkSlotData((100, 80, 320))
    assert cs.state == "i"
    with pytest.raises(ValueError, match="does not match"):
        with StateWarden(cs, current_state="s", next_state="e"):
            pass
    assert cs.state == "i"
    assert cs.task_lock.acquire(block=False)


def test_state_warden_changes_state_wrong_initial_2():
    cs = ChunkSlotData((100, 80, 320))
    assert cs.state == "i"
    with pytest.raises(ValueError, match="does not match"):
        StateWarden(cs, current_state="s", next_state="e")
    assert cs.state == "i"
    assert cs.task_lock.acquire(block=False)


def test_state_warden_changes_state_wrong_initial_3():
    cs = ChunkSlotData((100, 80, 320))
    cs.state = "s"
    warden = StateWarden(cs, current_state="s", next_state="e")
    assert not cs.task_lock.acquire(block=False)
    cs.state = "i"
    with pytest.raises(ValueError, match="does not match"):
        with warden:
            pass
    assert cs.state == "i"
    assert cs.task_lock.acquire(block=False)


def test_state_warden_no_change_on_error():
    cs = ChunkSlotData((100, 80, 320))
    assert cs.state == "i"
    with pytest.raises(ValueError, match="custom test error"):
        with StateWarden(cs, current_state="i", next_state="s"):
            raise ValueError("custom test error")
    assert cs.state == "i"
    assert cs.task_lock.acquire(block=False)
