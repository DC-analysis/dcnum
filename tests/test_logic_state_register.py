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
    with warden as (cs, batch_range):
        assert warden.locked
        assert cs.state == "i"
        assert batch_range == (0, 100)
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
    with StateWarden(cs, current_state="i", next_state="s") as (cs2, b_range):
        assert cs is cs2
        assert b_range == (0, 100)
        # cannot acquire a lock when it is already acquired
        start, stop = cs.acquire_task_lock()
        assert start == stop == 0
    assert cs.state == "s"
    start, stop = cs.acquire_task_lock()
    # acquiring new lock for next state must be possible
    assert start == 0
    assert stop == cs.length


def test_state_warden_changes_state_wrong_initial():
    cs = ChunkSlotData((100, 80, 320))
    assert cs.state == "i"
    with pytest.raises(ValueError, match="does not match"):
        with StateWarden(cs, current_state="s", next_state="e"):
            pass
    assert cs.state == "i"
    start, stop = cs.acquire_task_lock()
    # acquiring new lock for next state must be possible
    assert start == 0
    assert stop == cs.length


def test_state_warden_changes_state_wrong_initial_2():
    cs = ChunkSlotData((100, 80, 320))
    assert cs.state == "i"
    with pytest.raises(ValueError, match="does not match"):
        StateWarden(cs, current_state="s", next_state="e")
    assert cs.state == "i"
    start, stop = cs.acquire_task_lock()
    # acquiring new lock for next state must be possible
    assert start == 0
    assert stop == cs.length


def test_state_warden_changes_state_wrong_initial_3():
    cs = ChunkSlotData((100, 80, 320))
    cs.state = "s"
    warden = StateWarden(cs, current_state="s", next_state="e")
    assert warden.batch_size == 100
    assert warden.batch_range == (0, 100)
    start, stop = cs.acquire_task_lock()
    assert start == stop == 0
    cs.state = "i"
    with pytest.raises(ValueError, match="does not match"):
        with warden:
            pass
    assert cs.state == "i"
    start, stop = cs.acquire_task_lock()
    # acquiring new lock for next state must be possible
    assert start == 0
    assert stop == cs.length


def test_state_warden_doubled():
    cs = ChunkSlotData((100, 80, 320))
    cs.state = "s"
    warden = StateWarden(cs, current_state="s", next_state="e")
    assert warden.batch_size == 100
    assert warden.batch_range == (0, 100)

    warden2 = StateWarden(cs, current_state="s", next_state="e")
    assert warden2.batch_size == 0
    assert warden2.batch_range == (0, 0)


def test_state_warden_no_change_on_error():
    cs = ChunkSlotData((100, 80, 320))
    assert cs.state == "i"
    with pytest.raises(ValueError, match="custom test error"):
        with StateWarden(cs, current_state="i", next_state="s"):
            raise ValueError("custom test error")
    assert cs.state == "i"
    start, stop = cs.acquire_task_lock()
    # acquiring new lock for next state must be possible
    assert start == 0
    assert stop == cs.length
