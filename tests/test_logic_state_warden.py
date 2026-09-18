import multiprocessing as mp

from dcnum.logic.chunk_slot_data import ChunkSlotData
from dcnum.logic.slot_register import StateWarden

import pytest


mp_spawn = mp.get_context("spawn")


def test_state_warden_changes_state():
    cs = ChunkSlotData((100, 80, 320))
    assert cs.state == "i"
    with StateWarden(cs, current_state="i", next_state="s") as (cs2, b_range):
        assert cs is cs2
        assert b_range == (0, 100)
        # cannot acquire a lock when it is already acquired
        start, stop = cs.acquire_task_lock("i")
        assert start == stop == 0
    assert cs.state == "s"
    start, stop = cs.acquire_task_lock("s")
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
    start, stop = cs.acquire_task_lock("i")
    # acquiring new lock for next state must be possible
    assert start == 0
    assert stop == cs.length


def test_state_warden_changes_state_wrong_initial_2():
    cs = ChunkSlotData((100, 80, 320))
    assert cs.state == "i"
    sw = StateWarden(cs, current_state="s", next_state="e")
    assert sw.batch_size == 0
    assert cs.state == "i"
    start, stop = cs.acquire_task_lock("i")
    # acquiring new lock for next state must be possible
    assert start == 0
    assert stop == cs.length


def test_state_warden_changes_state_wrong_initial_3():
    cs = ChunkSlotData((100, 80, 320))
    cs.state = "s"
    warden = StateWarden(cs, current_state="s", next_state="e")
    assert warden.batch_size == 100
    assert warden.batch_range == (0, 100)
    start, stop = cs.acquire_task_lock("s")
    assert start == stop == 0
    cs.state = "i"
    with pytest.raises(ValueError, match="does not match"):
        with warden:
            pass
    assert cs.state == "i"
    start, stop = cs.acquire_task_lock("i")
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
    start, stop = cs.acquire_task_lock("i")
    # acquiring new lock for next state must be possible
    assert start == 0
    assert stop == cs.length
