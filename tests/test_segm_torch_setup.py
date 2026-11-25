import inspect

import pytest

pytest.importorskip("torch")

from dcnum.segm.segm_torch import torch_setup  # noqa: E402


def test_setup_functions_are_not_generators():
    """LazyLoader just calls these; generators would do nothing"""
    assert not inspect.isgeneratorfunction(torch_setup.setup_torch)
    assert not inspect.isgeneratorfunction(torch_setup.setup_openvino)


def test_setup_torch_applied():
    """Accessing torch via the LazyLoader must configure torch"""
    torch = torch_setup.torch
    assert torch.are_deterministic_algorithms_enabled()
    assert not torch.backends.cudnn.benchmark
    # cuDNN must be enabled for fast convolutions on CUDA devices
    assert torch.backends.cudnn.enabled
    assert torch.get_num_threads() == 1
