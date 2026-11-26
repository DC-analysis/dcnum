"""Lazy loading and setup of PyTorch and OpenVINO"""
from __future__ import annotations

import os
import pathlib
import platform
import warnings

from ...common import LazyLoader

# Note: Any changes of the environment that we are making here affect
# all submodules, because the Python import system imports the parent
# modules before importing submodules (even if modules are imported with
# "from parent import child").

# https://docs.nvidia.com/cuda/cublas/#results-reproducibility
# Make sure that all computations on the GPU with cublas are reproducible.
# - ":16:8" may limit overall performance
# - ":4096:8" increases memory footprint by 24MB
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def setup_openvino():
    """Disable OpenVINO telemetry (opt-out); called before the import

    Note that this must be a regular function and not a generator,
    because :class:`.LazyLoader` just calls it.
    """
    # Disable telemetry
    try:
        import openvino_telemetry.main as tm  # type: ignore
        opt_in_checker = tm.OptInChecker()
        opt_in_checker.update_result(tm.ConsentCheckResult.DECLINED)
    except ImportError:
        # Package openvino_telemetry is not available
        pf = platform.system()
        if pf == "Windows":
            dir = pathlib.Path(os.path.expandvars("$LOCALAPPDATA"))
            subdir = "Intel Corporation"
        elif pf in ["Linux", "Darwin"]:
            dir = pathlib.Path.home()
            subdir = "intel"
        else:
            dir = subdir = None

        try:
            if dir is None or subdir is None or not dir.exists():
                raise OSError(f"Unknown telemetry directory for {pf}")
            consent_file = dir / subdir / "openvino_telemetry"
            consent_file.parent.mkdir(parents=True, exist_ok=True)
            consent_file.write_text("0")
        except OSError:
            print("Failed to opt out of openvino telemetry")


def setup_torch(torch):
    """Configure PyTorch; called after the import

    PyTorch is configured for reproducible, deterministic,
    single-threaded computation and a warning is issued if the
    PyTorch version is older than 2.2.

    Note that this must be a regular function and not a generator,
    because :class:`.LazyLoader` just calls it.
    """
    # https://docs.pytorch.org/docs/stable/notes/numerical_accuracy.html
    # Tell pytorch to only use deterministic algorithms.
    torch.use_deterministic_algorithms(True)
    # Disable CUDNN benchmarking for inference (just to be sure).
    torch.backends.cudnn.benchmark = False
    # Keep CUDNN enabled. Without CUDNN, PyTorch computes convolutions on
    # CUDA devices with a slow im2col + GEMM fallback (4x slower for the
    # U-Net models). TorchScript models optimized with
    # `torch.jit.optimize_for_inference` always used CUDNN for their fused
    # convolution + ReLU operations, regardless of this setting. CUDNN is
    # deterministic, because of `use_deterministic_algorithms` above.
    torch.backends.cudnn.enabled = True
    # We are parallelizing with mp.multiprocessing and do not want
    # pytorch to parallelize for us.
    torch.set_num_threads(1)

    req_maj = 2
    req_min = 2
    ver_tuple = torch.__version__.split(".")
    act_maj = int(ver_tuple[0])
    act_min = int(ver_tuple[1])
    if act_maj < req_maj or (act_maj == req_maj and act_min < req_min):
        warnings.warn(f"Your PyTorch version {act_maj}.{act_min} is "
                      f"not supported, please update to at least "
                      f"{req_maj}.{req_min} to use dcnum's PyTorch"
                      f"segmenters")


torch = LazyLoader("torch", action_after_import=setup_torch)
openvino = LazyLoader("openvino", action_before_import=setup_openvino)
