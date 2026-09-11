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
    # Executed before import.
    # Disable telemetry
    try:
        import openvino_telemetry.main as tm
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

        if dir is not None and subdir is not None and dir.exists():
            consent_file = dir / subdir / "openvino_telemetry"
            consent_file.write_text("0")
        else:
            print("Failed to opt out of openvino telemetry")

    yield
    # Executed after import.


def setup_torch(torch):
    # Executed before import.
    yield
    # Executed after import.
    # REPRODUCIBILITY: All of these settings, including CUBLAS_WORKSPACE_CONFIG
    # above resulted in a segmentation performance hit of about 10% for an
    # NVIDIA RTX 2050 Laptop.
    # https://docs.pytorch.org/docs/stable/notes/numerical_accuracy.html
    # Tell pytorch to only use deterministic algorithms.
    torch.use_deterministic_algorithms(True)
    # Disable CUDNN benchmarking for inference (just to be sure).
    torch.backends.cudnn.benchmark = False
    # Disable CUDNN altogether (this will free some GPU memory).
    torch.backends.cudnn.enabled = False
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
