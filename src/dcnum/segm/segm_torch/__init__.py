import importlib
import warnings

try:
    torch = importlib.import_module("torch")
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
        raise ImportError(
            f"Could not find PyTorch {req_maj}.{req_min}")
except ImportError:
    pass
else:
    from .segm_torch_mpo import SegmentTorchMPO  # noqa: F401
    if torch.cuda.is_available():
        from .segm_torch_sto import SegmentTorchSTO  # noqa: F401
