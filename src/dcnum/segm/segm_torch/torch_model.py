import errno
import functools
import hashlib
import json
import logging
import os
import pathlib

import torch

from ...meta import paths


logger = logging.getLogger(__name__)


def check_md5sum(path):
    """Verify the last five characters of the file stem with its MD5 hash"""
    md5 = hashlib.md5(path.read_bytes()).hexdigest()
    if md5[:5] != path.stem.split("_")[-1]:
        raise ValueError(f"MD5 mismatch for {path} ({md5})! Expected the "
                         f"input file to end with '{md5[:5]}{path.suffix}'.")


@functools.cache
def load_model(path_or_name, device):
    """Load a PyTorch model + metadata from a TorchScript jit checkpoint

    Parameters
    ----------
    path_or_name: str or pathlib.Path
        jit checkpoint file; For dcnum, these files have the suffix .dcnm
        and contain a special `_extra_files["dcnum_meta.json"]` extra
        file that can be loaded via `torch.jit.load` (see below).
    device: str or torch.device
        device on which to run the model

    Returns
    -------
    model_jit: torch.jit.ScriptModule
        loaded PyTorch model stored as a TorchScript module
    model_meta: dict
        metadata associated with the loaded model
    """
    model_path = retrieve_model_file(path_or_name)
    # define an extra files mapping dictionary that loads the model's metadata
    extra_files = {"dcnum_meta.json": ""}
    # load model
    model_jit = torch.jit.load(model_path,
                               _extra_files=extra_files,
                               map_location=device)
    # load model metadata
    model_meta = json.loads(extra_files["dcnum_meta.json"])
    # set model to evaluation mode
    model_jit.eval()
    # optimize for inference on device
    model_jit = torch.jit.optimize_for_inference(model_jit)
    return model_jit, model_meta


@functools.cache
def retrieve_model_file(path_or_name):
    """Retrieve a dcnum torch model file

    If a path to a model is given, then this path is returned directly.
    If a file name is given, then look for the file with
    :func:`dcnum.meta.paths.find_file` using the "torch_model_file"
    topic.
    """
    # Did the user already pass a path?
    if isinstance(path_or_name, pathlib.Path):
        if path_or_name.exists():
            path = path_or_name
        else:
            try:
                return retrieve_model_file(path_or_name.name)
            except BaseException:
                raise FileNotFoundError(errno.ENOENT,
                                        os.strerror(errno.ENOENT),
                                        str(path_or_name))
    elif isinstance(path_or_name, str):
        name = path_or_name.strip()
        # We now have a string for a filename, and we have to figure out what
        # the path is. There are several options, including cached files.
        if pathlib.Path(name).exists():
            path = pathlib.Path(name)
        else:
            path = paths.find_file("torch_model_files", name)
    else:
        raise ValueError(
            f"Please pass a string or a path, got {type(path_or_name)}!")

    logger.info(f"Found dcnum model file {path}")
    check_md5sum(path)
    return path
