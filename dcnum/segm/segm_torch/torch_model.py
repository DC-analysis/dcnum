from __future__ import annotations

import errno
import functools
import hashlib
import io
import json
import logging
import os
import pathlib
import tempfile
import warnings
import zipfile

import numpy as np

from ...meta import paths
from ...common import cpu_count

from .torch_setup import torch, openvino


logger = logging.getLogger(__name__)
if tempfile.tempdir is not None:
    tempdir = pathlib.Path(tempfile.tempdir) / "dcnum_compiled_models"


def check_md5sum(path):
    """Verify the last five characters of the file stem with its MD5 hash"""
    md5 = hashlib.md5(path.read_bytes()).hexdigest()
    if md5[:5] != path.stem.split("_")[-1]:
        raise ValueError(f"MD5 mismatch for {path} ({md5})! Expected the "
                         f"input file to end with '{md5[:5]}{path.suffix}'.")


@functools.cache
def load_model(path_or_name: str | pathlib.Path,
               backend: str | None = None,
               device: str | None = None,
               ):
    """Load a PyTorch model + metadata from a TorchScript jit checkpoint

    Parameters
    ----------
    path_or_name:
        jit checkpoint file; For dcnum, these files have the suffix .dcnm
        and contain a special `_extra_files["dcnum_meta.json"]` extra
        file that can be loaded via `torch.jit.load` (see below).
    device:
        device on which to run the model

    Returns
    -------
    model_jit: torch.jit.ScriptModule
        loaded PyTorch model stored as a TorchScript module
    model_meta: dict
        metadata associated with the loaded model
    """
    device = device or "cpu"
    with torch.inference_mode():
        model_path = retrieve_model_file(path_or_name)

        with open(model_path, "rb") as fd:
            is_version_2 = fd.read(4) == b"DCNM"

        if is_version_2:
            model_call, model_meta = load_model_v2_pt2(
                model_path=model_path,
                backend=backend,
                device=device)
        else:
            if not isinstance(device, (str, torch.device)):
                raise TypeError(
                    f"Model files version 1 only accept string or "
                    f"`torch.device` as `device`, got '{type(device)}'")
            model_call, model_meta = load_model_v1_jit(model_path, device)

        return model_call, model_meta


def load_model_v1_jit(model_path, device: str):
    """Load dcnm model file format version 1 (torch JIT)"""
    torch_device = torch.device(device or "cpu")

    # define an extra files mapping dictionary that loads the model's metadata
    extra_files = {"dcnum_meta.json": ""}
    # load model
    model_jit = torch.jit.load(model_path,
                               _extra_files=extra_files,
                               map_location=torch_device)
    # load model metadata
    model_meta = json.loads(extra_files["dcnum_meta.json"])
    # set model to evaluation mode
    model_jit.eval()
    # optimize for inference on device
    model_jit = torch.jit.optimize_for_inference(model_jit)

    if torch_device.type == "cuda":
        # Estimate the batch size for the current device.
        # In principle, we would be fine with a batch size of 50, but
        # there is a slight improvement in performance when going to
        # higher batch sizes and users will also see the GPU usage
        # in the task manager (to perform a sanity check).
        sy, sx = model_meta["preprocessing"]["image_shape"]

        # We estimate the batch size by determining the memory usage.
        size = 100
        for _ in range(50):
            memdat = {}
            memdat["raw"] = torch.tensor(
                np.zeros((size, 1, sy, sx), dtype=np.float32),
                device=torch_device)
            memdat["model"] = model_jit(memdat["raw"])
            memdat["thresh"] = memdat["model"] > 0.5
            torch.cuda.synchronize()
            free, total = torch.cuda.mem_get_info(torch_device)
            if free / total < 0.1:  # leave a bit of space for other things
                size -= 100
                break
            size += 100
            del memdat
            import gc
            gc.collect()
            torch.cuda.empty_cache()
        # 50 images should fit in any GPU
        size = max(size, 50)
        model_meta["estimated_batch_size_cuda"] = size

    model_meta["format_version"] = "1.0"

    return model_jit, model_meta


def load_model_v2_pt2(model_path: pathlib.Path,
                      backend: str | None = None,
                      device: str = "cpu",
                      ):
    """Load dcnm model file format version 2 (ExportedProgram .pt2)"""
    content = model_path.read_bytes()
    hash = hashlib.md5(content[:-32]).hexdigest().encode()

    # Make sure we have a valid .dcnm model file
    if hash != content[-32:]:
        raise ValueError(f"Not a valid DCNM model file: {model_path}")

    buffer = io.BytesIO()
    buffer.write(content)
    buffer.seek(0)
    buffer.write(b"PK\x03\x04")
    buffer.seek(0)

    with zipfile.ZipFile(buffer) as z:
        with z.open("dcnum_meta.json") as fd:
            model_meta = json.loads(fd.read())
            ident = model_meta["identifier"]

        with z.open(f"{ident}.pt2") as fd2:
            mdat = fd2.read()

    # Fail if the model gets recompiled. This should not be an issue,
    # because dynamic dimensions are defined by guards in the ExportedProgram.
    # torch.compiler.set_stance("fail_on_recompile")

    # load model
    buffer = io.BytesIO()
    buffer.write(mdat)
    buffer.seek(0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        pe = torch.export.load(buffer)

    if backend in [None, "openvino"] and openvino.module_available():
        example = torch.randint(
            low=100,
            high=200,
            size=tuple([model_meta["batch_size"]] + model_meta["image_shape"]),
            dtype=torch.uint8)
        ov_model = openvino.convert_model(pe, example_input=(example,))

        # compile the model for the specified device
        core = openvino.Core()
        ov_device_map = {
            "cpu": "CPU",
        }
        if device not in ov_device_map:
            warnings.warn(f"Openvino device `{device}` not known to dcnum")
            if device in core.available_devices:
                ov_device_map[device] = device
            else:
                raise ValueError(f"Unavailable openvino device '{device}'")
        model = core.compile_model(
            ov_model,
            ov_device_map[device],
            config={
                # Disable hyperthreading, it's not efficient.
                openvino.properties.hint.enable_hyper_threading(): False,
                # Only use the physical cores. Using virtual cores is not
                # efficient. Note that we have to set this here globally,
                # because openvino somehow treats all computations centrally.
                # If we set this to "1", then **all** worker instances will
                # share just one CPU. It's how openvino works.
                openvino.properties.inference_num_threads(): cpu_count(),
            }
        )
        model_meta["mask_func"] = lambda x: next(iter(x.values()))
        model_meta["backend"] = "openvino"
        model_meta["device"] = device
    else:
        # Fallback to default "inductor" compiler
        # https://docs.pytorch.org/docs/main/generated/torch.compile.html#torch.compile
        model = torch.compile(
            pe.module(),
            fullgraph=True,
            dynamic=False,
            backend="inductor",
            # TODO: Pytorch 3.13 supports setting this (avoid recompilations)?
            # dynamic_shapes=(10, 80, 320),
        )
        model_meta["mask_func"] = lambda x: x.detach().cpu().numpy()
        model_meta["backend"] = "inductor"
        model_meta["device"] = device or "cpu"

    return model, model_meta


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
        raise TypeError(
            f"Please pass a string or a path, got {type(path_or_name)}!")

    logger.info(f"Found dcnum model file {path}")
    check_md5sum(path)
    return path
