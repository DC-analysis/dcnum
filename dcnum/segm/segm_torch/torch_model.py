"""Loading of dcnum model files (.dcnm)"""

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
from typing import Any
import warnings
import zipfile

import numpy as np

from ...meta import paths
from ...common import cpu_count

from .torch_setup import torch, openvino


logger = logging.getLogger(__name__)


class WiringUnavailableError(ValueError):
    """None of the wiring options of a model file can be used"""


if tempfile.tempdir is not None:
    tempdir = pathlib.Path(tempfile.tempdir) / "dcnum_compiled_models"

#: Model cache with keys "{filename}::{backend}::{device}"
cache_model = {}
#: Model metadata cache with keys "{filename}::{backend}::{device}"
cache_model_meta = {}


def check_md5sum(path):
    """Verify the last five characters of the file stem with its MD5 hash"""
    md5 = hashlib.md5(path.read_bytes()).hexdigest()
    if md5[:5] != path.stem.split("_")[-1]:
        raise ValueError(f"MD5 mismatch for {path} ({md5})! Expected the "
                         f"input file to end with '{md5[:5]}{path.suffix}'.")


def get_model_meta(path_or_name: str | pathlib.Path,
                   backend: str | None = None,
                   device: str | None = None,
                   ) -> dict[str, Any]:
    """Return model metadata given the model file and backend/device conditions

    If the backend/device are not or only partially given, return the best
    options for running the model in the metadata. E.g. if the backend is
    set to "cudagraphs", then it follows that the device should be "cuda",
    given an installed NVIDIA GPU.

    Notes
    -----
    The metadata are cached in the :const:`cache_model_meta` dictionary.
    """
    # Make sure 'device' is a string
    if isinstance(device, torch.device):
        device = device.type
    # Use first CUDA device if not specified
    if device and device.startswith("cuda") and not device.count(":"):
        device = "cuda:0"
    # Fetch the model path
    model_path = retrieve_model_file(path_or_name)
    # setup the cache key
    cache_key = f"{model_path.name}::{backend}::{device}"

    # Either get the model metadata from the cache or retrieve them.
    if cache_key in cache_model_meta:
        model_meta = cache_model_meta[cache_key]
    else:
        model_meta = {
            "path": model_path,
            "wiring_options": [],  # this is defined in the model metadata
        }

        # Determine the dcnm file format version
        with open(model_path, "rb") as fd:
            if fd.read(4) == b"DCNM":
                dcnm_format_version = "2.0"
            else:
                dcnm_format_version = "1.0"
            model_meta["dcnm_format_version"] = dcnm_format_version

        if dcnm_format_version == "1.0":
            # define an extra files mapping dictionary that loads the metadata
            extra_files = {"dcnum_meta.json": ""}
            # load model
            _ = torch.jit.load(model_path,
                               _extra_files=extra_files,
                               map_location=torch.device("cpu"))
            # load model metadata
            model_meta.update(json.loads(extra_files["dcnum_meta.json"]))
            model_meta["backend"] = "torch.jit"
            if not model_meta["wiring_options"]:
                model_meta["wiring_options"] += [
                    {"backend": "torch.jit", "device": "cpu"},
                ]
                # check whether CUDA is available.
                if torch.cuda.is_available():
                    model_meta["wiring_options"].insert(
                        0, {"backend": "torch.jit", "device": "cuda"})
                    model_meta["wiring_options"].insert(
                        1, {"backend": "torch.jit", "device": "cuda:0"})
                    model_meta["wiring_options"].insert(
                        2, {"backend": "torch.jit", "device": "cuda:1"})

        elif dcnm_format_version == "2.0":
            # Extract the model metadata
            content = model_path.read_bytes()
            hash = hashlib.md5(content[:-32]).hexdigest().encode()
            # Make sure we have a valid .dcnm model file
            if hash != content[-32:]:
                raise ValueError(f"Not a valid DCNM model file: {model_path}")
            # prepare buffer
            buffer = io.BytesIO()
            buffer.write(content)
            buffer.seek(0)
            buffer.write(b"PK\x03\x04")
            buffer.seek(0)
            with zipfile.ZipFile(buffer) as z, z.open("dcnum_meta.json") as fd:
                model_meta.update(json.loads(fd.read()))
        else:
            assert False

        # The available backend/device wiring are baked into the model file.
        # Each entry consists of a dictionary with keys "device" and
        # "backend", ordered according to efficiency/suggestion decreasing.
        wiring_options: list[dict[str, str]] = model_meta["wiring_options"]
        # Determine the correct wiring based on the input.
        # Model files may only be used with the wiring options they list,
        # unless the user explicitly asks for a backend and device. Options
        # that cannot be used in the current environment (e.g. no GPU or
        # no Triton) are skipped.
        check_backend = True
        if backend is None and device is None:
            # we take the first available item from the wiring options
            candidates = wiring_options
        elif backend is None and device is not None:
            candidates = [o for o in wiring_options if o["device"] == device]
        elif backend is not None and device is None:
            # The user asked for this backend, so we only make sure
            # that the device exists.
            candidates = [o for o in wiring_options
                          if o["backend"] == backend]
            check_backend = False
        else:
            # This will also allow users to override if this configuration is
            # not baked into the model.
            candidates = [{"backend": backend, "device": device}]
            check_backend = None  # no checks at all

        if not candidates:
            if backend is None and device is None:
                raise ValueError(
                    f"The model {model_path.name} does not have wiring "
                    f"options specified. Please provide 'device' and "
                    f"'backend' manually.")
            else:
                raise ValueError(
                    f"The current wiring options {backend=}, {device=} "
                    f"are either not supported or invalid for the "
                    f"model {model_path.name}.")

        if check_backend is None:
            wiring_options = candidates
        else:
            wiring_options = []
            reasons = []
            for opt in candidates:
                reason = get_wiring_unavailable_reason(
                    opt, check_backend=check_backend)
                if reason is None:
                    wiring_options.append(opt)
                else:
                    reasons.append(f"'{opt['backend']}' on "
                                   f"'{opt['device']}' ({reason})")
            if not wiring_options:
                raise WiringUnavailableError(
                    f"None of the wiring options of the model "
                    f"{model_path.name} can be used in this environment: "
                    + "; ".join(reasons)
                    + ". Please install the missing dependency or specify "
                      "'backend' and 'device' manually.")

        model_meta["backend"] = wiring_options[0]["backend"]
        model_meta["device"] = wiring_options[0]["device"]

        # Update the cache
        cache_model_meta[cache_key] = model_meta

    # We might have an updated cache key (e.g. if device was `None`)
    cache_key_2 = (f"{model_path.name}::"
                   f"{model_meta['backend']}::{model_meta['device']}")
    if cache_key_2 not in cache_model_meta:
        cache_model_meta[cache_key_2] = model_meta

    return model_meta


@functools.cache
def has_cpp_compiler() -> bool:
    """Return True if a C++ compiler for TorchInductor on the CPU exists"""
    from torch._inductor import cpp_builder
    try:
        cpp_builder.get_cpp_compiler()
    except BaseException:
        return False
    else:
        return True


@functools.cache
def has_triton() -> bool:
    """Return True if Triton is available for TorchInductor on the GPU

    Triton is an optional dependency. It is installed with the CUDA
    builds of PyTorch on Linux. On Windows, it can be installed with
    the community package "triton-windows" (the version must match the
    PyTorch version).
    """
    try:
        from torch.utils._triton import has_triton as torch_has_triton
    except ImportError:
        import importlib.util
        return importlib.util.find_spec("triton") is not None
    else:
        return torch_has_triton()


def get_wiring_unavailable_reason(wiring: dict[str, str],
                                  check_backend: bool = True) -> str | None:
    """Return why a backend/device wiring option cannot be used

    This checks whether the CUDA device exists and, if `check_backend`
    is True, whether OpenVINO is installed for the "openvino" backend
    and whether the "inductor" backend can generate code for the device
    (Triton on CUDA devices, a C++ compiler on the CPU).

    Returns None if the wiring option can be used.
    """
    backend = wiring["backend"]
    device = str(wiring["device"])
    if device.startswith("cuda"):
        if not torch.cuda.is_available():
            return "CUDA is not available"
        index = int(device.split(":")[1]) if device.count(":") else 0
        if index >= torch.cuda.device_count():
            return f"there is no CUDA device with index {index}"
    if not check_backend:
        return None
    if backend == "inductor":
        if device.startswith("cuda") and not has_triton():
            return "Triton is not installed"
        if device == "cpu" and not has_cpp_compiler():
            return "no C++ compiler found"
    if backend == "openvino" and not openvino.module_available():
        return "OpenVINO is not installed"
    return None


def is_wiring_available(wiring: dict[str, str],
                        check_backend: bool = True) -> bool:
    """Check whether a backend/device wiring option can be used

    See :func:`get_wiring_unavailable_reason` for details.
    """
    return get_wiring_unavailable_reason(
        wiring, check_backend=check_backend) is None


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

    Notes
    -----
    The models are cached in the :const:`cache_model` dictionary.
    """
    with torch.inference_mode():
        model_meta = get_model_meta(
            path_or_name,
            backend=backend,
            device=device,
        )

        cache_key = (f"{model_meta['path'].name}::"
                     f"{model_meta['backend']}::{model_meta['device']}")

        if cache_key in cache_model:
            return cache_model[cache_key]
        else:
            if model_meta["dcnm_format_version"] == "2.0":
                model_call, model_meta = load_model_v2_pt2(model_meta)
            else:
                if not isinstance(device, (str, torch.device)):
                    raise TypeError(
                        f"Model files version 1 only accept string or "
                        f"`torch.device` as `device`, got '{type(device)}'")
                model_call, model_meta = load_model_v1_jit(model_meta)
            cache_model[cache_key] = model_call, model_meta

        return model_call, model_meta


def load_model_v1_jit(model_meta):
    """Load dcnm model file format version 1 (torch JIT)"""
    torch_device = torch.device(model_meta["device"])

    model_jit = torch.jit.load(model_meta["path"],
                               map_location=torch_device)
    # set model to evaluation mode
    model_jit.eval()
    # optimize for inference on device
    model_jit = torch.jit.optimize_for_inference(model_jit)

    # This is time-consuming and is thus not part of `get_model_meta()`
    if torch_device.type.startswith("cuda"):
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
        model_meta["batch_size_recommended"] = size

    return model_jit, model_meta


def load_model_v2_pt2(model_meta: dict[str, Any]):
    """Load dcnm model file format version 2 (ExportedProgram .pt2)

    Notes
    -----
    The ``model_meta`` dict should contain the "backend" key. If "openvino"
    is installed, then the backend "openvino" is available.
    Torch's eager mode is available via the "torch.eager" backend (strictly
    speaking, this is not a backend at all, it's just the torch default
    execution path op-by-op on the available hardware).
    TorchScript models are loaded with the "torch.jit" backend.
    In all other cases, the backend is passed to ``torch.compile``.

    """
    content = model_meta["path"].read_bytes()
    buffer = io.BytesIO()
    buffer.write(content)
    buffer.seek(0)
    buffer.write(b"PK\x03\x04")
    buffer.seek(0)

    with zipfile.ZipFile(buffer) as z:
        ident = model_meta["identifier"]
        with z.open(f"{ident}.pt2") as fd2:
            mdat = fd2.read()

    # Fail if the model gets recompiled. This should not be an issue,
    # because dynamic dimensions are defined by guards in the ExportedProgram.
    # torch.compiler.set_stance("fail_on_recompile")

    backend = model_meta["backend"]
    device = model_meta["device"]

    # load model
    buffer = io.BytesIO()
    buffer.write(mdat)
    buffer.seek(0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        mff = model_meta.get("model_file_format")
        if mff == "ExportedProgram":
            mod = torch.export.load(buffer)
        elif mff == "TorchScript":
            mod = torch.jit.load(buffer, map_location=device)
            # set model to evaluation mode
            mod.eval()
            # optimize for inference on device
            mod = torch.jit.optimize_for_inference(mod)
            if backend != "torch.jit":
                raise ValueError(
                    f"Expected 'torch.jit' backend, got '{backend}'")
        else:
            raise ValueError(f"Unknown model file format '{mff}'")

    if backend == "torch.jit":
        model = mod
        model_meta["mask_func"] = lambda x: x.detach().cpu().numpy()
    elif backend == "torch.eager":
        # In eager mode, nothing is compiled at all.
        model = get_optimized_graph_module(mod, model_meta)
        model_meta["mask_func"] = lambda x: x.detach().cpu().numpy()
    elif backend == "openvino":
        assert openvino.module_available()
        model_meta["batch_size"] = batch_size = 10
        example = torch.randint(
            low=100,
            high=200,
            size=tuple([batch_size] + model_meta["image_shape"]),
            dtype=torch.uint8)
        ov_model = openvino.convert_model(mod, example_input=(example,))

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
    else:
        # https://docs.pytorch.org/docs/main/generated/torch.compile.html#torch.compile
        compile_kwargs = {"fullgraph": True,
                          "dynamic": False,
                          "backend": backend,
                          }
        if backend == "inductor":
            # Let `torch.compile` decide which dimensions are dynamic,
            # so that it respects `maybe_mark_dynamic` below.
            compile_kwargs["dynamic"] = None
            # By default, inductor converts the convolutions to the
            # channels-last memory format. For U-Net models with a
            # single input channel, cuDNN then selects convolution
            # kernels that are about three times slower.
            compile_kwargs["options"] = {"layout_optimization": False}
        model_compiled = torch.compile(
            get_optimized_graph_module(mod, model_meta), **compile_kwargs)
        if backend == "inductor":
            def model(images):
                # Compile the model once for all batch sizes. The last
                # chunk of a dataset (and its batches on the GPU) is
                # usually smaller than the others, and with static shapes,
                # inductor compiles the model again for every new batch
                # size. Only the batch size is marked dynamic, all other
                # shapes (images and weights) stay static (`dynamic=True`
                # would make them dynamic, too, which makes the generated
                # code slower on the CPU).
                torch._dynamo.maybe_mark_dynamic(images, 0)
                return model_compiled(images)
        else:
            model = model_compiled
        model_meta["mask_func"] = lambda x: x.detach().cpu().numpy()

    return model, model_meta


def get_optimized_graph_module(exported_program, model_meta):
    """Return the module of an ExportedProgram optimized for inference

    On CUDA devices, convolutions followed by a ReLU are replaced with
    cuDNN's fused convolution + bias + ReLU operation (see
    :mod:`.torch_graph_opt`), as :func:`torch.jit.optimize_for_inference`
    does for TorchScript models. The number of applied optimizations is
    stored in ``model_meta["graph_optimizations"]``.
    """
    gm = exported_program.module()
    device = str(model_meta["device"])
    if (device.startswith("cuda")
            # custom operators require torch 2.4
            and hasattr(torch.library, "custom_op")):
        from . import torch_graph_opt  # registers a custom operator
        model_meta["graph_optimizations"] = \
            torch_graph_opt.optimize_for_inference(gm, device=device)
    else:
        model_meta["graph_optimizations"] = {}
    return gm


@functools.cache
def retrieve_model_file(path_or_name):
    """Retrieve a dcnum torch model file

    If a path to a model is given, then this path is returned directly.
    If the path does not exist, the file name of the path is looked up
    as described below.
    If a file name is given, then look for the file with
    :func:`dcnum.meta.paths.find_file` using the "torch_model_files"
    topic.

    The MD5 sum of the model file is checked (see :func:`check_md5sum`)
    and the result is cached.

    Raises
    ------
    FileNotFoundError
        If a path is given that cannot be found
    KeyError
        If a file name is given that cannot be found in the search paths
    TypeError
        If ``path_or_name`` is neither a string nor a path
    ValueError
        If the MD5 sum of the model file does not match its file name
    """
    # Did the user already pass a path?
    if isinstance(path_or_name, pathlib.Path):
        if path_or_name.exists():
            path = path_or_name
        else:
            try:
                return retrieve_model_file(path_or_name.name)
            except KeyboardInterrupt:
                raise
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
