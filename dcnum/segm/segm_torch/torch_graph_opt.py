"""Inference graph optimizations for exported models (dcnm format 2.0)

Note that this module registers a custom operator when it is imported
and thus imports PyTorch. Only import it when a model is loaded.

:func:`fuse_conv_relu` replaces a 2D convolution followed by a ReLU with
the custom operator ``dcnum::conv2d_relu``, which uses cuDNN's fused
convolution + bias + ReLU kernel on CUDA devices. This is the kernel that
:func:`torch.jit.optimize_for_inference` uses for TorchScript models.
It avoids writing and reading the full-resolution convolution output a
second time just to add the bias and apply the ReLU. TorchInductor
cannot generate this fusion itself, because it calls cuDNN for the
convolution and applies bias and activation in a separate kernel.

The pattern "convolution -> batch normalization -> ReLU" is not fused.
Batch normalization should be folded into the convolutions when the
model is exported (dcnm-model-files does this).
"""
from .torch_setup import torch


def _conv_args(node):
    """Return the normalized arguments of an `aten.conv2d` node"""
    args = list(node.args) + [None, None, None, [1, 1], [0, 0], [1, 1], 1][
        len(node.args):]
    for key, idx in (("bias", 2), ("stride", 3), ("padding", 4),
                     ("dilation", 5), ("groups", 6)):
        if key in node.kwargs:
            args[idx] = node.kwargs[key]
    return args


def _is_conv2d(node):
    return (node.op == "call_function"
            and node.target is torch.ops.aten.conv2d.default)


@torch.library.custom_op("dcnum::conv2d_relu", mutates_args=())
def conv2d_relu(x: torch.Tensor,
                weight: torch.Tensor,
                bias: torch.Tensor | None,
                stride: list[int],
                padding: list[int],
                dilation: list[int],
                groups: int) -> torch.Tensor:
    """2D convolution followed by ReLU

    On CUDA devices with cuDNN, this uses cuDNN's fused convolution +
    bias + ReLU kernel (the same kernel that
    :func:`torch.jit.optimize_for_inference` uses).
    """
    if x.is_cuda and torch.backends.cudnn.is_available():
        return torch.ops.aten.cudnn_convolution_relu(
            x.contiguous(), weight, bias, stride, padding, dilation, groups)
    return torch.relu(torch.nn.functional.conv2d(
        x, weight, bias, stride, padding, dilation, groups))


@conv2d_relu.register_fake
def _conv2d_relu_fake(x, weight, bias, stride, padding, dilation, groups):
    # Required for compiling the model (also with dynamic batch size).
    return torch.ops.aten.convolution(
        x, weight, bias, stride, padding, dilation, False, [0, 0], groups)


def fuse_conv_relu(gm: torch.fx.GraphModule) -> int:
    """Replace `conv2d -> relu` with ``dcnum::conv2d_relu`` in-place

    Only convolutions whose output is used by the ReLU alone are fused.
    Returns the number of fused convolutions.
    """
    aten = torch.ops.aten
    graph = gm.graph
    count = 0
    for relu in list(graph.nodes):
        if (relu.op != "call_function"
                or relu.target not in (aten.relu.default,
                                       aten.relu_.default)):
            continue
        conv = relu.args[0]
        if not _is_conv2d(conv) or len(conv.users) != 1:
            continue
        with graph.inserting_before(relu):
            fused = graph.call_function(torch.ops.dcnum.conv2d_relu.default,
                                        tuple(_conv_args(conv)))
        relu.replace_all_uses_with(fused)
        graph.erase_node(relu)
        graph.erase_node(conv)
        count += 1
    gm.recompile()
    return count


def optimize_for_inference(gm: torch.fx.GraphModule, device: str) -> dict:
    """Apply all inference optimizations suitable for `device` in-place

    Returns a dictionary with the number of applied optimizations.
    """
    info = {}
    if str(device).startswith("cuda"):
        info["fused_conv_relu"] = fuse_conv_relu(gm)
    return info
