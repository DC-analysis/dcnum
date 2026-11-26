import numpy as np
import pytest

from helper_methods import retrieve_model

torch = pytest.importorskip("torch")

if not hasattr(torch.library, "custom_op"):
    pytest.skip("custom operators require torch 2.4", allow_module_level=True)

from dcnum.segm.segm_torch import torch_graph_opt, torch_model  # noqa: E402


class SmallUNetBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 4, 3, padding=1)
        self.conv2 = torch.nn.Conv2d(4, 4, 3, padding=1, bias=False)
        self.conv3 = torch.nn.Conv2d(8, 1, 1)

    def forward(self, x):
        a = torch.relu(self.conv1(x))
        b = torch.relu(self.conv2(a))
        return self.conv3(torch.cat([a, b], dim=1))


def get_exported_module():
    torch.manual_seed(42)
    model = SmallUNetBlock().eval()
    x = torch.rand(2, 1, 16, 32)
    ep = torch.export.export(
        model, (x,),
        dynamic_shapes=[(torch.export.Dim("batch", min=1, max=64),
                         torch.export.Dim.STATIC,
                         torch.export.Dim.STATIC,
                         torch.export.Dim.STATIC)])
    return ep, model


def count_targets(gm, target):
    return sum(1 for n in gm.graph.nodes
               if n.op == "call_function" and n.target is target)


def test_fuse_conv_relu():
    ep, model = get_exported_module()
    gm = ep.module()
    x = torch.rand(3, 1, 16, 32)
    with torch.inference_mode():
        ref = model(x)
    assert torch_graph_opt.fuse_conv_relu(gm) == 2
    assert count_targets(gm, torch.ops.dcnum.conv2d_relu.default) == 2
    assert count_targets(gm, torch.ops.aten.relu.default) == 0
    # the last convolution is not followed by a ReLU
    assert count_targets(gm, torch.ops.aten.conv2d.default) == 1
    with torch.inference_mode():
        res = gm(x)
    # on the CPU, the custom operator computes conv2d and relu separately
    assert np.array_equal(ref.numpy(), res.numpy())


def test_fuse_conv_relu_compile_dynamic():
    """The custom operator must support dynamic batch sizes"""
    ep, model = get_exported_module()
    gm = ep.module()
    torch_graph_opt.fuse_conv_relu(gm)
    compiled = torch.compile(gm, backend="aot_eager", fullgraph=True,
                             dynamic=True)
    with torch.inference_mode():
        for size in [3, 7]:
            x = torch.rand(size, 1, 16, 32)
            assert np.allclose(model(x).numpy(), compiled(x).numpy(),
                               atol=1e-6, rtol=0)


def test_optimize_for_inference_only_on_cuda():
    ep, _ = get_exported_module()
    gm = ep.module()
    assert torch_graph_opt.optimize_for_inference(gm, device="cpu") == {}
    assert count_targets(gm, torch.ops.dcnum.conv2d_relu.default) == 0
    assert torch_graph_opt.optimize_for_inference(gm, device="cuda") == {
        "fused_conv_relu": 2}


def test_load_model_graph_optimizations_cpu(monkeypatch):
    monkeypatch.setattr(torch_model, "cache_model", {})
    model_file = retrieve_model(
        "segm-torch-model_unet-dcnum-test_g2_0901b.zip")
    _, meta = torch_model.load_model(model_file,
                                     backend="torch.eager",
                                     device="cpu")
    assert meta["graph_optimizations"] == {}
