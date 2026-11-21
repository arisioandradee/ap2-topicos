import os, sys
import numpy as np, pytest, timm, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from src import xai


@pytest.fixture(scope="module")
def net():
    torch.manual_seed(0)
    m = timm.create_model("resnet18", pretrained=False, num_classes=6).eval()
    return m


def test_select_cases_criterion():
    y = np.array([1, 1, 0, 0, 1, 0, 1, 0]); p = np.array([.9, .2, .8, .1, .6, .45, .55, .05])
    s = xai.select_cases(y, p, k=1, thr=0.5)
    assert s["TP"] == [0] and s["FN"] == [1] and s["FP"] == [2] and s["TN"] == [7]
    assert s["limiar"] == [6] or s["limiar"] == [5]


def test_maps_and_metrics(net):
    x = torch.randn(2, 3, 64, 64)
    g = xai.saliency_gradcam(net, x, net.layer4[-1])
    assert g.shape == (2, 64, 64) and g.min() >= 0 and g.max() <= 1.0001
    o = xai.saliency_occlusion(net, x[:1], window=16, stride=16)
    assert o.shape == (1, 64, 64)
    d, i, _ = xai.deletion_insertion(net, x[:1], g[0], steps=5)
    assert 0 <= d <= 1 and 0 <= i <= 1
    assert 0 <= xai.border_fraction(g[0]) <= 1
    r = xai.sanity_correlation(lambda m, z: xai.saliency_gradcam(m, z, m.layer4[-1])[0], net, x[:1], "full")
    assert -1 <= r <= 1


def test_vit_reshape():
    m = timm.create_model("vit_small_patch16_224", pretrained=False, num_classes=6, img_size=64).eval()
    layer, rs = xai.target_layer(m, "vit_small_sup")
    g = xai.saliency_gradcam(m, torch.randn(1, 3, 64, 64), layer, rs)
    assert g.shape == (1, 64, 64)
