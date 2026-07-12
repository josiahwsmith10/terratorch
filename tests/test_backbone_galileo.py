# Copyright contributors to the Terratorch project
"""Galileo backbone tests: registry build, band-group mapping, EncoderDecoderFactory
end-to-end, temporal input, and the (network-gated) pretrained path."""

import gc
import os

import pytest
import torch

from terratorch.models import EncoderDecoderFactory
from terratorch.registry import BACKBONE_REGISTRY

IN_GITHUB_ACTIONS = os.getenv("GITHUB_ACTIONS") == "true"


def _huggingface_unreachable() -> bool:
    import urllib.request

    try:
        urllib.request.urlopen("https://huggingface.co", timeout=5)
    except Exception:
        return True
    return False


NUM_CLASSES = 2
IMG_SIZE = 64
PATCH_SIZE = 4
RGB_BANDS = ["BLUE", "GREEN", "RED"]
NANO_EMBED_DIM = 128

NOCLS_UPERNET_NECK = [
    {"name": "SelectIndices", "indices": [0, 1, 2, 3]},
    {"name": "ReshapeTokensToImage", "remove_cls_token": False},
    {"name": "LearnedInterpolateToPyramidal"},
]


@pytest.fixture(scope="module")
def model_factory() -> EncoderDecoderFactory:
    return EncoderDecoderFactory()


@pytest.fixture
def model_input() -> torch.Tensor:
    return torch.ones((2, len(RGB_BANDS), IMG_SIZE, IMG_SIZE))


def test_registry_build_returns_token_list(model_input):
    backbone = BACKBONE_REGISTRY.build("galileo_nano", model_bands=RGB_BANDS, pretrained=False)
    backbone.eval()
    assert not backbone.has_cls_token
    with torch.no_grad():
        features = backbone(model_input)
    assert isinstance(features, list)
    assert len(features) == len(backbone.out_channels)
    n_tokens = (IMG_SIZE // PATCH_SIZE) ** 2  # no cls token
    for feature, channels in zip(features, backbone.out_channels):
        assert channels == NANO_EMBED_DIM
        assert feature.shape == (model_input.shape[0], n_tokens, NANO_EMBED_DIM)
    gc.collect()


def test_out_indices(model_input):
    out_indices = [0, 1, 2, 3]
    backbone = BACKBONE_REGISTRY.build(
        "galileo_nano", model_bands=RGB_BANDS, pretrained=False, out_indices=out_indices
    )
    backbone.eval()
    assert backbone.out_channels == [NANO_EMBED_DIM] * len(out_indices)
    with torch.no_grad():
        features = backbone(model_input)
    assert len(features) == len(out_indices)
    gc.collect()


@pytest.mark.parametrize(
    "bands",
    [
        RGB_BANDS,  # S2_RGB group
        ["VV", "VH"],  # S1 group
        ["B2", "B3", "B4"],  # native names
        ["BLUE", "GREEN", "RED", "RED_EDGE_1", "RED_EDGE_2", "RED_EDGE_3", "NIR_BROAD", "NIR_NARROW", "SWIR_1", "SWIR_2"],
    ],
)
def test_complete_band_groups_accepted(bands):
    backbone = BACKBONE_REGISTRY.build("galileo_nano", model_bands=bands, pretrained=False)
    backbone.eval()
    with torch.no_grad():
        features = backbone(torch.ones(1, len(bands), IMG_SIZE, IMG_SIZE))
    assert features[0].shape[-1] == NANO_EMBED_DIM
    gc.collect()


def test_partial_band_group_raises():
    with pytest.raises(ValueError, match="only partially declared"):
        BACKBONE_REGISTRY.build("galileo_nano", model_bands=["BLUE", "GREEN"], pretrained=False)


def test_unknown_band_raises():
    with pytest.raises(ValueError, match="Unknown band"):
        BACKBONE_REGISTRY.build("galileo_nano", model_bands=["NOT_A_BAND"], pretrained=False)


def test_months_kwarg_and_temporal_input():
    backbone = BACKBONE_REGISTRY.build("galileo_nano", model_bands=RGB_BANDS, pretrained=False)
    backbone.eval()
    with torch.no_grad():
        features = backbone(torch.ones(2, len(RGB_BANDS), IMG_SIZE, IMG_SIZE), months=3)
        temporal = backbone(torch.ones(2, len(RGB_BANDS), 2, IMG_SIZE, IMG_SIZE))
    n_tokens = (IMG_SIZE // PATCH_SIZE) ** 2
    assert features[-1].shape == (2, n_tokens, NANO_EMBED_DIM)
    assert temporal[-1].shape == (2, n_tokens, NANO_EMBED_DIM)
    gc.collect()


@pytest.mark.parametrize(
    ("task", "expected"),
    [
        ("segmentation", (2, NUM_CLASSES, IMG_SIZE, IMG_SIZE)),
        ("regression", (2, IMG_SIZE, IMG_SIZE)),
    ],
)
def test_pixelwise_model_end_to_end(task, expected, model_factory, model_input):
    model_args = {
        "task": task,
        "backbone": "galileo_nano",
        "decoder": "UperNetDecoder",
        "backbone_model_bands": RGB_BANDS,
        "backbone_pretrained": False,
        "backbone_out_indices": [0, 1, 2, 3],
        "necks": NOCLS_UPERNET_NECK,
    }
    if task == "segmentation":
        model_args["num_classes"] = NUM_CLASSES
    model = model_factory.build_model(**model_args)
    model.eval()
    with torch.no_grad():
        assert model(model_input).output.shape == expected
    gc.collect()


def test_classification_model_end_to_end(model_factory, model_input):
    model = model_factory.build_model(
        "classification",
        backbone="galileo_nano",
        decoder="IdentityDecoder",
        backbone_model_bands=RGB_BANDS,
        backbone_pretrained=False,
        num_classes=NUM_CLASSES,
        necks=[{"name": "PermuteDims", "new_order": [0, 2, 1]}],
    )
    model.eval()
    with torch.no_grad():
        assert model(model_input).output.shape == (2, NUM_CLASSES)
    gc.collect()


@pytest.mark.parametrize("size", ["tiny", "base"])
def test_larger_sizes_build_offline(size):
    backbone = BACKBONE_REGISTRY.build(f"galileo_{size}", model_bands=RGB_BANDS, pretrained=False)
    backbone.eval()
    with torch.no_grad():
        features = backbone(torch.ones(1, len(RGB_BANDS), 32, 32))
    assert features[0].shape[-1] == backbone.out_channels[0]
    gc.collect()


@pytest.mark.skipif(
    IN_GITHUB_ACTIONS or _huggingface_unreachable(), reason="Weight download requires network access"
)
def test_pretrained_weights_load():
    backbone = BACKBONE_REGISTRY.build("galileo_nano", model_bands=RGB_BANDS, pretrained=True)
    backbone.eval()
    with torch.no_grad():
        features = backbone(torch.ones(1, len(RGB_BANDS), IMG_SIZE, IMG_SIZE))
    assert features[-1].shape[-1] == backbone.out_channels[-1]
    gc.collect()
