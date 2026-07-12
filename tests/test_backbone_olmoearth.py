# Copyright contributors to the Terratorch project
"""OlmoEarth backbone tests: registry build across the v1/v1.1/v1.2 matrix, band-set
mapping, hook-based out_indices, EncoderDecoderFactory end-to-end, and the
(network-gated) pretrained paths."""

import gc
import os

import pytest
import torch

pytest.importorskip("olmoearth_pretrain_minimal")

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
S2_SET0_BANDS = ["BLUE", "GREEN", "RED", "NIR_BROAD"]  # complete 10m band set
S2_FULL_BANDS = [
    "COASTAL_AEROSOL", "BLUE", "GREEN", "RED",
    "RED_EDGE_1", "RED_EDGE_2", "RED_EDGE_3",
    "NIR_BROAD", "NIR_NARROW", "WATER_VAPOR", "SWIR_1", "SWIR_2",
]
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
    return torch.ones((2, len(S2_SET0_BANDS), IMG_SIZE, IMG_SIZE))


def test_registry_build_returns_token_list(model_input):
    backbone = BACKBONE_REGISTRY.build("olmoearth_v1_nano", model_bands=S2_SET0_BANDS, pretrained=False)
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


# offline (random-weight) build matrix: every (version, size) combination that
# OlmoEarthPretrain_v1 can construct without network access. v1.2 "small" is excluded:
# olmoearth-pretrain-minimal 0.0.6 cannot random-init it (upstream KeyError bug), so
# terratorch falls back to a config.json download for that one (network-gated test below).
OFFLINE_MATRIX = [
    "olmoearth_v1_nano",
    "olmoearth_v1_tiny",
    "olmoearth_v1_1_nano",
    "olmoearth_v1_1_tiny",
    "olmoearth_v1_2_nano",
    "olmoearth_v1_2_tiny",
]


@pytest.mark.parametrize("backbone_name", OFFLINE_MATRIX)
def test_offline_matrix_out_indices(backbone_name):
    out_indices = [0, 1, 2, 3]
    backbone = BACKBONE_REGISTRY.build(
        backbone_name, model_bands=S2_SET0_BANDS, pretrained=False, out_indices=out_indices
    )
    backbone.eval()
    with torch.no_grad():
        features = backbone(torch.ones(1, len(S2_SET0_BANDS), 32, 32))
    assert len(features) == len(out_indices)
    for feature, channels in zip(features, backbone.out_channels):
        assert feature.shape == (1, (32 // PATCH_SIZE) ** 2, channels)
    gc.collect()


@pytest.mark.parametrize(
    "bands",
    [
        S2_SET0_BANDS,
        S2_FULL_BANDS,
        ["VV", "VH"],  # sentinel1
        ["B02", "B03", "B04", "B08"],  # native names
    ],
)
def test_complete_band_sets_accepted(bands):
    backbone = BACKBONE_REGISTRY.build("olmoearth_v1_nano", model_bands=bands, pretrained=False)
    backbone.eval()
    with torch.no_grad():
        features = backbone(torch.ones(1, len(bands), 32, 32))
    assert features[0].shape[-1] == NANO_EMBED_DIM
    gc.collect()


def test_partial_band_set_raises():
    with pytest.raises(ValueError, match="only partially declared"):
        BACKBONE_REGISTRY.build("olmoearth_v1_nano", model_bands=["BLUE", "GREEN", "RED"], pretrained=False)


def test_mixed_modalities_raise():
    with pytest.raises(ValueError, match="single OlmoEarth modality"):
        BACKBONE_REGISTRY.build("olmoearth_v1_nano", model_bands=["BLUE", "VV"], pretrained=False)


def test_timestamps_kwarg_and_temporal_input():
    backbone = BACKBONE_REGISTRY.build("olmoearth_v1_nano", model_bands=S2_SET0_BANDS, pretrained=False)
    backbone.eval()
    with torch.no_grad():
        features = backbone(
            torch.ones(2, len(S2_SET0_BANDS), IMG_SIZE, IMG_SIZE),
            timestamps=torch.tensor([1, 0, 2024]),
        )
        temporal = backbone(torch.ones(2, len(S2_SET0_BANDS), 2, IMG_SIZE, IMG_SIZE))
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
        "backbone": "olmoearth_v1_nano",
        "decoder": "UperNetDecoder",
        "backbone_model_bands": S2_SET0_BANDS,
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
        backbone="olmoearth_v1_nano",
        decoder="IdentityDecoder",
        backbone_model_bands=S2_SET0_BANDS,
        backbone_pretrained=False,
        num_classes=NUM_CLASSES,
        necks=[{"name": "PermuteDims", "new_order": [0, 2, 1]}],
    )
    model.eval()
    with torch.no_grad():
        assert model(model_input).output.shape == (2, NUM_CLASSES)
    gc.collect()


@pytest.mark.skipif(
    IN_GITHUB_ACTIONS or _huggingface_unreachable(), reason="Config download requires network access"
)
def test_v1_2_small_builds_via_config_fallback():
    # random init of v1.2 small hits an upstream KeyError in olmoearth-pretrain-minimal
    # 0.0.6; terratorch falls back to building from the HF config.json (no weights).
    backbone = BACKBONE_REGISTRY.build("olmoearth_v1_2_small", model_bands=S2_SET0_BANDS, pretrained=False)
    backbone.eval()
    with torch.no_grad():
        features = backbone(torch.ones(1, len(S2_SET0_BANDS), 32, 32))
    assert features[-1].shape[-1] == backbone.out_channels[-1] == 384
    gc.collect()


@pytest.mark.skipif(
    IN_GITHUB_ACTIONS or _huggingface_unreachable(), reason="Weight download requires network access"
)
@pytest.mark.parametrize(
    "backbone_name", ["olmoearth_v1_nano", "olmoearth_v1_1_nano", "olmoearth_v1_2_nano"]
)
def test_pretrained_weights_load(backbone_name):
    backbone = BACKBONE_REGISTRY.build(backbone_name, model_bands=S2_SET0_BANDS, pretrained=True)
    backbone.eval()
    with torch.no_grad():
        # out_indices=[-1] default; the hook path's runtime reshape assertion
        # (N == H'*W'*T*S + register tokens) is exercised by intermediate indices
        features = backbone(torch.ones(1, len(S2_SET0_BANDS), IMG_SIZE, IMG_SIZE))
    assert features[-1].shape[-1] == backbone.out_channels[-1]
    gc.collect()
