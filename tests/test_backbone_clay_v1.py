# Copyright contributors to the Terratorch project
"""First-class clay_v1 backbone tests: registry build, EncoderDecoderFactory end-to-end,
metadata kwargs, and the vendored posemb/Datacuber fixes."""

import gc

import pytest
import torch

from terratorch.models import EncoderDecoderFactory
from terratorch.models.backbones.clay_v1.modules import Datacuber
from terratorch.models.backbones.clay_v1.utils import posemb_sincos_1d, posemb_sincos_2d_with_gsd
from terratorch.registry import BACKBONE_REGISTRY

NUM_CLASSES = 2
IMG_SIZE = 96
BANDS = ["BLUE", "GREEN", "RED", "NIR_NARROW", "SWIR_1", "SWIR_2"]
EMBED_DIM = 768

VIT_UPERNET_NECK = [
    {"name": "SelectIndices", "indices": [0, 1, 2, 3]},
    {"name": "ReshapeTokensToImage"},
    {"name": "LearnedInterpolateToPyramidal"},
]


@pytest.fixture(scope="module")
def model_factory() -> EncoderDecoderFactory:
    return EncoderDecoderFactory()


@pytest.fixture
def model_input() -> torch.Tensor:
    return torch.ones((2, len(BANDS), IMG_SIZE, IMG_SIZE))


def test_registry_build_returns_token_list(model_input):
    backbone = BACKBONE_REGISTRY.build(
        "clay_v1_base", model_bands=BANDS, pretrained=False, img_size=IMG_SIZE
    )
    backbone.eval()
    assert backbone.has_cls_token
    with torch.no_grad():
        features = backbone(model_input)
    assert isinstance(features, list)
    assert len(features) == len(backbone.out_channels)
    n_tokens = (IMG_SIZE // 8) ** 2 + 1  # patch 8, + cls token
    for feature, channels in zip(features, backbone.out_channels):
        assert channels == EMBED_DIM
        assert feature.shape == (model_input.shape[0], n_tokens, EMBED_DIM)
    gc.collect()


def test_out_indices(model_input):
    out_indices = [2, 5, 8, 11]
    backbone = BACKBONE_REGISTRY.build(
        "clay_v1_base", model_bands=BANDS, pretrained=False, img_size=IMG_SIZE, out_indices=out_indices
    )
    backbone.eval()
    assert backbone.out_channels == [EMBED_DIM] * len(out_indices)
    with torch.no_grad():
        features = backbone(model_input)
    assert len(features) == len(out_indices)
    gc.collect()


def test_metadata_kwargs_forward(model_input):
    backbone = BACKBONE_REGISTRY.build(
        "clay_v1_base", model_bands=BANDS, pretrained=False, img_size=IMG_SIZE
    )
    backbone.eval()
    batch = model_input.shape[0]
    with torch.no_grad():
        features = backbone(
            model_input,
            time=torch.zeros(batch, 4),
            latlon=torch.zeros(batch, 4),
            gsd=10.0,
            waves=[0.49, 0.56, 0.665, 0.864, 1.61, 2.2],
        )
    assert len(features) == len(backbone.out_channels)
    gc.collect()


def test_unknown_band_raises():
    backbone = BACKBONE_REGISTRY.build(
        "clay_v1_base", model_bands=["BLUE", "NOT_A_BAND"], pretrained=False, img_size=IMG_SIZE
    )
    with pytest.raises(ValueError, match="Unknown band"):
        backbone(torch.ones(1, 2, IMG_SIZE, IMG_SIZE))
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
        "backbone": "clay_v1_base",
        "decoder": "UperNetDecoder",
        "backbone_model_bands": BANDS,
        "backbone_pretrained": False,
        "backbone_img_size": IMG_SIZE,
        "backbone_out_indices": [2, 5, 8, 11],
        "necks": VIT_UPERNET_NECK,
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
        backbone="clay_v1_base",
        decoder="IdentityDecoder",
        backbone_model_bands=BANDS,
        backbone_pretrained=False,
        backbone_img_size=IMG_SIZE,
        num_classes=NUM_CLASSES,
        necks=[{"name": "PermuteDims", "new_order": [0, 2, 1]}],
    )
    model.eval()
    with torch.no_grad():
        assert model(model_input).output.shape == (2, NUM_CLASSES)
    gc.collect()


# ---- vendored fixes ----


def test_posemb_sincos_1d_integer_positions_are_not_truncated():
    int_waves = torch.tensor([490, 560, 665])
    float_waves = int_waves.float()
    pe_int = posemb_sincos_1d(int_waves, 64)
    pe_float = posemb_sincos_1d(float_waves, 64)
    assert pe_int.dtype == torch.float32
    torch.testing.assert_close(pe_int, pe_float)


def test_posemb_sincos_2d_with_gsd_accepts_tensor_gsd():
    pe_float = posemb_sincos_2d_with_gsd(4, 4, 64, gsd=10.0)
    pe_tensor = posemb_sincos_2d_with_gsd(4, 4, 64, gsd=torch.tensor(10.0))
    torch.testing.assert_close(pe_float, pe_tensor)


def test_datacuber_unknown_band_raises():
    datacuber = Datacuber(bands=["BLUE", "NOT_A_BAND"])
    with pytest.raises(ValueError, match="Unknown band"):
        datacuber(torch.ones(1, 2, 16, 16))


def test_datacuber_band_count_mismatch_raises():
    datacuber = Datacuber(bands=["BLUE", "GREEN"])
    with pytest.raises(ValueError, match="does not match input channels"):
        datacuber(torch.ones(1, 3, 16, 16))
