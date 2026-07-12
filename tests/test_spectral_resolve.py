"""Tests for terratorch.spectral.resolve: dataset bands -> model bands,
returning both index selection and normalization."""

import pytest

from terratorch.datasets.utils import HLSBands
from terratorch.spectral.resolve import resolve_bands
from terratorch.spectral.schema import (
    BandSpec,
    Modality,
    NormalizationSpec,
    SpectralSpec,
)


def _spec_13_band_s2() -> SpectralSpec:
    """Shaped like ResNet50_Weights.SENTINEL2_ALL_MOCO's spec: 13 bands with
    distinct per-band stats so index selection is observable in means/stds."""
    names = [
        "COASTAL_AEROSOL", "BLUE", "GREEN", "RED", "RED_EDGE_1", "RED_EDGE_2",
        "RED_EDGE_3", "NIR_BROAD", "NIR_NARROW", "WATER_VAPOR", "CIRRUS",
        "SWIR_1", "SWIR_2",
    ]
    codes = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B10", "B11", "B12"]
    return SpectralSpec(
        bands=tuple(
            BandSpec(name=n, modality=Modality.OPTICAL, sensor_code=c)
            for n, c in zip(names, codes)
        ),
        normalization=NormalizationSpec(
            means=tuple(float(i) for i in range(13)),
            stds=tuple(float(10 + i) for i in range(13)),
        ),
    )


class TestIndexSelection:
    def test_identity(self):
        spec = _spec_13_band_s2()
        res = resolve_bands([b.name for b in spec.bands], spec)
        assert res.indices == tuple(range(13))
        assert res.missing == ()
        assert res.pretrained_band_names == spec.band_names()

    def test_subset_and_reorder(self):
        spec = _spec_13_band_s2()
        res = resolve_bands(["SWIR_1", "BLUE", "RED"], spec)
        assert res.indices == (11, 1, 3)
        assert res.matched == ("SWIR_1", "BLUE", "RED")
        # normalization sliced per requested band, in request order
        assert res.means == (11.0, 1.0, 3.0)
        assert res.stds == (21.0, 11.0, 13.0)

    def test_sensor_code_spellings_match(self):
        spec = _spec_13_band_s2()
        res = resolve_bands(["B04", "B4", "R"], spec)
        assert res.indices == (3, 3, 3)
        assert res.requested == ("RED", "RED", "RED")

    def test_enum_inputs(self):
        spec = _spec_13_band_s2()
        res = resolve_bands([HLSBands.RED, HLSBands.NIR_NARROW], spec)
        assert res.indices == (3, 8)

    def test_missing_bands_reported_with_identity_normalization(self):
        spec = _spec_13_band_s2()
        res = resolve_bands(["RED", "DEM"], spec)
        assert res.indices == (3, None)
        assert res.missing == ("DEM",)
        assert res.matched == ("RED",)
        assert res.means == (3.0, 0.0)
        assert res.stds == (13.0, 1.0)

    def test_strict_raises_on_missing(self):
        spec = _spec_13_band_s2()
        with pytest.raises(ValueError, match="DEM"):
            resolve_bands(["RED", "DEM"], spec, strict=True)


class TestNormalizationAbsence:
    def test_spec_without_normalization_yields_none(self):
        spec = SpectralSpec(
            bands=(BandSpec(name="RED", modality=Modality.OPTICAL, sensor_code="B04"),)
        )
        res = resolve_bands(["RED"], spec)
        assert res.means is None
        assert res.stds is None
        assert res.indices == (0,)

    def test_empty_dynamic_spec(self):
        spec = SpectralSpec(bands=(), dynamic_bands=True)
        res = resolve_bands(["RED", "BLUE"], spec)
        assert res.indices == (None, None)
        assert res.missing == ("RED", "BLUE")
        assert res.pretrained_band_names == ()


class TestSpecOwnAliases:
    def test_spec_alias_catches_spelling_catalog_does_not_know(self):
        spec = SpectralSpec(
            bands=(
                BandSpec(
                    name="CUSTOM_HYPERSPECTRAL_17",
                    modality=Modality.OPTICAL,
                    aliases=("H17",),
                ),
            )
        )
        res = resolve_bands(["H17"], spec)
        assert res.indices == (0,)
        assert res.matched == ("H17",)
