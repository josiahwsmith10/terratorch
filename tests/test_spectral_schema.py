"""Tests for terratorch.spectral.schema (RFC 0001).

The schema modules are deliberately stdlib-only so the package can be lifted
verbatim into torchgeo; the purity test at the bottom enforces that.
"""

import json
import re
from pathlib import Path

import pytest

from terratorch.spectral.schema import (
    BandSpec,
    Modality,
    NormalizationSpec,
    Polarization,
    Representation,
    SARFrequencyBand,
    SARInfo,
    SpectralSpec,
)

SPECTRAL_PKG = Path(__file__).parent.parent / "terratorch" / "spectral"


def _s1_band(name: str, pol: Polarization | None, representation: Representation) -> BandSpec:
    return BandSpec(
        name=name,
        modality=Modality.SAR,
        sensor_code=name,
        sar=SARInfo(
            polarization=pol,
            frequency_band=SARFrequencyBand.C,
            representation=representation,
            center_frequency_ghz=5.405,
        ),
    )


class TestValidation:
    def test_sar_modality_requires_sar_info(self):
        with pytest.raises(ValueError, match="modality is SAR"):
            BandSpec(name="VV", modality=Modality.SAR)

    def test_sar_info_requires_sar_modality(self):
        with pytest.raises(ValueError, match="SARInfo given"):
            BandSpec(
                name="RED",
                modality=Modality.OPTICAL,
                sar=SARInfo(Polarization.VV, SARFrequencyBand.C, Representation.DB),
            )

    def test_normalization_length_mismatch(self):
        with pytest.raises(ValueError, match="means .* stds"):
            NormalizationSpec(means=(0.0, 0.0), stds=(1.0,))

    def test_normalization_zero_std(self):
        with pytest.raises(ValueError, match="non-zero"):
            NormalizationSpec(means=(0.0,), stds=(0.0,))

    def test_spec_normalization_band_count_mismatch(self):
        band = BandSpec(name="RED", modality=Modality.OPTICAL)
        with pytest.raises(ValueError, match="2 channels .* 1 bands"):
            SpectralSpec(bands=(band,), normalization=NormalizationSpec((0.0, 0.0), (1.0, 1.0)))

    def test_sequences_coerced_to_tuples(self):
        spec = SpectralSpec(
            bands=[BandSpec(name="RED", modality=Modality.OPTICAL, aliases=["R"])],
            normalization=NormalizationSpec(means=[0.0], stds=[1.0]),
        )
        assert isinstance(spec.bands, tuple)
        assert isinstance(spec.bands[0].aliases, tuple)
        assert isinstance(spec.normalization.means, tuple)


class TestMatching:
    def test_matches_name_code_alias_case_insensitive(self):
        band = BandSpec(
            name="RED", modality=Modality.OPTICAL, sensor_code="B04", aliases=("B4", "R")
        )
        for spelling in ("RED", "red", "B04", "b04", "B4", "r"):
            assert band.matches(spelling), spelling
        assert not band.matches("B05")

    def test_dofa_wavelength_convention(self):
        optical = BandSpec(name="RED", modality=Modality.OPTICAL, wavelength_um=0.6646)
        assert optical.dofa_wavelength_um == 0.6646
        sar = _s1_band("VV", Polarization.VV, Representation.DB)
        assert sar.dofa_wavelength_um == 5.405
        dem = BandSpec(name="DEM", modality=Modality.DEM)
        assert dem.dofa_wavelength_um is None


class TestSerde:
    def test_optical_round_trip(self):
        spec = SpectralSpec(
            bands=(
                BandSpec(
                    name="RED",
                    modality=Modality.OPTICAL,
                    sensor_code="B04",
                    aliases=("B4", "R"),
                    wavelength_um=0.6646,
                    bandwidth_um=0.031,
                ),
            ),
            normalization=NormalizationSpec(means=(0.0,), stds=(10000.0,), note="10k"),
            source="test",
        )
        assert SpectralSpec.from_json(spec.to_json()) == spec

    def test_sar_complex_round_trip(self):
        """Deliverable 4 core: a complex/amplitude-aware SAR entry survives
        JSON serialization with enum identity intact."""
        spec = SpectralSpec(
            bands=(
                _s1_band("VV", Polarization.VV, Representation.COMPLEX),
                _s1_band("VH", Polarization.VH, Representation.COMPLEX),
            ),
            source="demo:SENTINEL1_SLC",
        )
        restored = SpectralSpec.from_json(spec.to_json())
        assert restored == spec
        assert restored.bands[0].sar.representation is Representation.COMPLEX
        assert restored.bands[0].sar.frequency_band is SARFrequencyBand.C
        assert restored.bands[0].sar.polarization is Polarization.VV
        assert restored.bands[1].sar.polarization is Polarization.VH

    def test_composite_sar_band_without_polarization(self):
        ratio = BandSpec(
            name="VV_VH",
            modality=Modality.SAR,
            aliases=("VV-VH",),
            sar=SARInfo(None, SARFrequencyBand.C, Representation.DB),
        )
        restored = BandSpec.from_dict(ratio.to_dict())
        assert restored == ratio
        assert restored.sar.polarization is None

    def test_dynamic_bands_round_trip(self):
        spec = SpectralSpec(bands=(), dynamic_bands=True, source="torchgeo:DOFA_MAE")
        restored = SpectralSpec.from_json(spec.to_json())
        assert restored == spec
        assert restored.dynamic_bands

    def test_dict_is_json_compatible(self):
        spec = SpectralSpec(
            bands=(_s1_band("VV", Polarization.VV, Representation.DB),),
            normalization=NormalizationSpec(means=(-12.59,), stds=(5.26,)),
        )
        # to_dict must contain only JSON primitives (embeddable in Weights.meta)
        json.dumps(spec.to_dict())

    def test_from_dict_ignores_unknown_keys(self):
        """Forward compatibility: older consumers tolerate newer producers."""
        d = {
            "version": 1,
            "bands": [
                {"name": "RED", "modality": "optical", "future_field": 123},
            ],
            "normalization": {"means": [0.0], "stds": [1.0], "another_future_field": "x"},
            "top_level_future_field": True,
        }
        spec = SpectralSpec.from_dict(d)
        assert spec.band_names() == ("RED",)
        assert spec.normalization.means == (0.0,)


class TestModulePurity:
    """The spectral package must stay stdlib-only (plus intra-package imports)
    so it can be lifted verbatim into torchgeo (RFC 0001 §9)."""

    FORBIDDEN = re.compile(
        r"^\s*(?:import|from)\s+(torch|torchvision|torchgeo|kornia|numpy|timm|albumentations)\b",
        re.MULTILINE,
    )

    @pytest.mark.parametrize(
        "module", ["__init__.py", "schema.py", "catalog.py", "resolve.py", "torchgeo_specs.py"]
    )
    def test_no_heavy_imports(self, module):
        source = (SPECTRAL_PKG / module).read_text()
        match = self.FORBIDDEN.search(source)
        assert match is None, f"{module} imports {match.group(1)!r}; must stay stdlib-only"
