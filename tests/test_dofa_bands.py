"""Tests for the catalog-backed DOFA wavelength resolution (RFC 0001)."""

import pytest
import torch

from terratorch.models.backbones.dofa_vit import (
    DOFAEncoderWrapper,
    dofa_base_patch16_224,
    get_wavelenghts,
    get_wavelengths,
    waves_list,
)


class TestGetWavelengths:
    def test_optical_bands_use_torchgeo_wavelengths(self):
        # authoritative torchgeo Sentinel2.wavelengths values (um)
        assert get_wavelengths(["RED", "GREEN", "BLUE"]) == [0.6646, 0.5598, 0.4927]
        assert get_wavelengths(["NIR_NARROW", "SWIR_1", "SWIR_2"]) == [0.8647, 1.6137, 2.2024]

    def test_sensor_code_spellings(self):
        # DOFA users can now pass sensor codes, not just semantic names
        assert get_wavelengths(["B04", "B4"]) == [0.6646, 0.6646]

    def test_sar_bands_use_dofa_frequency_convention(self):
        assert get_wavelengths(["VV", "VH", "ASC_VV", "DSC_VH"]) == [5.405] * 4

    def test_legacy_waves_list_typos_still_resolve(self):
        # "THERMAL_INFRARED_12" and "VV-VH" were data bugs in waves_list;
        # both old and corrected spellings work
        assert get_wavelengths(["THERMAL_INFRARED_12"]) == [12.00]
        assert get_wavelengths(["THERMAL_INFRARED_2"]) == [12.00]
        assert get_wavelengths(["VV-VH"]) == [5.405]
        assert get_wavelengths(["VV_VH"]) == [5.405]

    def test_values_close_to_legacy_waves_list(self):
        """The catalog corrects hand-rounded values by at most 0.005 um."""
        for key, legacy in waves_list.items():
            assert get_wavelengths([key])[0] == pytest.approx(legacy, abs=6e-3), key

    def test_unknown_band_raises_descriptive_error(self):
        with pytest.raises(ValueError, match="NOT_A_BAND"):
            get_wavelengths(["RED", "NOT_A_BAND"])

    def test_band_without_wavelength_raises(self):
        with pytest.raises(ValueError, match="DEM"):
            get_wavelengths(["DEM"])

    def test_deprecated_misspelled_alias(self):
        with pytest.warns(DeprecationWarning, match="get_wavelenghts"):
            assert get_wavelenghts(["RED"]) == get_wavelengths(["RED"])


class TestDOFAWrapperSpec:
    def test_wrapper_exposes_dynamic_spec(self):
        pytest.importorskip("torchgeo.models.dofa")
        from torchgeo.models import dofa

        wrapper = dofa_base_patch16_224(
            model_bands=["RED", "GREEN", "BLUE"],
            pretrained=False,
            weights=dofa.DOFABase16_Weights.DOFA_MAE,
        )
        assert isinstance(wrapper, DOFAEncoderWrapper)
        assert wrapper.spectral_spec is not None
        assert wrapper.spectral_spec.dynamic_bands
        assert wrapper.spectral_spec.normalization is None  # torchgeo: CenterCrop only
        assert wrapper.wavelengths == [0.6646, 0.5598, 0.4927]

    def test_wrapper_forward(self):
        wrapper = dofa_base_patch16_224(
            model_bands=["RED", "GREEN", "BLUE"], pretrained=False, weights=None
        )
        x = torch.randn(1, 3, 224, 224)
        outs = wrapper(x)
        assert len(outs) >= 1
        assert all(isinstance(o, torch.Tensor) for o in outs)
