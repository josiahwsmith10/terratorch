"""Tests for terratorch.spectral.catalog: canonicalization, aliasing, and
parity with the legacy lookup tables it replaces."""

import pytest

from terratorch.datasets.utils import HLSBands, OpticalBands, SARBands
from terratorch.spectral.catalog import (
    SENSORS,
    canonical_name,
    canonicalize_band,
    translate_bands,
)
from terratorch.spectral.schema import Modality, Polarization, Representation

# The dict this catalog replaces (was copy-pasted byte-identically in
# torchgeo_resnet.py, torchgeo_vit.py, torchgeo_swin_satlas.py).
LEGACY_LOOK_UP_TABLE = {
    "B01": "COASTAL_AEROSOL",
    "B02": "BLUE",
    "B03": "GREEN",
    "B04": "RED",
    "B05": "RED_EDGE_1",
    "B06": "RED_EDGE_2",
    "B07": "RED_EDGE_3",
    "B08": "NIR_BROAD",
    "B8A": "NIR_NARROW",
    "B09": "WATER_VAPOR",
    "B10": "CIRRUS",
    "B11": "SWIR_1",
    "B12": "SWIR_2",
    "VV": "VV",
    "VH": "VH",
    "R": "RED",
    "G": "GREEN",
    "B": "BLUE",
}

# The dict this catalog replaces in dofa_vit.py (waves_list), with its two
# legacy data bugs ("THERMAL_INFRARED_12", "VV-VH") intentionally included.
LEGACY_WAVES_LIST = {
    "COASTAL_AEROSOL": 0.44,
    "BLUE": 0.49,
    "GREEN": 0.56,
    "RED": 0.665,
    "RED_EDGE_1": 0.705,
    "RED_EDGE_2": 0.74,
    "RED_EDGE_3": 0.783,
    "NIR_BROAD": 0.832,
    "NIR_NARROW": 0.864,
    "WATER_VAPOR": 0.945,
    "CIRRUS": 1.373,
    "SWIR_1": 1.61,
    "SWIR_2": 2.20,
    "THERMAL_INFRARED_1": 10.90,
    "THERMAL_INFRARED_12": 12.00,
    "VV": 5.405,
    "VH": 5.405,
    "ASC_VV": 5.405,
    "ASC_VH": 5.405,
    "DSC_VV": 5.405,
    "DSC_VH": 5.405,
    "VV-VH": 5.405,
}


class TestLegacyParity:
    @pytest.mark.parametrize(("code", "expected"), sorted(LEGACY_LOOK_UP_TABLE.items()))
    def test_every_look_up_table_key_resolves_identically(self, code, expected):
        assert canonical_name(code) == expected

    @pytest.mark.parametrize("key", sorted(LEGACY_WAVES_LIST))
    def test_every_waves_list_key_resolves(self, key):
        band = canonicalize_band(key)
        assert band is not None, key

    @pytest.mark.parametrize(("key", "legacy_um"), sorted(LEGACY_WAVES_LIST.items()))
    def test_waves_list_values_within_rounding_of_catalog(self, key, legacy_um):
        """Catalog wavelengths come from torchgeo's authoritative
        Sentinel2.wavelengths table; the legacy waves_list values were
        hand-rounded. Assert the correction is bounded (< 0.006 um)."""
        band = canonicalize_band(key)
        assert band.dofa_wavelength_um == pytest.approx(legacy_um, abs=6e-3)


class TestUnpaddedTorchgeoModelSpellings:
    """torchgeo WeightsEnum meta uses un-padded codes ('B1'..'B8a'..'B12');
    the legacy table only knew padded ones — the root cause of issue #928."""

    @pytest.mark.parametrize(
        ("code", "expected"),
        [
            ("B1", "COASTAL_AEROSOL"),
            ("B2", "BLUE"),
            ("B4", "RED"),
            ("B8", "NIR_BROAD"),
            ("B8a", "NIR_NARROW"),  # lowercase, as in torchgeo _sentinel2_toa_bands
            ("B9", "WATER_VAPOR"),
        ],
    )
    def test_unpadded_resolves_with_sentinel2_hint(self, code, expected):
        assert canonical_name(code, sensor="sentinel2") == expected

    def test_unpadded_unhinted_defaults_to_sentinel2_with_warning(self):
        with pytest.warns(UserWarning, match="ambiguous"):
            assert canonical_name("B4") == "RED"

    def test_issue_928_rgb_moco_bands(self):
        assert translate_bands(["B4", "B3", "B2"], sensor="sentinel2") == ["RED", "GREEN", "BLUE"]


class TestSensorHints:
    def test_dotted_prefix_is_a_sensor_hint(self):
        assert canonical_name("SENTINEL2.B02") == "BLUE"
        assert canonical_name("S2.B8a") == "NIR_NARROW"

    def test_landsat_hint_changes_resolution(self):
        # B1 is coastal aerosol on Sentinel-2 but blue on Landsat TM
        assert canonical_name("B1", sensor="sentinel2") == "COASTAL_AEROSOL"
        assert canonical_name("B1", sensor="landsat_tm_toa") == "BLUE"
        assert canonical_name("SR_B1", sensor="landsat_oli_sr") == "COASTAL_AEROSOL"

    def test_unknown_sensor_raises(self):
        with pytest.raises(ValueError, match="Unknown sensor"):
            canonicalize_band("B1", sensor="not_a_sensor")

    def test_landsat_weights_meta_bands_all_resolve(self):
        """The latent sibling of #928: SSL4EO-L weight meta bands."""
        for sensor, bands in [
            ("landsat_tm_toa", ["B1", "B2", "B3", "B4", "B5", "B6", "B7"]),
            ("landsat_etm_toa", ["B1", "B2", "B3", "B4", "B5", "B6_VCID_1", "B6_VCID_2", "B7", "B8"]),
            ("landsat_etm_sr", ["SR_B1", "SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B7"]),
            ("landsat_oli_tirs_toa", ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B9", "B10", "B11"]),
            ("landsat_oli_sr", ["SR_B1", "SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B6", "SR_B7"]),
        ]:
            for code in bands:
                assert canonicalize_band(code, sensor=sensor) is not None, (sensor, code)


class TestEnumInterop:
    """Catalog names deliberately equal TerraTorch's enum values so specs are
    consumed with no translation layer."""

    @pytest.mark.parametrize("member", list(OpticalBands))
    def test_every_optical_band_resolves_to_itself(self, member):
        if member is OpticalBands.THERMAL_INFRARED_1 or member is OpticalBands.THERMAL_INFRARED_2:
            band = canonicalize_band(member, sensor="hls")
        else:
            band = canonicalize_band(member, sensor="sentinel2")
        assert band is not None and band.name == member.value

    @pytest.mark.parametrize("member", list(HLSBands))
    def test_every_hls_band_resolves_to_itself(self, member):
        band = canonicalize_band(member, sensor="hls")
        assert band is not None and band.name == member.value

    @pytest.mark.parametrize("member", list(SARBands))
    def test_every_sar_band_resolves_to_itself(self, member):
        band = canonicalize_band(member, sensor="sentinel1")
        assert band is not None and band.name == member.value


class TestSARTyping:
    def test_polarization_and_orbit(self):
        asc = canonicalize_band("ASC_VV")
        assert asc.modality is Modality.SAR
        assert asc.sar.polarization is Polarization.VV
        assert asc.sar.orbit == "ascending"
        dsc = canonicalize_band("DSC_VH")
        assert dsc.sar.polarization is Polarization.VH
        assert dsc.sar.orbit == "descending"

    def test_ratio_band_has_no_polarization(self):
        ratio = canonicalize_band("VV_VH")
        assert ratio.sar.polarization is None
        assert canonicalize_band("VV-VH") == ratio  # legacy DOFA spelling

    def test_default_representation_is_db(self):
        assert canonicalize_band("VV").sar.representation is Representation.DB


class TestTolerance:
    def test_unknown_band_warns_and_passes_through(self):
        with pytest.warns(UserWarning, match="Unknown band"):
            assert translate_bands(["NOT_A_BAND"]) == ["NOT_A_BAND"]

    def test_unknown_band_returns_none(self):
        assert canonicalize_band("NOT_A_BAND") is None

    def test_every_sensor_table_has_unique_names(self):
        for sensor, table in SENSORS.items():
            names = [b.name for b in table]
            assert len(names) == len(set(names)), f"duplicate names in {sensor}"
