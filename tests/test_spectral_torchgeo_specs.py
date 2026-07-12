"""Tests for terratorch.spectral.torchgeo_specs: the per-weight registry, the
spec_for_weights fallback ladder, the SAR round-trip demo, and numerical
equivalence of registry normalization constants against torchgeo's real bound
transforms (which construct from meta alone — no checkpoint download)."""

import json
from unittest.mock import Mock

import pytest

from terratorch.spectral.resolve import resolve_bands
from terratorch.spectral.schema import (
    SPECTRAL_SPEC_META_KEY,
    BandSpec,
    Modality,
    NormalizationSpec,
    Polarization,
    Representation,
    SARFrequencyBand,
    SARInfo,
    SpectralSpec,
)
from terratorch.spectral.torchgeo_specs import (
    WEIGHT_SPECS,
    register_weight_spec,
    spec_for_weights,
)

S2_ALL_NAMES = (
    "COASTAL_AEROSOL", "BLUE", "GREEN", "RED", "RED_EDGE_1", "RED_EDGE_2",
    "RED_EDGE_3", "NIR_BROAD", "NIR_NARROW", "WATER_VAPOR", "CIRRUS",
    "SWIR_1", "SWIR_2",
)


class TestRegistryEntries:
    def test_sentinel2_all_moco(self):
        spec = WEIGHT_SPECS["ResNet50_Weights.SENTINEL2_ALL_MOCO"]
        assert spec.band_names() == S2_ALL_NAMES
        assert spec.bands[8].sensor_code == "B8A"  # torchgeo checkpoint channel 8
        assert spec.normalization.means == (0.0,) * 13
        assert spec.normalization.stds == (10000.0,) * 13

    def test_sentinel2_rgb_moco_the_928_weight(self):
        spec = WEIGHT_SPECS["ResNet50_Weights.SENTINEL2_RGB_MOCO"]
        assert spec.band_names() == ("RED", "GREEN", "BLUE")
        assert [b.sensor_code for b in spec.bands] == ["B04", "B03", "B02"]
        assert spec.normalization.stds == (10000.0,) * 3

    def test_sentinel1_all_moco_sar_typing(self):
        spec = WEIGHT_SPECS["ResNet50_Weights.SENTINEL1_ALL_MOCO"]
        assert spec.band_names() == ("VV", "VH")
        assert spec.normalization.means == (-12.59, -20.26)
        assert spec.normalization.stds == (5.26, 5.91)
        for band, pol in zip(spec.bands, (Polarization.VV, Polarization.VH)):
            assert band.modality is Modality.SAR
            assert band.sar.polarization is pol
            assert band.sar.frequency_band is SARFrequencyBand.C
            assert band.sar.representation is Representation.DB
            assert band.sar.center_frequency_ghz == 5.405

    def test_dofa_dynamic(self):
        spec = WEIGHT_SPECS["DOFABase16_Weights.DOFA_MAE"]
        assert spec.dynamic_bands
        assert spec.bands == ()
        assert spec.normalization is None

    def test_all_entries_json_serializable(self):
        for key, spec in WEIGHT_SPECS.items():
            restored = SpectralSpec.from_json(spec.to_json())
            assert restored == spec, key


class TestSpecForWeightsLadder:
    def test_none_weights(self):
        assert spec_for_weights(None) is None

    def test_embedded_meta_dict_wins_over_registry(self):
        """Ladder step 2: the future torchgeo-native path takes precedence."""
        custom = SpectralSpec(
            bands=(BandSpec(name="RED", modality=Modality.OPTICAL, sensor_code="B04"),),
            source="embedded",
        )
        fake = Mock()
        fake.meta = {SPECTRAL_SPEC_META_KEY: custom.to_dict()}
        assert spec_for_weights(fake) == custom

    def test_malformed_embedded_spec_never_raises(self):
        fake = Mock()
        fake.meta = {SPECTRAL_SPEC_META_KEY: {"bands": [{"no_name": True}]}, "bands": ["B02"]}
        with pytest.warns(UserWarning, match="malformed"):
            spec = spec_for_weights(fake)
        # falls through to the identity path built from meta['bands']
        assert spec is not None and spec.band_names() == ("BLUE",)

    def test_registry_by_enum_identity(self):
        torchgeo_resnet = pytest.importorskip("torchgeo.models.resnet")
        weights = torchgeo_resnet.ResNet50_Weights.SENTINEL2_RGB_MOCO
        spec = spec_for_weights(weights)
        assert spec is WEIGHT_SPECS["ResNet50_Weights.SENTINEL2_RGB_MOCO"]

    def test_mock_name_falls_through_to_identity_path(self):
        """Mock objects have a Mock .name attribute, not str — must not match
        the registry, must still get an identity spec from meta['bands']."""
        fake = Mock()
        fake.meta = {"bands": ["B4", "B3", "B2"]}
        fake.name = Mock()  # not a str
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # unhinted B4/B3/B2 ambiguity warnings
            spec = spec_for_weights(fake)
        assert spec.band_names() == ("RED", "GREEN", "BLUE")
        assert spec.normalization is None  # identity path carries no stats

    def test_meta_without_bands_returns_none(self):
        fake = Mock()
        fake.meta = {"dataset": "something"}
        fake.name = Mock()
        assert spec_for_weights(fake) is None

    def test_sensor_hint_derived_from_weight_name(self):
        """A str .name like SENTINEL2_RGB_MOCO hints the catalog sensor, so
        un-padded codes resolve unambiguously (no warning)."""

        class FakeWeights:  # plain class: real str name, no Mock magic
            name = "SENTINEL2_RGB_MOCO"
            meta = {"bands": ["B4", "B3", "B2"]}

        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("error")  # any ambiguity warning fails the test
            spec = spec_for_weights(FakeWeights())
        assert spec.band_names() == ("RED", "GREEN", "BLUE")


class TestSARRoundTrip:
    """Deliverable 4: a complex-representation Sentinel-1 spec registers,
    serializes, and round-trips through the exact channel a future torchgeo
    WeightsEnum would use — with no complex-valued backbone existing yet."""

    def _slc_spec(self) -> SpectralSpec:
        def band(name: str, pol: Polarization) -> BandSpec:
            return BandSpec(
                name=name,
                modality=Modality.SAR,
                sensor_code=name,
                sar=SARInfo(
                    polarization=pol,
                    frequency_band=SARFrequencyBand.C,
                    representation=Representation.COMPLEX,
                    center_frequency_ghz=5.405,
                ),
            )

        return SpectralSpec(
            bands=(band("VV", Polarization.VV), band("VH", Polarization.VH)),
            source="demo:ResNet50_Weights.SENTINEL1_SLC_DEMO",
        )

    def test_register_and_lookup(self):
        spec = self._slc_spec()
        register_weight_spec("ResNet50_Weights.SENTINEL1_SLC_DEMO", spec)
        try:

            class FakeWeights:
                name = "SENTINEL1_SLC_DEMO"
                meta = {"bands": ["VV", "VH"]}

            FakeWeights.__name__ = "ResNet50_Weights"
            resolved = spec_for_weights(FakeWeights())
            assert resolved is spec
        finally:
            del WEIGHT_SPECS["ResNet50_Weights.SENTINEL1_SLC_DEMO"]

    def test_json_round_trip_preserves_complex_representation(self):
        spec = self._slc_spec()
        payload = json.dumps(spec.to_dict())
        restored = SpectralSpec.from_dict(json.loads(payload))
        assert restored == spec
        assert restored.bands[0].sar.representation is Representation.COMPLEX
        assert restored.bands[0].sar.polarization is Polarization.VV
        assert restored.bands[1].sar.polarization is Polarization.VH
        assert restored.bands[0].sar.frequency_band is SARFrequencyBand.C

    def test_round_trip_via_weights_meta_channel(self):
        """Embed in meta['spectral_spec'] exactly as torchgeo would ship it."""
        spec = self._slc_spec()
        fake = Mock()
        fake.meta = {SPECTRAL_SPEC_META_KEY: spec.to_dict(), "bands": ["VV", "VH"]}
        recovered = spec_for_weights(fake)
        assert recovered == spec
        assert recovered.bands[0].sar.representation is Representation.COMPLEX

    def test_db_and_complex_variants_coexist(self):
        db_spec = WEIGHT_SPECS["ResNet50_Weights.SENTINEL1_ALL_MOCO"]
        slc_spec = self._slc_spec()
        assert db_spec.bands[0].sar.representation is Representation.DB
        assert slc_spec.bands[0].sar.representation is Representation.COMPLEX
        # identical band identity, different representation: exactly the
        # distinction the bare string "VV" could not express
        assert db_spec.band_names() == slc_spec.band_names()


class TestNormalizationEquivalenceVsTorchgeo:
    """Pin registry constants against the real bound transforms. torchgeo
    Weights.transforms construct from meta alone — no checkpoint download."""

    @staticmethod
    def _apply_weight_transforms(weights, x):
        """Call a torchgeo transforms pipeline, tolerating the dict-vs-tensor
        call conventions that vary across torchgeo/kornia versions."""
        transforms = weights.transforms
        if not callable(transforms):
            transforms = transforms()
        try:
            out = transforms(x)
        except (TypeError, KeyError):
            out = transforms({"image": x})
        if isinstance(out, dict):
            out = out["image"]
        return out

    @pytest.mark.parametrize(
        ("weight_name", "num_channels"),
        [
            ("SENTINEL2_ALL_MOCO", 13),
            ("SENTINEL2_ALL_DINO", 13),
            ("SENTINEL2_ALL_DECUR", 13),
            ("SENTINEL2_RGB_MOCO", 3),
            ("SENTINEL1_ALL_MOCO", 2),
            ("SENTINEL1_ALL_DECUR", 2),
        ],
    )
    def test_registry_matches_bound_transform(self, weight_name, num_channels):
        """Extract the pipeline's effective per-channel affine constants by
        feeding constant images (invariant under any resize/crop geometry):
        pipeline(c) = (c - mean) / std, so two constants recover mean and std
        exactly, regardless of transform framework or version."""
        torch = pytest.importorskip("torch")
        torchgeo_resnet = pytest.importorskip("torchgeo.models.resnet")
        weights = getattr(torchgeo_resnet.ResNet50_Weights, weight_name)
        spec = spec_for_weights(weights)
        assert spec is not None and spec.normalization is not None
        assert len(spec.bands) == num_channels

        c0, c1 = 100.0, 300.0  # arbitrary distinct constants
        out0 = self._apply_weight_transforms(
            weights, torch.full((1, num_channels, 256, 256), c0)
        )[0, :, 0, 0]
        out1 = self._apply_weight_transforms(
            weights, torch.full((1, num_channels, 256, 256), c1)
        )[0, :, 0, 0]
        # out = (c - mean)/std  =>  std = (c1-c0)/(out1-out0), mean = c0 - out0*std
        extracted_std = (c1 - c0) / (out1 - out0)
        extracted_mean = c0 - out0 * extracted_std

        torch.testing.assert_close(
            extracted_mean, torch.tensor(spec.normalization.means), rtol=1e-4, atol=1e-4
        )
        torch.testing.assert_close(
            extracted_std, torch.tensor(spec.normalization.stds), rtol=1e-4, atol=1e-4
        )

    def test_dofa_transform_is_geometry_only(self):
        """DOFA's spec deliberately has normalization=None: its bound
        transform must not change values, only crop."""
        torch = pytest.importorskip("torch")
        dofa = pytest.importorskip("torchgeo.models.dofa")
        weights = dofa.DOFABase16_Weights.DOFA_MAE
        spec = spec_for_weights(weights)
        assert spec.normalization is None and spec.dynamic_bands

        torch.manual_seed(0)
        x = torch.rand(1, 4, 256, 256) * 500
        out = self._apply_weight_transforms(weights, x)
        # center crop of the input: values unchanged
        top = (256 - out.shape[-2]) // 2
        left = (256 - out.shape[-1]) // 2
        torch.testing.assert_close(
            out, x[..., top : top + out.shape[-2], left : left + out.shape[-1]]
        )


class TestFactoryResolutionForIssue928:
    """End-to-end resolver behavior for the exact #928 configuration."""

    def test_rgb_bands_resolve_with_indices_and_normalization(self):
        spec = WEIGHT_SPECS["ResNet50_Weights.SENTINEL2_RGB_MOCO"]
        res = resolve_bands(["R", "G", "B"], spec)
        assert res.indices == (0, 1, 2)  # R->B4 ckpt ch 0, G->B3 ch 1, B->B2 ch 2
        assert res.missing == ()
        assert res.means == (0.0, 0.0, 0.0)
        assert res.stds == (10000.0, 10000.0, 10000.0)
        assert res.pretrained_band_names == ("RED", "GREEN", "BLUE")
