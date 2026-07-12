"""Typed spectral band metadata schema for the TorchGeo/TerraTorch boundary.

Self-contained (stdlib-only) prototype of RFC 0001
(docs/rfc/0001-spectral-band-metadata.md); proposed eventual home:
``torchgeo.spectral``.
"""

from terratorch.spectral.catalog import (
    SENSORS,
    canonical_name,
    canonicalize_band,
    translate_bands,
)
from terratorch.spectral.resolve import BandResolution, resolve_bands
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

__all__ = [
    "SPECTRAL_SPEC_META_KEY",
    "BandSpec",
    "BandResolution",
    "Modality",
    "NormalizationSpec",
    "Polarization",
    "Representation",
    "SARFrequencyBand",
    "SARInfo",
    "SpectralSpec",
    "SENSORS",
    "WEIGHT_SPECS",
    "canonical_name",
    "canonicalize_band",
    "register_weight_spec",
    "resolve_bands",
    "spec_for_weights",
    "translate_bands",
]
