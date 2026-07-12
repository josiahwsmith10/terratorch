"""Typed spectral band metadata schema for the TorchGeo/TerraTorch boundary.

See docs/rfc/0001-spectral-band-metadata.md. This module is intentionally
stdlib-only (no torch/torchgeo/terratorch imports) so the whole package can be
lifted verbatim into torchgeo (proposed home: ``torchgeo.spectral``).

Serialization contract: ``SpectralSpec.to_dict()`` produces a JSON-compatible
dict suitable for embedding in ``torchvision.models._api.Weights.meta`` under
the key ``"spectral_spec"``; ``from_dict`` ignores unknown keys so older
consumers tolerate newer producers.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

__all__ = [
    "SPECTRAL_SPEC_META_KEY",
    "SCHEMA_VERSION",
    "Modality",
    "Polarization",
    "SARFrequencyBand",
    "Representation",
    "SARInfo",
    "BandSpec",
    "NormalizationSpec",
    "SpectralSpec",
]

#: Key under which a serialized SpectralSpec dict lives in ``Weights.meta``.
SPECTRAL_SPEC_META_KEY = "spectral_spec"

#: Current schema version emitted by :meth:`SpectralSpec.to_dict`.
SCHEMA_VERSION = 1


class Modality(str, Enum):
    OPTICAL = "optical"
    SAR = "sar"
    DEM = "dem"
    DERIVED = "derived"  # computed products: NDVI, ...
    MASK = "mask"  # categorical layers: LULC, ...
    UNKNOWN = "unknown"


class Polarization(str, Enum):
    VV = "VV"
    VH = "VH"
    HH = "HH"
    HV = "HV"


class SARFrequencyBand(str, Enum):
    P = "P"
    L = "L"
    S = "S"
    C = "C"
    X = "X"
    KU = "Ku"
    K = "K"
    KA = "Ka"


class Representation(str, Enum):
    AMPLITUDE = "amplitude"  # linear amplitude
    INTENSITY = "intensity"  # power / sigma0, linear
    DB = "db"  # log-scaled (10*log10)
    COMPLEX = "complex"  # single-look complex (I/Q)


@dataclass(frozen=True)
class SARInfo:
    """SAR acquisition properties a bare band string like "VV" cannot express.

    ``polarization`` is None for composite/derived SAR bands (e.g. a VV/VH
    ratio channel) that have no single polarization.
    """

    polarization: Polarization | None
    frequency_band: SARFrequencyBand
    representation: Representation
    center_frequency_ghz: float | None = None  # 5.405 for Sentinel-1
    orbit: str | None = None  # "ascending" | "descending" for ASC_*/DSC_* bands

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "polarization": self.polarization.value if self.polarization is not None else None,
            "frequency_band": self.frequency_band.value,
            "representation": self.representation.value,
        }
        if self.center_frequency_ghz is not None:
            d["center_frequency_ghz"] = self.center_frequency_ghz
        if self.orbit is not None:
            d["orbit"] = self.orbit
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "SARInfo":
        return cls(
            polarization=Polarization(d["polarization"]) if d.get("polarization") is not None else None,
            frequency_band=SARFrequencyBand(d["frequency_band"]),
            representation=Representation(d["representation"]),
            center_frequency_ghz=d.get("center_frequency_ghz"),
            orbit=d.get("orbit"),
        )


@dataclass(frozen=True)
class BandSpec:
    """Identity of one band: sensor physics, reusable across weights.

    ``name`` uses TerraTorch's semantic enum values ("RED", "NIR_NARROW", "VV")
    so TerraTorch consumes specs without a translation layer; ``sensor_code``
    and ``aliases`` cover sensor-native spellings ("B04", "B4", "R", "B8a").
    """

    name: str
    modality: Modality
    sensor_code: str | None = None
    aliases: tuple[str, ...] = ()
    wavelength_um: float | None = None  # None where not meaningful (SAR, DEM)
    bandwidth_um: float | None = None
    sar: SARInfo | None = None

    def __post_init__(self) -> None:
        if self.modality is Modality.SAR and self.sar is None:
            msg = f"BandSpec {self.name!r}: modality is SAR but no SARInfo given"
            raise ValueError(msg)
        if self.modality is not Modality.SAR and self.sar is not None:
            msg = f"BandSpec {self.name!r}: SARInfo given but modality is {self.modality.value!r}"
            raise ValueError(msg)
        # normalize aliases passed as a list (e.g. straight from JSON)
        if not isinstance(self.aliases, tuple):
            object.__setattr__(self, "aliases", tuple(self.aliases))

    @property
    def dofa_wavelength_um(self) -> float | None:
        """Wavelength under DOFA's convention: optical bands report their center
        wavelength in um; SAR bands report the carrier frequency in GHz (5.405
        for Sentinel-1), reusing the same positional slot. Kept behind an
        explicitly named property rather than polluting ``wavelength_um``."""
        if self.wavelength_um is not None:
            return self.wavelength_um
        if self.sar is not None:
            return self.sar.center_frequency_ghz
        return None

    def matches(self, name: str) -> bool:
        """Case-insensitive match against name, sensor code, or any alias."""
        folded = name.casefold()
        if folded == self.name.casefold():
            return True
        if self.sensor_code is not None and folded == self.sensor_code.casefold():
            return True
        return any(folded == a.casefold() for a in self.aliases)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"name": self.name, "modality": self.modality.value}
        if self.sensor_code is not None:
            d["sensor_code"] = self.sensor_code
        if self.aliases:
            d["aliases"] = list(self.aliases)
        if self.wavelength_um is not None:
            d["wavelength_um"] = self.wavelength_um
        if self.bandwidth_um is not None:
            d["bandwidth_um"] = self.bandwidth_um
        if self.sar is not None:
            d["sar"] = self.sar.to_dict()
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "BandSpec":
        return cls(
            name=d["name"],
            modality=Modality(d["modality"]),
            sensor_code=d.get("sensor_code"),
            aliases=tuple(d.get("aliases", ())),
            wavelength_um=d.get("wavelength_um"),
            bandwidth_um=d.get("bandwidth_um"),
            sar=SARInfo.from_dict(d["sar"]) if d.get("sar") is not None else None,
        )


@dataclass(frozen=True)
class NormalizationSpec:
    """Per-channel normalization: ``out = (x - mean) / std`` — exactly
    ``kornia.augmentation.Normalize`` semantics, so values are directly
    checkable against a torchgeo weight's bound transform. Scale-only
    normalization ("divide by 10000") encodes as mean=0.0, std=10000.0.

    Statistics belong to one training run, hence they live here (index-aligned
    with ``SpectralSpec.bands``) and not on the reusable :class:`BandSpec`.
    """

    means: tuple[float, ...]
    stds: tuple[float, ...]
    note: str | None = None  # provenance, e.g. "torchgeo _ssl4eo_s12_transforms_s2_10k"

    def __post_init__(self) -> None:
        if not isinstance(self.means, tuple):
            object.__setattr__(self, "means", tuple(self.means))
        if not isinstance(self.stds, tuple):
            object.__setattr__(self, "stds", tuple(self.stds))
        if len(self.means) != len(self.stds):
            msg = f"means ({len(self.means)}) and stds ({len(self.stds)}) differ in length"
            raise ValueError(msg)
        if any(s == 0 for s in self.stds):
            msg = "stds must be non-zero"
            raise ValueError(msg)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"means": list(self.means), "stds": list(self.stds)}
        if self.note is not None:
            d["note"] = self.note
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "NormalizationSpec":
        return cls(means=tuple(d["means"]), stds=tuple(d["stds"]), note=d.get("note"))


@dataclass(frozen=True)
class SpectralSpec:
    """The full spec bound to one weight: ordered bands + their normalization.

    ``dynamic_bands=True`` describes wavelength-conditioned models (DOFA) that
    accept arbitrary bands described by wavelength; such specs typically have
    ``bands=()``.
    """

    bands: tuple[BandSpec, ...]
    normalization: NormalizationSpec | None = None
    dynamic_bands: bool = False
    source: str | None = None  # e.g. "torchgeo:ResNet50_Weights.SENTINEL2_ALL_MOCO"
    version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.bands, tuple):
            object.__setattr__(self, "bands", tuple(self.bands))
        if self.normalization is not None and len(self.normalization.means) != len(self.bands):
            msg = (
                f"normalization has {len(self.normalization.means)} channels "
                f"but spec has {len(self.bands)} bands"
            )
            raise ValueError(msg)

    def band_names(self) -> tuple[str, ...]:
        return tuple(b.name for b in self.bands)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "version": self.version,
            "bands": [b.to_dict() for b in self.bands],
        }
        if self.normalization is not None:
            d["normalization"] = self.normalization.to_dict()
        if self.dynamic_bands:
            d["dynamic_bands"] = True
        if self.source is not None:
            d["source"] = self.source
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "SpectralSpec":
        norm = d.get("normalization")
        return cls(
            bands=tuple(BandSpec.from_dict(b) for b in d.get("bands", ())),
            normalization=NormalizationSpec.from_dict(norm) if norm is not None else None,
            dynamic_bands=bool(d.get("dynamic_bands", False)),
            source=d.get("source"),
            version=int(d.get("version", SCHEMA_VERSION)),
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, s: str) -> "SpectralSpec":
        return cls.from_dict(json.loads(s))
