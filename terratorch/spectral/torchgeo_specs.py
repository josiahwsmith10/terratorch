"""SpectralSpec registry for torchgeo pretrained weights.

Bridges unmodified torchgeo ``WeightsEnum`` entries to the schema until
torchgeo ships specs natively in ``Weights.meta['spectral_spec']`` (RFC 0001
phase 2 — at which point entries here are deleted one by one and the
``meta``-embedded path in :func:`spec_for_weights` takes over automatically).

Normalization constants are copied from torchgeo's transform definitions
(``torchgeo/models/resnet.py`` etc.) and pinned by numerical-equivalence
tests against the real bound ``weights.transforms`` callables — see
``tests/test_spectral_torchgeo_specs.py``. Duck-typed against ``weights.meta``
/ ``weights.name``: no torchgeo import at runtime.
"""

from __future__ import annotations

import warnings
from typing import Any

from terratorch.spectral.catalog import SENSORS, SENTINEL1, SENTINEL2, canonicalize_band
from terratorch.spectral.schema import (
    SPECTRAL_SPEC_META_KEY,
    BandSpec,
    Modality,
    NormalizationSpec,
    SpectralSpec,
)

__all__ = ["WEIGHT_SPECS", "spec_for_weights", "register_weight_spec"]


def _sentinel2_spec(
    codes: tuple[str, ...],
    source: str,
    *,
    means: tuple[float, ...] | None = None,
    stds: tuple[float, ...] | None = None,
    note: str | None = None,
) -> SpectralSpec:
    by_code = {b.sensor_code: b for b in SENTINEL2}
    bands = tuple(by_code[c] for c in codes)
    normalization = None
    if means is not None and stds is not None:
        normalization = NormalizationSpec(means=means, stds=stds, note=note)
    return SpectralSpec(bands=bands, normalization=normalization, source=source)


_S2_ALL_CODES = ("B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B10", "B11", "B12")
# torchgeo _ssl4eo_s12_transforms_s2_10k: K.Normalize(mean=0, std=10000)
_S2_10K = {
    "means": (0.0,) * 13,
    "stds": (10000.0,) * 13,
    "note": "torchgeo _ssl4eo_s12_transforms_s2_10k: K.Normalize(mean=0, std=10000)",
}


def _ssl4eo_s2_all(source: str) -> SpectralSpec:
    return _sentinel2_spec(_S2_ALL_CODES, source, **_S2_10K)


def _ssl4eo_s2_rgb(source: str) -> SpectralSpec:
    return _sentinel2_spec(
        ("B04", "B03", "B02"),
        source,
        means=(0.0, 0.0, 0.0),
        stds=(10000.0, 10000.0, 10000.0),
        note="torchgeo _ssl4eo_s12_transforms_s2_10k: K.Normalize(mean=0, std=10000)",
    )


def _ssl4eo_s1_grd(source: str) -> SpectralSpec:
    by_name = {b.name: b for b in SENTINEL1}
    return SpectralSpec(
        bands=(by_name["VV"], by_name["VH"]),
        normalization=NormalizationSpec(
            means=(-12.59, -20.26),
            stds=(5.26, 5.91),
            note="torchgeo _ssl4eo_s12_transforms_s1; statistics are in dB domain",
        ),
        source=source,
    )


def _dofa(source: str) -> SpectralSpec:
    # Faithful to torchgeo: DOFA meta has no bands/in_chans (wavelength-
    # conditioned dynamic patch embed) and its transform is a CenterCrop only.
    return SpectralSpec(bands=(), dynamic_bands=True, normalization=None, source=source)


def _make_default_specs() -> dict[str, SpectralSpec]:
    # Members verified to exist in torchgeo (0.7.x and 0.8.x) with these
    # exact bands/normalizations; extending coverage is a data-only change.
    specs: dict[str, SpectralSpec] = {}
    # SSL4EO-S12 ResNets (torchgeo/models/resnet.py). The RGB entries are the
    # fix for terratorch issue #928.
    for key in (
        "ResNet18_Weights.SENTINEL2_ALL_MOCO",
        "ResNet50_Weights.SENTINEL2_ALL_MOCO",
        "ResNet50_Weights.SENTINEL2_ALL_DINO",
        "ResNet50_Weights.SENTINEL2_ALL_DECUR",
        "ViTSmall16_Weights.SENTINEL2_ALL_MOCO",  # vit.py _zhu_xlab_transforms
        "ViTSmall16_Weights.SENTINEL2_ALL_DINO",
    ):
        specs[key] = _ssl4eo_s2_all(f"torchgeo:{key}")
    for key in (
        "ResNet18_Weights.SENTINEL2_RGB_MOCO",
        "ResNet50_Weights.SENTINEL2_RGB_MOCO",
    ):
        specs[key] = _ssl4eo_s2_rgb(f"torchgeo:{key}")
    for key in (
        "ResNet50_Weights.SENTINEL1_ALL_MOCO",
        "ResNet50_Weights.SENTINEL1_ALL_DECUR",
    ):
        specs[key] = _ssl4eo_s1_grd(f"torchgeo:{key}")
    # DOFA (torchgeo/models/dofa.py)
    for arch in ("DOFABase16_Weights", "DOFALarge16_Weights"):
        key = f"{arch}.DOFA_MAE"
        specs[key] = _dofa(f"torchgeo:{key}")
    return specs


#: Registry keyed by torchgeo weight identity, "<EnumClassName>.<MEMBER_NAME>".
WEIGHT_SPECS: dict[str, SpectralSpec] = _make_default_specs()

#: Secondary lookup by checkpoint URL for weights not exposed as enum members.
WEIGHT_URL_SPECS: dict[str, SpectralSpec] = {}


def register_weight_spec(key: str, spec: SpectralSpec, *, url: str | None = None) -> None:
    """Register (or override) a spec for a weight identity.

    ``key`` is ``"<EnumClassName>.<MEMBER_NAME>"`` (e.g.
    ``"ResNet50_Weights.SENTINEL1_ALL_MOCO"``); ``url`` optionally also
    registers the checkpoint URL for non-enum weights.
    """
    WEIGHT_SPECS[key] = spec
    if url is not None:
        WEIGHT_URL_SPECS[url] = spec


def _sensor_hint_from_name(weight_name: str) -> str | None:
    """Infer a catalog sensor hint from a weight member name like
    ``SENTINEL2_RGB_MOCO`` or ``LANDSAT_TM_TOA_MOCO``."""
    name = weight_name.casefold()
    for sensor in SENSORS:
        if name.startswith(sensor):
            return sensor
    if name.startswith("sentinel2"):
        return "sentinel2"
    if name.startswith("sentinel1"):
        return "sentinel1"
    return None


def _identity_spec_from_meta(meta: dict[str, Any], weight_name: str | None) -> SpectralSpec | None:
    bands = meta.get("bands")
    if not isinstance(bands, (list, tuple)) or not bands:
        return None
    sensor = _sensor_hint_from_name(weight_name) if isinstance(weight_name, str) else None
    resolved: list[BandSpec] = []
    for band in bands:
        spec = canonicalize_band(str(band), sensor=sensor) if sensor else canonicalize_band(str(band))
        if spec is None:
            spec = BandSpec(name=str(band), modality=Modality.UNKNOWN)
        resolved.append(spec)
    return SpectralSpec(bands=tuple(resolved), source="meta:bands")


def spec_for_weights(weights: object | None) -> SpectralSpec | None:
    """Resolve a SpectralSpec for a (torchgeo) weights object. Never raises.

    Fallback ladder (RFC 0001 §4): embedded ``meta['spectral_spec']`` >
    registry by enum identity > registry by URL > identity-only spec derived
    from ``meta['bands']`` > None (callers keep legacy behavior).
    """
    if weights is None:
        return None

    meta = getattr(weights, "meta", None)
    if isinstance(meta, dict):
        embedded = meta.get(SPECTRAL_SPEC_META_KEY)
        if isinstance(embedded, SpectralSpec):
            return embedded
        if isinstance(embedded, dict):
            try:
                return SpectralSpec.from_dict(embedded)
            except Exception as err:  # malformed spec: fall through, never crash
                warnings.warn(
                    f"Ignoring malformed {SPECTRAL_SPEC_META_KEY!r} in weights meta: {err}",
                    stacklevel=2,
                )

    # guarded so Mock/duck weights without a real .name fall through
    name = getattr(weights, "name", None)
    if isinstance(name, str):
        key = f"{type(weights).__name__}.{name}"
        if key in WEIGHT_SPECS:
            return WEIGHT_SPECS[key]

    url = getattr(weights, "url", None)
    if isinstance(url, str) and url in WEIGHT_URL_SPECS:
        return WEIGHT_URL_SPECS[url]

    if isinstance(meta, dict):
        return _identity_spec_from_meta(meta, name if isinstance(name, str) else None)
    return None
