"""Canonical per-sensor band definitions and band-name canonicalization.

This is the single replacement for the three byte-identical ``look_up_table``
dicts (torchgeo_resnet/vit/swin_satlas) and DOFA's ``waves_list``. Stdlib-only
so it can be lifted verbatim into torchgeo (see RFC 0001).

Sentinel-2 wavelengths follow torchgeo's authoritative table
(``torchgeo.datasets.Sentinel2.wavelengths``, um). Note this corrects the
hand-rounded values of the old DOFA ``waves_list`` by at most 0.005 um.
"""

from __future__ import annotations

import warnings
from enum import Enum

from terratorch.spectral.schema import (
    BandSpec,
    Modality,
    Polarization,
    Representation,
    SARFrequencyBand,
    SARInfo,
)

__all__ = [
    "SENTINEL2",
    "HLS",
    "SENTINEL1",
    "AUXILIARY",
    "LANDSAT_TM_TOA",
    "LANDSAT_ETM_TOA",
    "LANDSAT_ETM_SR",
    "LANDSAT_OLI_TIRS_TOA",
    "LANDSAT_OLI_SR",
    "SENSORS",
    "canonicalize_band",
    "canonical_name",
    "translate_bands",
]


def _optical(name: str, code: str | None, wavelength: float | None, *aliases: str) -> BandSpec:
    return BandSpec(
        name=name,
        modality=Modality.OPTICAL,
        sensor_code=code,
        aliases=tuple(aliases),
        wavelength_um=wavelength,
    )


def _s1(name: str, pol: Polarization | None, *aliases: str, orbit: str | None = None) -> BandSpec:
    return BandSpec(
        name=name,
        modality=Modality.SAR,
        sensor_code=name if pol is not None and orbit is None else None,
        aliases=tuple(aliases),
        sar=SARInfo(
            polarization=pol,
            frequency_band=SARFrequencyBand.C,
            representation=Representation.DB,
            center_frequency_ghz=5.405,
            orbit=orbit,
        ),
    )


# Sentinel-2 MSI. Canonical codes are the padded dataset-side spelling
# (torchgeo Sentinel2.all_bands); aliases cover the un-padded model-side
# spelling of torchgeo WeightsEnum meta ('B1', lowercase 'B8a') and plain RGB.
SENTINEL2: tuple[BandSpec, ...] = (
    _optical("COASTAL_AEROSOL", "B01", 0.4427, "B1"),
    _optical("BLUE", "B02", 0.4927, "B2", "B"),
    _optical("GREEN", "B03", 0.5598, "B3", "G"),
    _optical("RED", "B04", 0.6646, "B4", "R"),
    _optical("RED_EDGE_1", "B05", 0.7041, "B5"),
    _optical("RED_EDGE_2", "B06", 0.7405, "B6"),
    _optical("RED_EDGE_3", "B07", 0.7828, "B7"),
    _optical("NIR_BROAD", "B08", 0.8328, "B8"),
    _optical("NIR_NARROW", "B8A", 0.8647, "B8a"),
    _optical("WATER_VAPOR", "B09", 0.9451, "B9"),
    _optical("CIRRUS", "B10", 1.3735),
    _optical("SWIR_1", "B11", 1.6137),
    _optical("SWIR_2", "B12", 2.2024),
)

# Harmonized Landsat Sentinel: the S2-style vocabulary plus the two
# Landsat-sourced thermal bands of L30 products (TerraTorch's HLSBands enum).
HLS: tuple[BandSpec, ...] = SENTINEL2 + (
    _optical("THERMAL_INFRARED_1", None, 10.90),
    _optical("THERMAL_INFRARED_2", None, 12.00, "THERMAL_INFRARED_12"),  # legacy waves_list typo
)

# Sentinel-1 C-band SAR. Includes TerraTorch's orbit composites (SARBands) and
# the VV/VH ratio channel ('VV-VH' is the legacy DOFA waves_list spelling).
SENTINEL1: tuple[BandSpec, ...] = (
    _s1("VV", Polarization.VV),
    _s1("VH", Polarization.VH),
    _s1("HH", Polarization.HH),
    _s1("HV", Polarization.HV),
    _s1("ASC_VV", Polarization.VV, orbit="ascending"),
    _s1("ASC_VH", Polarization.VH, orbit="ascending"),
    _s1("DSC_VV", Polarization.VV, orbit="descending"),
    _s1("DSC_VH", Polarization.VH, orbit="descending"),
    _s1("VV_VH", None, "VV-VH"),
)

# Landsat band tables (codes as in torchgeo WeightsEnum meta; wavelengths um).
LANDSAT_TM_TOA: tuple[BandSpec, ...] = (
    _optical("BLUE", "B1", 0.485),
    _optical("GREEN", "B2", 0.560),
    _optical("RED", "B3", 0.660),
    _optical("NIR_BROAD", "B4", 0.830),
    _optical("SWIR_1", "B5", 1.650),
    _optical("THERMAL_INFRARED_1", "B6", 11.450),
    _optical("SWIR_2", "B7", 2.215),
)

LANDSAT_ETM_TOA: tuple[BandSpec, ...] = (
    _optical("BLUE", "B1", 0.485),
    _optical("GREEN", "B2", 0.565),
    _optical("RED", "B3", 0.660),
    _optical("NIR_BROAD", "B4", 0.825),
    _optical("SWIR_1", "B5", 1.650),
    # low/high gain acquisitions of the same thermal band
    _optical("B6_VCID_1", "B6_VCID_1", 11.450),
    _optical("B6_VCID_2", "B6_VCID_2", 11.450),
    _optical("SWIR_2", "B7", 2.220),
    _optical("PAN", "B8", 0.710),
)

LANDSAT_ETM_SR: tuple[BandSpec, ...] = (
    _optical("BLUE", "SR_B1", 0.485),
    _optical("GREEN", "SR_B2", 0.565),
    _optical("RED", "SR_B3", 0.660),
    _optical("NIR_BROAD", "SR_B4", 0.825),
    _optical("SWIR_1", "SR_B5", 1.650),
    _optical("SWIR_2", "SR_B7", 2.220),
)

LANDSAT_OLI_TIRS_TOA: tuple[BandSpec, ...] = (
    _optical("COASTAL_AEROSOL", "B1", 0.443),
    _optical("BLUE", "B2", 0.482),
    _optical("GREEN", "B3", 0.5615),
    _optical("RED", "B4", 0.655),
    _optical("NIR_NARROW", "B5", 0.865),
    _optical("SWIR_1", "B6", 1.610),
    _optical("SWIR_2", "B7", 2.200),
    _optical("PAN", "B8", 0.590),
    _optical("CIRRUS", "B9", 1.375),
    _optical("THERMAL_INFRARED_1", "B10", 10.895),
    _optical("THERMAL_INFRARED_2", "B11", 12.005),
)

LANDSAT_OLI_SR: tuple[BandSpec, ...] = (
    _optical("COASTAL_AEROSOL", "SR_B1", 0.443),
    _optical("BLUE", "SR_B2", 0.482),
    _optical("GREEN", "SR_B3", 0.5615),
    _optical("RED", "SR_B4", 0.655),
    _optical("NIR_NARROW", "SR_B5", 0.865),
    _optical("SWIR_1", "SR_B6", 1.610),
    _optical("SWIR_2", "SR_B7", 2.200),
)

# Non-spectral layers TerraTorch declares as bands (MetadataBands enum).
AUXILIARY: tuple[BandSpec, ...] = (
    BandSpec(name="DEM", modality=Modality.DEM),
    BandSpec(name="NDVI", modality=Modality.DERIVED),
    BandSpec(name="LULC", modality=Modality.MASK),
)

# Search order matters: unhinted un-padded codes ('B1') resolve to Sentinel-2
# first, matching the legacy look_up_table's S2-only worldview.
SENSORS: dict[str, tuple[BandSpec, ...]] = {
    "sentinel2": SENTINEL2,
    "hls": HLS,
    "sentinel1": SENTINEL1,
    "landsat_tm_toa": LANDSAT_TM_TOA,
    "landsat_etm_toa": LANDSAT_ETM_TOA,
    "landsat_etm_sr": LANDSAT_ETM_SR,
    "landsat_oli_tirs_toa": LANDSAT_OLI_TIRS_TOA,
    "landsat_oli_sr": LANDSAT_OLI_SR,
    "auxiliary": AUXILIARY,
}

# Dotted-prefix spellings ("SENTINEL2.B02") and other sensor hints.
_SENSOR_ALIASES: dict[str, str] = {
    "s2": "sentinel2",
    "sentinel-2": "sentinel2",
    "s1": "sentinel1",
    "sentinel-1": "sentinel1",
}


def _resolve_sensor(hint: str) -> str | None:
    key = hint.casefold().replace(" ", "")
    if key in SENSORS:
        return key
    return _SENSOR_ALIASES.get(key)


def _find_in(table: tuple[BandSpec, ...], name: str) -> BandSpec | None:
    for band in table:
        if band.matches(name):
            return band
    return None


def canonicalize_band(name: str | Enum, *, sensor: str | None = None) -> BandSpec | None:
    """Resolve a band spelling to its canonical :class:`BandSpec`.

    Accepts enums (matched by ``.value``), dotted prefixes used by torchgeo
    weight metadata ("SENTINEL2.B02" — the prefix becomes a sensor hint), and
    case-insensitive aliases ("B4", "b04", "R" -> Sentinel-2 RED).

    Un-padded codes B1-B8 are ambiguous between Sentinel-2 and Landsat; with a
    ``sensor`` hint resolution is exact, otherwise Sentinel-2 wins (matching
    the legacy ``look_up_table``) and a warning is emitted. Returns None when
    nothing matches — callers decide whether that is an error.
    """
    if isinstance(name, Enum):
        name = str(name.value)
    name = name.strip()
    if "." in name:
        prefix, _, rest = name.rpartition(".")
        hinted = _resolve_sensor(prefix)
        if sensor is None and hinted is not None:
            sensor = hinted
        name = rest
    if not name:
        return None

    if sensor is not None:
        sensor_key = _resolve_sensor(sensor)
        if sensor_key is None:
            msg = f"Unknown sensor {sensor!r}; known: {sorted(SENSORS)}"
            raise ValueError(msg)
        return _find_in(SENSORS[sensor_key], name)

    matches: list[BandSpec] = []
    for table in SENSORS.values():
        band = _find_in(table, name)
        if band is not None:
            matches.append(band)
    if not matches:
        return None
    distinct_names = {b.name for b in matches}
    exact_code = (
        matches[0].sensor_code is not None
        and name.casefold() == matches[0].sensor_code.casefold()
    )
    if len(distinct_names) > 1 and not exact_code:
        warnings.warn(
            f"Band {name!r} is ambiguous across sensors ({sorted(distinct_names)}); "
            f"resolving to {matches[0].name!r} (Sentinel-2 convention). "
            "Pass a sensor hint to disambiguate.",
            stacklevel=2,
        )
    return matches[0]


def canonical_name(name: str | Enum, *, sensor: str | None = None) -> str | None:
    """Canonical semantic name for a band spelling, or None if unknown."""
    band = canonicalize_band(name, sensor=sensor)
    return band.name if band is not None else None


def translate_bands(
    bands: list[str] | tuple[str, ...],
    *,
    sensor: str | None = None,
) -> list[str]:
    """Translate a list of band spellings to canonical semantic names.

    Drop-in, tolerant replacement for the legacy ``get_pretrained_bands``:
    unknown spellings warn and pass through unchanged (downstream weight
    selection then simply finds no match and leaves that channel randomly
    initialized — the pre-existing behavior for genuinely unknown bands)
    instead of raising ``KeyError``.
    """
    out: list[str] = []
    for band in bands:
        canonical = canonical_name(band, sensor=sensor)
        if canonical is None:
            warnings.warn(
                f"Unknown band {band!r}: not found in the spectral band catalog; "
                "passing it through unchanged.",
                stacklevel=2,
            )
            out.append(band if isinstance(band, str) else str(band))
        else:
            out.append(canonical)
    return out
