"""Map dataset-declared bands onto a weight's SpectralSpec.

One call answers both questions the dataset/model boundary has:
which checkpoint channels correspond to my bands (``indices`` /
``pretrained_band_names``) and how must those channels be normalized
(``means`` / ``stds``). Stdlib-only (see RFC 0001).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum

from terratorch.spectral.catalog import canonicalize_band
from terratorch.spectral.schema import SpectralSpec

__all__ = ["BandResolution", "resolve_bands"]


@dataclass(frozen=True)
class BandResolution:
    #: canonicalized requested names, input order
    requested: tuple[str, ...]
    #: per requested band: index into ``spec.bands`` (i.e. the checkpoint
    #: channel), or None when the spec does not provide that band
    indices: tuple[int | None, ...]
    #: subset of ``requested`` found in the spec, input order
    matched: tuple[str, ...]
    #: requested bands the spec lacks (downstream: randomly initialized)
    missing: tuple[str, ...]
    #: per requested band; 0.0/1.0 identity for missing bands; None when the
    #: spec carries no normalization at all
    means: tuple[float, ...] | None
    stds: tuple[float, ...] | None
    #: canonical names of ALL spec bands in spec (checkpoint) order — a
    #: drop-in ``pretrained_bands`` argument for select_patch_embed_weights
    pretrained_band_names: tuple[str, ...]


def resolve_bands(
    requested_bands: Sequence[str | Enum],
    spec: SpectralSpec,
    *,
    strict: bool = False,
) -> BandResolution:
    """Resolve ``requested_bands`` (any catalog spelling or enum) against ``spec``.

    Matching per requested band: canonicalize through the catalog where
    possible, then match against the spec's bands by name / sensor code /
    aliases (case-insensitive). With ``strict=True`` unresolved bands raise
    ``ValueError`` instead of being reported in ``missing``.
    """
    requested: list[str] = []
    indices: list[int | None] = []
    matched: list[str] = []
    missing: list[str] = []

    for band in requested_bands:
        raw = str(band.value) if isinstance(band, Enum) else str(band)
        canonical = canonicalize_band(raw)
        name = canonical.name if canonical is not None else raw
        requested.append(name)

        index = None
        for i, spec_band in enumerate(spec.bands):
            # match on the canonical name first, then let the spec's own
            # aliases catch spellings the catalog does not know
            if spec_band.matches(name) or spec_band.matches(raw):
                index = i
                break
        indices.append(index)
        if index is None:
            missing.append(name)
        else:
            matched.append(name)

    if strict and missing:
        msg = (
            f"Bands {missing} not provided by spec "
            f"{spec.source or spec.band_names()}; available: {list(spec.band_names())}"
        )
        raise ValueError(msg)

    means: tuple[float, ...] | None = None
    stds: tuple[float, ...] | None = None
    if spec.normalization is not None:
        means = tuple(
            spec.normalization.means[i] if i is not None else 0.0 for i in indices
        )
        stds = tuple(
            spec.normalization.stds[i] if i is not None else 1.0 for i in indices
        )

    return BandResolution(
        requested=tuple(requested),
        indices=tuple(indices),
        matched=tuple(matched),
        missing=tuple(missing),
        means=means,
        stds=stds,
        pretrained_band_names=spec.band_names(),
    )
