# Copyright contributors to the Terratorch project
"""First-class terratorch backbones for OlmoEarth (Ai2), v1 / v1.1 / v1.2.

Requires the optional dependency ``olmoearth-pretrain-minimal`` (install with
``pip install terratorch[olmoearth]``; needs Python >= 3.11 and torch >= 2.7). This module
is imported behind a guard in ``terratorch.models.backbones.__init__`` so base installs
are unaffected.

OlmoEarth groups each modality's bands into band *sets* (for Sentinel-2 L2A by native
resolution: [B02,B03,B04,B08], [B05,B06,B07,B8A,B11,B12], [B01,B09]); each set becomes a
token and masking happens at set granularity. Band subsetting is therefore handled
natively by declaring complete sets — undeclared sets are marked MISSING. Because the
patch-embed weights are per-set, ``select_patch_embed_weights`` does not apply; slicing
within a set is a possible follow-up. Band sets are read from the loaded model's
tokenization config, never hardcoded, so checkpoints that override the default grouping
keep working.

Input metadata (all optional): bands via ``model_bands`` (terratorch names such as
BLUE/GREEN/... or native names B02..B8A/vv/vh; all declared bands must belong to a single
modality — Sentinel-2 L2A or Sentinel-1); acquisition ``timestamps`` ([day, month(0-11),
year], default 2023-06-15) at construction or per batch as a forward kwarg; token
``patch_size`` (1-8) at construction.

Upstream has no official API for intermediate layer outputs (``token_exit_cfg`` is a
per-modality early exit, not multi-layer capture), so ``out_indices`` is implemented with
standard forward hooks on ``encoder.blocks[i]``. With ``fast_pass=True`` and
``use_flash_attn=False`` (the default) the encoder keeps a dense, order-preserving
[B, N, D] token sequence inside the block loop, with N = H'*W'*T*S in row-major order —
this is asserted at runtime.
"""

import logging

import torch
from einops import rearrange
from torch import nn

from olmoearth_pretrain_minimal import ModelID, OlmoEarthPretrain_v1, load_model_from_id
from olmoearth_pretrain_minimal.olmoearth_pretrain_v1.utils.constants import Modality
from olmoearth_pretrain_minimal.olmoearth_pretrain_v1.utils.datatypes import (
    MaskedOlmoEarthSample,
    MaskValue,
)

from terratorch.registry import TERRATORCH_BACKBONE_REGISTRY

logger = logging.getLogger(__name__)

# terratorch band names -> native OlmoEarth band names, per modality.
# Native names pass through.
OLMOEARTH_BAND_ALIASES = {
    Modality.SENTINEL2_L2A.name: {
        "COASTAL_AEROSOL": "B01",
        "BLUE": "B02",
        "GREEN": "B03",
        "RED": "B04",
        "RED_EDGE_1": "B05",
        "RED_EDGE_2": "B06",
        "RED_EDGE_3": "B07",
        "NIR_BROAD": "B08",
        "NIR_NARROW": "B8A",
        "WATER_VAPOR": "B09",
        "SWIR_1": "B11",
        "SWIR_2": "B12",
    },
    Modality.SENTINEL1.name: {
        "VV": "vv",
        "VH": "vh",
    },
}

_MODALITY_SPECS = {
    Modality.SENTINEL2_L2A.name: Modality.SENTINEL2_L2A,
    Modality.SENTINEL1.name: Modality.SENTINEL1,
}

DEFAULT_TIMESTAMP = (15, 5, 2023)  # day, month (0-indexed: June), year


def _resolve_modality_and_bands(model_bands: list[str]) -> tuple[str, list[str]]:
    """Find the single modality all declared bands belong to; return native band names.

    Multi-modality inputs are not supported: the upstream encoder concatenates modality
    tokens in an order derived from a set (not contractually stable), which would make
    hook-based intermediate features ambiguous.
    """
    failures = {}
    for modality_name, spec in _MODALITY_SPECS.items():
        aliases = OLMOEARTH_BAND_ALIASES[modality_name]
        band_order = list(spec.band_order)
        native = []
        for band in model_bands:
            name = aliases.get(str(band), str(band))
            if name not in band_order:
                failures[modality_name] = f"{band!r} is not a {modality_name} band"
                break
            native.append(name)
        else:
            if len(set(native)) != len(native):
                msg = f"Duplicate bands declared: {model_bands}."
                raise ValueError(msg)
            return modality_name, native
    msg = (
        f"Declared bands {model_bands} do not all belong to a single OlmoEarth modality "
        f"({list(_MODALITY_SPECS)}): {failures}. Mixing modalities is not supported."
    )
    raise ValueError(msg)


class OlmoEarthEncoderWrapper(nn.Module):
    """Wraps an OlmoEarth encoder to the terratorch backbone contract.

    forward(x, **kwargs) takes ``x`` of shape [B, C, H, W] (or [B, C, T, H, W]) and
    returns a list of token tensors [B, N, D], one per entry of ``out_indices``. Tokens
    are pooled over the visible band sets and time; there is no CLS token.
    """

    has_cls_token = False

    def __init__(
        self,
        encoder: nn.Module,
        model_bands: list[str],
        out_indices: list[int] | None = None,
        patch_size: int = 4,
        timestamps: tuple[int, int, int] | list[int] | None = None,
    ) -> None:
        super().__init__()
        if getattr(encoder, "use_flash_attn", False):
            msg = (
                "OlmoEarthEncoderWrapper requires use_flash_attn=False: flash attention "
                "packs tokens inside the block loop, which breaks hook-based intermediate "
                "feature capture."
            )
            raise ValueError(msg)
        self.encoder = encoder
        self.patch_size = patch_size

        self.modality, native_bands = _resolve_modality_and_bands(model_bands)
        spec = _MODALITY_SPECS[self.modality]
        band_order = list(spec.band_order)
        self.channel_indices = [band_order.index(b) for b in native_bands]
        self.num_channels_total = len(band_order)

        # band sets come from the model's own tokenization config, so checkpoints with a
        # custom band grouping (a knob in v1.1+/upstream) are handled transparently
        bandset_indices = encoder.tokenization_config.get_bandset_indices(self.modality)
        declared = set(self.channel_indices)
        set_mask_values = []
        for set_channel_indices in bandset_indices:
            n_declared = len(declared & set(set_channel_indices))
            if n_declared == len(set_channel_indices):
                set_mask_values.append(MaskValue.ONLINE_ENCODER.value)
            elif n_declared == 0:
                set_mask_values.append(MaskValue.MISSING.value)
            else:
                set_bands = [band_order[i] for i in set_channel_indices]
                msg = (
                    f"OlmoEarth band set {set_bands} is only partially declared. Band sets "
                    f"are masked as a whole, so declare all of them or none of them."
                )
                raise ValueError(msg)
        self.register_buffer("set_mask_values", torch.tensor(set_mask_values, dtype=torch.float32))
        self.visible_set_indices = [
            i for i, v in enumerate(set_mask_values) if v == MaskValue.ONLINE_ENCODER.value
        ]
        if not self.visible_set_indices:
            msg = "No OlmoEarth band sets are covered by the declared bands."
            raise ValueError(msg)

        self.num_register_tokens = int(getattr(encoder, "num_register_tokens", 0) or 0)

        n_blocks = len(encoder.blocks)
        out_indices = out_indices if out_indices else [-1]
        self.out_indices = [i % n_blocks for i in out_indices]
        self.out_channels = [encoder.embedding_size] * len(self.out_indices)

        timestamps = timestamps if timestamps is not None else DEFAULT_TIMESTAMP
        self.register_buffer("default_timestamp", torch.tensor(timestamps, dtype=torch.long))

    def _build_timestamps(self, timestamps, b: int, t: int, device) -> torch.Tensor:
        if timestamps is None:
            return self.default_timestamp.to(device).expand(b, t, 3)
        if not isinstance(timestamps, torch.Tensor):
            timestamps = torch.tensor(timestamps, dtype=torch.long, device=device)
        if timestamps.dim() == 1:
            timestamps = timestamps.expand(b, t, 3)
        elif timestamps.dim() == 2:
            timestamps = timestamps.expand(b, t, 3) if timestamps.shape[0] != b else timestamps.unsqueeze(1).expand(b, t, 3)
        return timestamps

    def forward(self, x: torch.Tensor, timestamps: torch.Tensor | None = None, **kwargs) -> list[torch.Tensor]:
        if x.dim() == 4:
            x = x.unsqueeze(2)  # [B, C, H, W] -> [B, C, T=1, H, W]
        b, c, t, h, w = x.shape
        if c != len(self.channel_indices):
            msg = f"Expected {len(self.channel_indices)} input channels (declared bands), got {c}."
            raise ValueError(msg)
        if h % self.patch_size != 0 or w % self.patch_size != 0:
            msg = f"Input size {h}x{w} must be divisible by patch_size {self.patch_size}."
            raise ValueError(msg)
        device, dtype = x.device, x.dtype

        data = torch.zeros(b, h, w, t, self.num_channels_total, dtype=dtype, device=device)
        data[..., self.channel_indices] = rearrange(x, "b c t h w -> b h w t c")
        mask = self.set_mask_values.to(device=device, dtype=dtype).expand(b, h, w, t, -1)

        sample = MaskedOlmoEarthSample(
            timestamps=self._build_timestamps(timestamps, b, t, device),
            **{
                self.modality: data,
                f"{self.modality}_mask": mask,
            },
        )

        last = len(self.encoder.blocks) - 1
        hook_indices = sorted({i for i in self.out_indices if i != last})
        captured: dict[int, torch.Tensor] = {}
        handles = [
            self.encoder.blocks[i].register_forward_hook(
                lambda module, args, output, i=i: captured.__setitem__(i, output)
            )
            for i in hook_indices
        ]
        try:
            output = self.encoder(sample, patch_size=self.patch_size, fast_pass=True)
        finally:
            for handle in handles:
                handle.remove()

        tokens = getattr(output["tokens_and_masks"], self.modality)  # [B, H', W', T, S, D]
        _, h_p, w_p, t_p, n_sets, dim = tokens.shape
        n_expected = h_p * w_p * t_p * n_sets

        features = []
        for i in self.out_indices:
            if i == last:
                feature = tokens
            else:
                hidden = captured[i]
                if hidden.shape[1] == n_expected + self.num_register_tokens and self.num_register_tokens:
                    hidden = hidden[:, self.num_register_tokens :]
                if hidden.shape[1] != n_expected:
                    msg = (
                        f"Cannot reshape intermediate tokens of length {hidden.shape[1]} to "
                        f"({h_p}, {w_p}, {t_p}, {n_sets}) = {n_expected} tokens "
                        f"(+{self.num_register_tokens} register tokens)."
                    )
                    raise RuntimeError(msg)
                feature = hidden.view(b, h_p, w_p, t_p, n_sets, dim)
            # masked (MISSING) sets carry no signal: pool over visible sets, then time
            feature = feature[..., self.visible_set_indices, :].mean(dim=(3, 4))
            features.append(rearrange(feature, "b h w d -> b (h w) d"))
        return features

    def freeze(self):
        for param in self.encoder.parameters():
            param.requires_grad_(False)


def _create_olmoearth(
    version: str,
    size: str,
    model_id: ModelID,
    model_bands: list[str],
    pretrained: bool = False,  # noqa: FBT001, FBT002
    out_indices: list[int] | None = None,
    patch_size: int = 4,
    timestamps: tuple[int, int, int] | None = None,
    **kwargs,
) -> OlmoEarthEncoderWrapper:
    if pretrained:
        # builds the architecture from the checkpoint's own config.json, then loads
        # weights. Never load pretrained state dicts into the random-init class below:
        # its tokenization / register-token configuration may differ.
        full_model = load_model_from_id(model_id)
        encoder = full_model.encoder
    else:
        try:
            full_model = OlmoEarthPretrain_v1(
                model_size=size,
                model_version=version,
                supported_modality_names=[Modality.SENTINEL2_L2A.name, Modality.SENTINEL1.name],
                **kwargs,
            )
            encoder = full_model.model.encoder
        except KeyError:
            if size != "small":
                raise
            # upstream bug in olmoearth-pretrain-minimal 0.0.6: v1.2 "small" is listed as
            # supported but has no patch-embed hidden-size entry, so offline random init
            # raises KeyError. Fall back to building from the HF config.json (downloads
            # the config only, no weights).
            logger.warning(
                "OlmoEarthPretrain_v1 cannot build %s/%s offline (upstream KeyError); "
                "falling back to load_model_from_id(load_weights=False), which downloads "
                "the checkpoint config.json.",
                version,
                size,
            )
            full_model = load_model_from_id(model_id, load_weights=False)
            encoder = full_model.encoder
    # keep only the online encoder; the LatentMIM decoder/target_encoder are not needed
    # for feature extraction and would otherwise stay referenced.
    return OlmoEarthEncoderWrapper(
        encoder,
        model_bands,
        out_indices=out_indices,
        patch_size=patch_size,
        timestamps=timestamps,
    )


@TERRATORCH_BACKBONE_REGISTRY.register
def olmoearth_v1_nano(model_bands: list[str], **kwargs) -> OlmoEarthEncoderWrapper:
    return _create_olmoearth("v1", "nano", ModelID.OLMOEARTH_V1_NANO, model_bands, **kwargs)


@TERRATORCH_BACKBONE_REGISTRY.register
def olmoearth_v1_tiny(model_bands: list[str], **kwargs) -> OlmoEarthEncoderWrapper:
    return _create_olmoearth("v1", "tiny", ModelID.OLMOEARTH_V1_TINY, model_bands, **kwargs)


@TERRATORCH_BACKBONE_REGISTRY.register
def olmoearth_v1_base(model_bands: list[str], **kwargs) -> OlmoEarthEncoderWrapper:
    return _create_olmoearth("v1", "base", ModelID.OLMOEARTH_V1_BASE, model_bands, **kwargs)


@TERRATORCH_BACKBONE_REGISTRY.register
def olmoearth_v1_large(model_bands: list[str], **kwargs) -> OlmoEarthEncoderWrapper:
    return _create_olmoearth("v1", "large", ModelID.OLMOEARTH_V1_LARGE, model_bands, **kwargs)


@TERRATORCH_BACKBONE_REGISTRY.register
def olmoearth_v1_1_nano(model_bands: list[str], **kwargs) -> OlmoEarthEncoderWrapper:
    return _create_olmoearth("v1.1", "nano", ModelID.OLMOEARTH_V1_1_NANO, model_bands, **kwargs)


@TERRATORCH_BACKBONE_REGISTRY.register
def olmoearth_v1_1_tiny(model_bands: list[str], **kwargs) -> OlmoEarthEncoderWrapper:
    return _create_olmoearth("v1.1", "tiny", ModelID.OLMOEARTH_V1_1_TINY, model_bands, **kwargs)


@TERRATORCH_BACKBONE_REGISTRY.register
def olmoearth_v1_1_base(model_bands: list[str], **kwargs) -> OlmoEarthEncoderWrapper:
    return _create_olmoearth("v1.1", "base", ModelID.OLMOEARTH_V1_1_BASE, model_bands, **kwargs)


@TERRATORCH_BACKBONE_REGISTRY.register
def olmoearth_v1_2_nano(model_bands: list[str], **kwargs) -> OlmoEarthEncoderWrapper:
    return _create_olmoearth("v1.2", "nano", ModelID.OLMOEARTH_V1_2_NANO, model_bands, **kwargs)


@TERRATORCH_BACKBONE_REGISTRY.register
def olmoearth_v1_2_tiny(model_bands: list[str], **kwargs) -> OlmoEarthEncoderWrapper:
    return _create_olmoearth("v1.2", "tiny", ModelID.OLMOEARTH_V1_2_TINY, model_bands, **kwargs)


@TERRATORCH_BACKBONE_REGISTRY.register
def olmoearth_v1_2_small(model_bands: list[str], **kwargs) -> OlmoEarthEncoderWrapper:
    return _create_olmoearth("v1.2", "small", ModelID.OLMOEARTH_V1_2_SMALL, model_bands, **kwargs)


@TERRATORCH_BACKBONE_REGISTRY.register
def olmoearth_v1_2_base(model_bands: list[str], **kwargs) -> OlmoEarthEncoderWrapper:
    return _create_olmoearth("v1.2", "base", ModelID.OLMOEARTH_V1_2_BASE, model_bands, **kwargs)
