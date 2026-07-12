# Copyright contributors to the Terratorch project
"""First-class terratorch backbones for Galileo (https://github.com/nasaharvest/galileo).

Galileo is a multimodal pretrained remote-sensing transformer. Its inputs are grouped by
band *group* (e.g. S2_RGB = B2,B3,B4), and masking happens at group granularity, so band
subsetting is handled natively by declaring only complete groups — the per-group patch
embeddings of undeclared groups are simply masked out. Because the patch-embedding weights
are per-group (not per arbitrary channel stack), select_patch_embed_weights does not apply.

Input metadata (all optional):
    - bands are declared at construction via ``model_bands`` (terratorch names such as
      BLUE/GREEN/RED/... or native Galileo names B2..B12, VV, VH, NDVI);
    - ``month`` (constructor, default June) or a per-batch ``months`` forward kwarg;
    - ``patch_size`` and ``input_resolution_m`` (constructor).

The vendored ``single_file_galileo.py`` has no PyPI package; checkpoints live on
HuggingFace (``nasaharvest/galileo``). ``pretrained=True`` builds the encoder from the
checkpoint's own config.json, so the offline default configs below only affect
``pretrained=False`` (random weight) builds.
"""

import logging
from pathlib import Path

import torch
from einops import rearrange
from torch import nn

from terratorch.models.backbones.galileo.single_file_galileo import (
    BASE_GSD,
    SPACE_BAND_GROUPS_IDX,
    SPACE_BANDS,
    SPACE_TIME_BANDS,
    SPACE_TIME_BANDS_GROUPS_IDX,
    STATIC_BAND_GROUPS_IDX,
    STATIC_BANDS,
    TIME_BAND_GROUPS_IDX,
    TIME_BANDS,
    Encoder,
)
from terratorch.registry import TERRATORCH_BACKBONE_REGISTRY

logger = logging.getLogger(__name__)

# terratorch band names -> native Galileo band names. Native names pass through.
GALILEO_BAND_ALIASES = {
    "BLUE": "B2",
    "GREEN": "B3",
    "RED": "B4",
    "RED_EDGE_1": "B5",
    "RED_EDGE_2": "B6",
    "RED_EDGE_3": "B7",
    "NIR_BROAD": "B8",
    "NIR_NARROW": "B8A",
    "SWIR_1": "B11",
    "SWIR_2": "B12",
}

# The nano config is verified against the upstream repo's committed checkpoint config
# (data/models/nano/config.json). tiny/base follow the model table of the Galileo paper
# (arXiv:2502.09356); pretrained builds never use these — they always instantiate from the
# downloaded checkpoint's config.json.
GALILEO_DEFAULT_CONFIGS = {
    "nano": {"embedding_size": 128, "depth": 4, "num_heads": 8, "mlp_ratio": 4},
    "tiny": {"embedding_size": 192, "depth": 12, "num_heads": 3, "mlp_ratio": 4},
    "base": {"embedding_size": 768, "depth": 12, "num_heads": 12, "mlp_ratio": 4},
}


def _bands_to_indices(model_bands: list[str]) -> list[int]:
    """Map declared band names to their positions in Galileo's SPACE_TIME_BANDS vector."""
    indices = []
    for band in model_bands:
        name = GALILEO_BAND_ALIASES.get(str(band), str(band))
        if name not in SPACE_TIME_BANDS:
            msg = (
                f"Unknown band {band!r} for galileo. Known bands: {SPACE_TIME_BANDS} "
                f"(or terratorch aliases {sorted(GALILEO_BAND_ALIASES)})."
            )
            raise ValueError(msg)
        index = SPACE_TIME_BANDS.index(name)
        if index in indices:
            msg = f"Band {band!r} declared more than once."
            raise ValueError(msg)
        indices.append(index)
    return indices


def _active_band_groups(band_indices: list[int]) -> list[int]:
    """Return the indices of the band groups covered by the declared bands.

    Galileo masks inputs at band-group granularity, so every declared band's group must be
    fully covered. Partial groups raise with the group's full member list.
    """
    declared = set(band_indices)
    active = []
    for group_index, (group_name, group_band_indices) in enumerate(SPACE_TIME_BANDS_GROUPS_IDX.items()):
        n_declared = len(declared & set(group_band_indices))
        if n_declared == len(group_band_indices):
            active.append(group_index)
        elif n_declared > 0:
            group_bands = [SPACE_TIME_BANDS[i] for i in group_band_indices]
            msg = (
                f"Galileo band group {group_name!r} is only partially declared. Groups are "
                f"masked as a whole, so declare all of {group_bands} or none of them."
            )
            raise ValueError(msg)
    if not active:
        msg = "No Galileo band groups are covered by the declared bands."
        raise ValueError(msg)
    return active


class GalileoEncoderWrapper(nn.Module):
    """Wraps a Galileo Encoder to the terratorch backbone contract.

    forward(x, **kwargs) takes ``x`` of shape [B, C, H, W] (or [B, C, T, H, W]) and
    returns a list of token tensors [B, N, D], one per entry of ``out_indices``. Tokens
    are the space-time tokens pooled over the (masked) band-group and time dimensions;
    there is no CLS token.
    """

    has_cls_token = False

    def __init__(
        self,
        encoder: Encoder,
        model_bands: list[str],
        out_indices: list[int] | None = None,
        patch_size: int = 4,
        month: int = 6,
        input_resolution_m: int = BASE_GSD,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.patch_size = patch_size
        self.month = month
        self.input_resolution_m = input_resolution_m

        self.band_indices = _bands_to_indices(model_bands)
        self.active_group_indices = _active_band_groups(self.band_indices)

        n_blocks = len(encoder.blocks)
        out_indices = out_indices if out_indices else [-1]
        self.out_indices = [i % n_blocks for i in out_indices]
        self.out_channels = [encoder.embedding_size] * len(self.out_indices)

    def forward(self, x: torch.Tensor, months: torch.Tensor | int | None = None, **kwargs) -> list[torch.Tensor]:
        if x.dim() == 4:
            x = x.unsqueeze(2)  # [B, C, H, W] -> [B, C, T=1, H, W]
        b, c, t, h, w = x.shape
        if c != len(self.band_indices):
            msg = f"Expected {len(self.band_indices)} input channels (declared bands), got {c}."
            raise ValueError(msg)
        if h != w:
            msg = f"Galileo requires square inputs, got {h}x{w}."
            raise ValueError(msg)
        if h % self.patch_size != 0:
            msg = f"Input size {h} must be divisible by patch_size {self.patch_size}."
            raise ValueError(msg)
        device, dtype = x.device, x.dtype

        s_t_x = torch.zeros(b, h, w, t, len(SPACE_TIME_BANDS), dtype=dtype, device=device)
        s_t_x[..., self.band_indices] = rearrange(x, "b c t h w -> b h w t c")
        s_t_m = torch.ones(b, h, w, t, len(SPACE_TIME_BANDS_GROUPS_IDX), dtype=dtype, device=device)
        s_t_m[..., self.active_group_indices] = 0

        sp_x = torch.zeros(b, h, w, len(SPACE_BANDS), dtype=dtype, device=device)
        sp_m = torch.ones(b, h, w, len(SPACE_BAND_GROUPS_IDX), dtype=dtype, device=device)
        t_x = torch.zeros(b, t, len(TIME_BANDS), dtype=dtype, device=device)
        t_m = torch.ones(b, t, len(TIME_BAND_GROUPS_IDX), dtype=dtype, device=device)
        st_x = torch.zeros(b, len(STATIC_BANDS), dtype=dtype, device=device)
        st_m = torch.ones(b, len(STATIC_BAND_GROUPS_IDX), dtype=dtype, device=device)

        if months is None:
            months = torch.full((b, t), self.month, dtype=torch.long, device=device)
        elif isinstance(months, int):
            months = torch.full((b, t), months, dtype=torch.long, device=device)
        elif months.dim() == 1:
            months = months.unsqueeze(0).expand(b, -1)

        outputs = self.encoder(
            s_t_x,
            sp_x,
            t_x,
            st_x,
            s_t_m,
            sp_m,
            t_m,
            st_m,
            months,
            patch_size=self.patch_size,
            input_resolution_m=self.input_resolution_m,
            return_hidden_states=True,
        )
        s_t_out = outputs[0]  # [B, H', W', T, groups, D], layer-normed
        hidden_states = outputs[-1]  # list (per block) of (s_t_x, sp_x, t_x, st_x), pre-norm

        last = len(self.encoder.blocks) - 1
        features = []
        for i in self.out_indices:
            feature = s_t_out if i == last else hidden_states[i][0]
            # masked groups carry no signal: pool only over the declared (visible) groups,
            # then over time, to get one token per spatial patch.
            feature = feature[..., self.active_group_indices, :].mean(dim=(3, 4))
            features.append(rearrange(feature, "b h w d -> b (h w) d"))
        return features

    def freeze(self):
        for param in self.encoder.parameters():
            param.requires_grad_(False)


def _load_galileo_checkpoint_folder(size: str, ckpt_data: str | None) -> Path:
    if ckpt_data is not None:
        return Path(ckpt_data)
    from huggingface_hub import snapshot_download

    local_dir = snapshot_download(repo_id="nasaharvest/galileo", allow_patterns=[f"models/{size}/*"])
    return Path(local_dir) / "models" / size


def _create_galileo(
    size: str,
    model_bands: list[str],
    pretrained: bool = False,  # noqa: FBT001, FBT002
    ckpt_data: str | None = None,
    out_indices: list[int] | None = None,
    patch_size: int = 4,
    month: int = 6,
    input_resolution_m: int = BASE_GSD,
    **encoder_kwargs,
) -> GalileoEncoderWrapper:
    if pretrained or ckpt_data is not None:
        folder = _load_galileo_checkpoint_folder(size, ckpt_data)
        # builds the encoder from the checkpoint's own config.json
        encoder = Encoder.load_from_folder(folder, device=torch.device("cpu"))
    else:
        config = {
            **GALILEO_DEFAULT_CONFIGS[size],
            "max_sequence_length": 24,
            "max_patch_size": 8,
            **encoder_kwargs,
        }
        encoder = Encoder(**config)
    return GalileoEncoderWrapper(
        encoder,
        model_bands,
        out_indices=out_indices,
        patch_size=patch_size,
        month=month,
        input_resolution_m=input_resolution_m,
    )


@TERRATORCH_BACKBONE_REGISTRY.register
def galileo_nano(model_bands: list[str], **kwargs) -> GalileoEncoderWrapper:
    return _create_galileo("nano", model_bands, **kwargs)


@TERRATORCH_BACKBONE_REGISTRY.register
def galileo_tiny(model_bands: list[str], **kwargs) -> GalileoEncoderWrapper:
    return _create_galileo("tiny", model_bands, **kwargs)


@TERRATORCH_BACKBONE_REGISTRY.register
def galileo_base(model_bands: list[str], **kwargs) -> GalileoEncoderWrapper:
    return _create_galileo("base", model_bands, **kwargs)
