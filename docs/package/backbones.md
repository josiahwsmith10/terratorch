# Backbones

## Built-in Backbones

:::terratorch.models.backbones.terramind.model.terramind_vit.TerraMindViT
    options:
        toc_label: "TerraMind"

:::terratorch.models.backbones.prithvi_mae.PrithviViT
    options:
        toc_label: "Prithvi"

:::terratorch.models.backbones.swin_encoder_decoder.MMSegSwinTransformer
    options:
        toc_label: "Swin"

:::terratorch.models.backbones.unet.UNet
    options:
        toc_label: "U-Net"

:::terratorch.models.backbones.mmearth_convnextv2.ConvNeXtV2
    options:
        toc_label: "MMEarth ConvNeXt"

:::terratorch.models.backbones.dofa_vit.DOFAEncoderWrapper
    options:
        toc_label: "DOFA"

:::terratorch.models.backbones.clay_v1.embedder
    options:
        toc_label: "Clay v1"

:::terratorch.models.backbones.galileo.galileo_encoder.GalileoEncoderWrapper
    options:
        toc_label: "Galileo"

:::terratorch.models.backbones.olmoearth_register.OlmoEarthEncoderWrapper
    options:
        toc_label: "OlmoEarth"

## GFM backbones as a fair-comparison harness

DOFA, Clay v1, Galileo, and OlmoEarth are registered as first-class backbones with one
uniform contract: `forward(x, **kwargs)` takes `[B, C, H, W]` (Galileo/OlmoEarth also
accept `[B, C, T, H, W]`) and returns a list of token tensors `[B, N, D]`, one per
`out_indices` entry, with `out_channels = [embed_dim] * len(out_indices)`. All
model-specific metadata is derived at construction from the declared bands, with optional
per-batch overrides via named forward kwargs. Because the same neck stack and decoder work
for all four, they can be compared fairly with an identical decode path:

```yaml
# segmentation (identical for all four; remove_cls_token: false for galileo/olmoearth)
necks:
  - name: SelectIndices
    indices: [2, 5, 8, 11]     # [0, 1, 2, 3] for 4-block models (galileo/olmoearth nano)
  - name: ReshapeTokensToImage
  - name: LearnedInterpolateToPyramidal
decoder: UperNetDecoder
```

| Backbone | Registered names | CLS token | Band subsetting | Metadata (construction) | Metadata (forward kwargs) |
|---|---|---|---|---|---|
| DOFA | `dofa_{small,base,large}_patch16_224` | yes | any bands (wavelength-conditioned patch embed) | `model_bands` → wavelengths | `wavelengths` |
| Clay v1 | `clay_v1_base` | yes | any bands (wave-conditioned patch embed) | `model_bands`/`bands` → waves, `gsd`, `img_size` | `time`, `latlon`, `waves`, `gsd` |
| Galileo | `galileo_{nano,tiny,base}` | no | complete band *groups* (S1, S2_RGB, S2_Red_Edge, S2_NIR_10m, S2_NIR_20m, S2_SWIR, NDVI) | `model_bands` → group masks, `patch_size`, `month`, `input_resolution_m` | `months` |
| OlmoEarth | `olmoearth_v1_{nano,tiny,base,large}`, `olmoearth_v1_1_{nano,tiny,base}`, `olmoearth_v1_2_{nano,tiny,small,base}` | no | complete band *sets* (S2 L2A: `[B02,B03,B04,B08]`, `[B05,B06,B07,B8A,B11,B12]`, `[B01,B09]`; S1: `[vv,vh]`), one modality per model | `model_bands` → set masks, `patch_size`, `timestamps` | `timestamps` |

Notes and known upstream gaps:

- `select_patch_embed_weights` does not apply to any of the four: DOFA and Clay generate
  patch-embed weights from wavelengths (band subsetting is native), while Galileo and
  OlmoEarth have per-group/per-set patch embeds and subset natively by masking whole
  groups/sets. Declaring a partial group/set raises with the full member list.
- Galileo has no PyPI package and no official intermediate-features API, so
  `single_file_galileo.py` is vendored with a single additive, default-off
  `return_hidden_states` parameter (marked `TERRATORCH PATCH`). Its NDVI band group is not
  derived automatically — pass a precomputed `NDVI` channel to use it.
- OlmoEarth requires the optional `olmoearth-pretrain-minimal` dependency
  (`pip install terratorch[olmoearth]`; Python ≥ 3.11, torch ≥ 2.7). Upstream has no
  intermediate-features API (`token_exit_cfg` is a per-modality early exit), so
  `out_indices` uses forward hooks on `encoder.blocks[i]`, which requires the default
  `use_flash_attn=False`. Mixing S1 and S2 bands in one model is not supported because the
  upstream modality concatenation order is not contractually stable. `olmoearth_v1_2_small`
  cannot be randomly initialized offline in `olmoearth-pretrain-minimal` 0.0.6 (upstream
  `KeyError: 'small'`); terratorch falls back to building it from the checkpoint's
  config.json (requires network).

## APIs for External Models

!!! tip
    You find a detailed overview of all models in the [TorchGeo documentation](https://torchgeo.readthedocs.io/en/latest/api/models.html). 

:::terratorch.models.backbones.torchgeo_vit
    options:
        toc_label: "TorchGeo ViT models"

:::terratorch.models.backbones.torchgeo_resnet
    options:
        toc_label: "TorchGeo ResNet models"

:::terratorch.models.backbones.torchgeo_swin_satlas
    options:
        toc_label: "TorchGeo Swin Satlas"

:::terratorch.models.backbones.heliofm_register.heliofm_backbone_surya
    options:
        toc_label: "Surya"

<!--
### Timm

You can use any model from `timm` as a backbone. 

!!! tip
    List all available models with `timm.list_models` or filter by name using wildcards:
    
    ```python
    import timm
    timm.list_models('vit*')
    ```

::: timm.list_models
    options:
        heading_level: 4
        show_source: false
-->
