# RFC 0001: A typed spectral band metadata schema for the TorchGeo ↔ TerraTorch boundary

- **Status:** Draft
- **Audience:** TorchGeo and TerraTorch maintainers (both under the `torchgeo` GitHub org)
- **Target home:** TorchGeo (data-adjacent), consumed by TerraTorch; prototyped here as a
  self-contained `terratorch/spectral/` package that can be lifted verbatim into
  `torchgeo/spectral/` (zero non-stdlib imports, enforced by test)
- **Prototype:** this repository, branch `claude/spectral-band-metadata-schema-p1z3uo`

## 1. Motivation

TorchGeo pretrained weights are `WeightsEnum` entries carrying two pieces of information that
TerraTorch needs when it ingests a TorchGeo backbone:

1. **Band identity** — `meta['bands']`, sensor-native codes such as `'B4'`, `'B8a'`, `'VV'`.
2. **Normalization** — a Kornia `transforms` pipeline bound to the weight, e.g.
   `K.Normalize(mean=0, std=10000)` for the SSL4EO-S12 Sentinel-2 weights, or
   `K.Normalize(mean=[-12.59, -20.26], std=[5.26, 5.91])` (dB-domain statistics) for the
   Sentinel-1 weights.

TerraTorch currently consumes (1) through a hardcoded translation dict and drops (2) entirely.
Both cause real, user-visible failures:

### 1.1 The `look_up_table` KeyError (terratorch issue [#928](https://github.com/IBM/terratorch/issues/928))

TerraTorch re-declares band identity with its own semantic enums (`OpticalBands`, `SARBands`,
`HLSBands` in `terratorch/datasets/utils.py`) and translates TorchGeo band codes via a hardcoded
18-key dict, copy-pasted byte-identically into three backbone files
(`torchgeo_resnet.py`, `torchgeo_vit.py`, `torchgeo_swin_satlas.py`):

```python
look_up_table = {"B01": "COASTAL_AEROSOL", "B02": "BLUE", ..., "VV": "VV", "R": "RED", ...}

def get_pretrained_bands(model_bands):
    model_bands = [look_up_table[x.split('.')[-1]] for x in model_bands]   # <- KeyError here
    return model_bands
```

TorchGeo's *model* metadata uses **un-padded** band codes (`'B1'…'B8a'…'B12'` — note lowercase
`'B8a'`), while TorchGeo's *dataset* classes use **padded** codes (`Sentinel2.all_bands =
('B01', …, 'B8A', …)`). The `look_up_table` only knows the padded form. So:

```python
# issue #928 reproduction
model = EncoderDecoderFactory().build_model(
    backbone="ssl4eos12_resnet50_sentinel2_rgb_moco",
    backbone_pretrained=True,
    backbone_model_bands=["R", "G", "B"], ...)
# KeyError raised from get_pretrained_bands(weights.meta["bands"])
# because ResNet50_Weights.SENTINEL2_RGB_MOCO.meta['bands'] == ['B4', 'B3', 'B2']
```

The 13-band SSL4EO variants only avoid this because their factory functions **overwrite the
shared enum member's meta in place** before loading:

```python
if weights is not None:
    weights.meta['bands'] = ['B01', 'B02', ..., 'B8A', ..., 'B12']   # global side effect!
```

— a workaround the RGB variants forgot, and one that mutates `ResNet50_Weights.*.meta` for the
whole process. The same latent break exists for every SSL4EO-Landsat weight
(`meta['bands'] = ['B1'..'B7']`, also absent from the table).

### 1.2 Weight-bound normalization is silently dropped

No TerraTorch code path reads `weights.transforms`. Users must hand-copy each weight's means/stds
into their datamodule config. Consequences:

- Swapping `backbone: ssl4eos12_resnet50_sentinel2_all_moco` (expects raw reflectance / 10000)
  for `backbone: ssl4eos12_resnet50_sentinel1_all_moco` (expects dB values normalized with
  (−12.59, −20.26)/(5.26, 5.91)) silently feeds wrongly-scaled inputs unless the user knows to
  dig the constants out of TorchGeo's source.
- There is no programmatic way to even *ask* a built TerraTorch backbone what normalization its
  weights were trained with.

### 1.3 Both conventions are optical-first; SAR is under-specified

A SAR band today is the bare string `"VV"`. Nothing distinguishes:

- **polarization** (VV / VH / HH / HV),
- **frequency band** (Sentinel-1 C-band vs ALOS L-band vs TerraSAR-X X-band — models pretrained
  on one do not transfer blindly to another),
- **representation** (linear amplitude vs intensity vs dB vs complex SLC). The SSL4EO-S12 S1
  statistics above are only meaningful for dB inputs; a complex-valued backbone would need
  entirely different handling.

### 1.4 DOFA has a parallel, buggy table

`terratorch/models/backbones/dofa_vit.py` keeps its own `waves_list` band→wavelength dict with
two live data bugs (`"THERMAL_INFRARED_12"` for `THERMAL_INFRARED_2`; key `"VV-VH"` that can
never match the `SARBands.VV_VH` enum value `"VV_VH"`) and the same unguarded-`KeyError` shape.
Meanwhile TorchGeo already ships an authoritative wavelength table
(`torchgeo.datasets.Sentinel2.wavelengths`) that nothing in this path uses.

## 2. Goals and non-goals

**Goals**

- One typed, serializable schema describing per-band identity + modality + (for SAR)
  polarization/frequency/representation, and per-weight normalization.
- Derivable from, and attachable to, an existing `WeightsEnum.meta` **without any TorchGeo
  change required today**, and with a one-line-per-weight data-only TorchGeo change later.
- A resolver that maps dataset-declared bands → model-expected bands returning **both** the
  channel index selection and the correct normalization — replacing the lookup dicts.
- Graceful fallback at every layer: absent schema ⇒ exact legacy behavior.
- Zero new hard dependencies. Schema modules are stdlib-only (`dataclasses`, `enum`, `json`).

**Non-goals**

- Replacing Kornia / TorchGeo `transforms` pipelines (the schema *describes* normalization; the
  geometric parts of a transform stay where they are).
- Temporal / acquisition metadata (orbit direction is included only as an optional SAR field
  because TerraTorch's `SARBands` already encodes ASC_/DSC_ variants).
- Changing `select_patch_embed_weights` (it keeps consuming band-name lists).
- Complex-valued model support itself — the schema can *describe* complex SAR data (§7) so the
  representation is no longer erased, but no complex backbone is added here.

## 3. Schema

All types live in one stdlib-only module (`spectral/schema.py` — prototyped as
`terratorch.spectral.schema`, proposed home `torchgeo.spectral.schema`).

```python
class Modality(str, Enum):
    OPTICAL = "optical"; SAR = "sar"; DEM = "dem"
    DERIVED = "derived"        # computed products: NDVI, ...
    MASK = "mask"              # categorical layers: LULC, ...
    UNKNOWN = "unknown"

class Polarization(str, Enum): VV = "VV"; VH = "VH"; HH = "HH"; HV = "HV"
class SARFrequencyBand(str, Enum): P, L, S, C, X, KU, K, KA
class Representation(str, Enum):
    AMPLITUDE = "amplitude"    # linear amplitude
    INTENSITY = "intensity"    # power / sigma0, linear
    DB = "db"                  # log-scaled (10*log10)
    COMPLEX = "complex"        # single-look complex (I/Q)

@dataclass(frozen=True)
class SARInfo:
    polarization: Polarization
    frequency_band: SARFrequencyBand
    representation: Representation
    center_frequency_ghz: float | None = None    # 5.405 for Sentinel-1
    orbit: str | None = None                     # "ascending" | "descending" for ASC_*/DSC_* bands

@dataclass(frozen=True)
class BandSpec:
    name: str                          # canonical semantic name; equals TerraTorch enum values
                                       # ("RED", "NIR_NARROW", "VV") for zero-cost interop
    modality: Modality
    sensor_code: str | None = None     # canonical sensor code, e.g. "B04" (S2), "B4" (Landsat TM)
    aliases: tuple[str, ...] = ()      # case-insensitive: ("B4", "R", "RED", ...)
    wavelength_um: float | None = None # center wavelength; None where not meaningful (SAR, DEM)
    bandwidth_um: float | None = None
    sar: SARInfo | None = None         # required iff modality == SAR (validated)

@dataclass(frozen=True)
class NormalizationSpec:
    """Semantics: out = (x - mean) / std per channel — exactly kornia.augmentation.Normalize.
    Scale-only normalization ("divide by 10000") encodes as mean=0.0, std=10000.0."""
    means: tuple[float, ...]
    stds: tuple[float, ...]
    note: str | None = None            # provenance, e.g. "torchgeo _ssl4eo_s12_transforms_s2_10k"

@dataclass(frozen=True)
class SpectralSpec:
    bands: tuple[BandSpec, ...]
    normalization: NormalizationSpec | None = None   # index-aligned with bands (validated)
    dynamic_bands: bool = False        # True for wavelength-conditioned models (DOFA): the model
                                       # accepts arbitrary bands described by wavelength
    source: str | None = None          # e.g. "torchgeo:ResNet50_Weights.SENTINEL2_ALL_MOCO"
    version: int = 1                   # schema version for forward compatibility
```

`SpectralSpec` serializes to/from JSON-compatible dicts (`to_dict`/`from_dict`/`to_json`/
`from_json`); `from_dict` ignores unknown keys so older consumers tolerate newer producers.

**Design decisions**

- **Normalization lives on the spec, not the band.** Band identity is sensor physics — reusable
  across every weight trained on that sensor. Normalization statistics are a property of one
  training run. Putting mean/std on `BandSpec` would force duplicating the Sentinel-2 catalog
  per weight; instead `NormalizationSpec` is index-aligned with `bands` and the resolver slices it.
- **`name` uses TerraTorch's existing semantic strings** (`OpticalBands`/`SARBands` enum values)
  so TerraTorch consumes specs with no translation layer, while `sensor_code`+`aliases` cover
  every sensor-native spelling TorchGeo emits (`B4`, `B04`, `B8a`, `B8A`, `R`, …).
- **One normalization semantic** — `(x − mean) / std` — matching `K.Normalize`, so the numbers
  in a spec are directly checkable against the weight's bound transform (we do exactly that in
  tests, §8).
- **`dynamic_bands`** models DOFA-style weights honestly: `DOFA_MAE.meta` has *no* `bands` and
  no fixed channel count; the weight accepts any bands the user can describe by wavelength.

## 4. Attaching a spec to a weight

`Weights` (torchvision) is a frozen dataclass `(url, transforms, meta: dict)` whose `meta` keys
are already heterogeneous across TorchGeo weights (some have `ssl_method`, DOFA has no
`bands`/`in_chans`). Adding one optional key is therefore fully backward-compatible:

```python
SENTINEL2_RGB_MOCO = Weights(
    url=..., transforms=_ssl4eo_s12_transforms_s2_10k,
    meta={
        ...,
        'bands': _sentinel2_rgb_bands,
        'spectral_spec': {                      # NEW, optional, plain JSON-able dict
            'version': 1,
            'bands': [
                {'name': 'RED',   'modality': 'optical', 'sensor_code': 'B04',
                 'wavelength_um': 0.665, 'aliases': ['B4', 'R']},
                {'name': 'GREEN', 'modality': 'optical', 'sensor_code': 'B03',
                 'wavelength_um': 0.560, 'aliases': ['B3', 'G']},
                {'name': 'BLUE',  'modality': 'optical', 'sensor_code': 'B02',
                 'wavelength_um': 0.492, 'aliases': ['B2', 'B']},
            ],
            'normalization': {'means': [0.0, 0.0, 0.0], 'stds': [10000.0, 10000.0, 10000.0]},
        },
    },
)
```

Consumers resolve a spec through a **fallback ladder** (`spec_for_weights(weights)`, never raises):

1. `weights is None` → `None`.
2. `weights.meta['spectral_spec']` present → `SpectralSpec.from_dict(...)`.
   *This is the future TorchGeo-native path.*
3. A consumer-side registry keyed by weight identity
   (`"ResNet50_Weights.SENTINEL2_RGB_MOCO"`, falling back to URL match).
   *This is the bridge that works today with unmodified TorchGeo.*
4. `weights.meta['bands']` present → identity-only spec via the band catalog (canonicalized
   names, wavelengths from the catalog, `normalization=None`).
5. Otherwise `None` → callers keep exact legacy behavior.

## 5. Strawman specs for three weights

### 5.1 `ResNet50_Weights.SENTINEL2_ALL_MOCO` (S2 optical ResNet)

```json
{
  "version": 1,
  "source": "torchgeo:ResNet50_Weights.SENTINEL2_ALL_MOCO",
  "bands": [
    {"name": "COASTAL_AEROSOL", "modality": "optical", "sensor_code": "B01", "wavelength_um": 0.4427, "aliases": ["B1"]},
    {"name": "BLUE",            "modality": "optical", "sensor_code": "B02", "wavelength_um": 0.4927, "aliases": ["B2", "B"]},
    {"name": "GREEN",           "modality": "optical", "sensor_code": "B03", "wavelength_um": 0.5598, "aliases": ["B3", "G"]},
    {"name": "RED",             "modality": "optical", "sensor_code": "B04", "wavelength_um": 0.6646, "aliases": ["B4", "R"]},
    {"name": "RED_EDGE_1",      "modality": "optical", "sensor_code": "B05", "wavelength_um": 0.7041, "aliases": ["B5"]},
    {"name": "RED_EDGE_2",      "modality": "optical", "sensor_code": "B06", "wavelength_um": 0.7405, "aliases": ["B6"]},
    {"name": "RED_EDGE_3",      "modality": "optical", "sensor_code": "B07", "wavelength_um": 0.7828, "aliases": ["B7"]},
    {"name": "NIR_BROAD",       "modality": "optical", "sensor_code": "B08", "wavelength_um": 0.8328, "aliases": ["B8"]},
    {"name": "NIR_NARROW",      "modality": "optical", "sensor_code": "B8A", "wavelength_um": 0.8647, "aliases": ["B8a"]},
    {"name": "WATER_VAPOR",     "modality": "optical", "sensor_code": "B09", "wavelength_um": 0.9451, "aliases": ["B9"]},
    {"name": "CIRRUS",          "modality": "optical", "sensor_code": "B10", "wavelength_um": 1.3735, "aliases": []},
    {"name": "SWIR_1",          "modality": "optical", "sensor_code": "B11", "wavelength_um": 1.6137, "aliases": []},
    {"name": "SWIR_2",          "modality": "optical", "sensor_code": "B12", "wavelength_um": 2.2024, "aliases": []}
  ],
  "normalization": {
    "means": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "stds": [10000.0, 10000.0, 10000.0, 10000.0, 10000.0, 10000.0, 10000.0, 10000.0, 10000.0, 10000.0, 10000.0, 10000.0, 10000.0],
    "note": "torchgeo _ssl4eo_s12_transforms_s2_10k: K.Normalize(mean=0, std=10000)"
  }
}
```

Band order matches `meta['bands']` (`B1..B8, B8a, B9..B12` — B8A at index 8), i.e. exactly the
order the checkpoint's `conv1.weight` channels were trained in. Wavelengths come from TorchGeo's
own `Sentinel2.wavelengths` table.

### 5.2 `ResNet50_Weights.SENTINEL1_ALL_MOCO` (S1 SAR ResNet)

```json
{
  "version": 1,
  "source": "torchgeo:ResNet50_Weights.SENTINEL1_ALL_MOCO",
  "bands": [
    {"name": "VV", "modality": "sar", "sensor_code": "VV",
     "sar": {"polarization": "VV", "frequency_band": "C", "representation": "db", "center_frequency_ghz": 5.405}},
    {"name": "VH", "modality": "sar", "sensor_code": "VH",
     "sar": {"polarization": "VH", "frequency_band": "C", "representation": "db", "center_frequency_ghz": 5.405}}
  ],
  "normalization": {
    "means": [-12.59, -20.26],
    "stds": [5.26, 5.91],
    "note": "torchgeo _ssl4eo_s12_transforms_s1; statistics are in dB domain"
  }
}
```

What the plain string `"VV"` never said, now typed: C-band, dB representation (the −12.59/−20.26
statistics are only meaningful in log domain — flagged for TorchGeo maintainer confirmation),
5.405 GHz center frequency. A complex-SLC variant of the same sensor differs only in
`"representation": "complex"` (§7) instead of being indistinguishable.

### 5.3 `DOFABase16_Weights.DOFA_MAE` (wavelength-conditioned)

```json
{
  "version": 1,
  "source": "torchgeo:DOFABase16_Weights.DOFA_MAE",
  "bands": [],
  "dynamic_bands": true,
  "normalization": null
}
```

Faithful to TorchGeo: `DOFA_MAE.meta` has no `bands`/`in_chans` and its bound transform is a
`CenterCrop` only. The spec says so *explicitly* instead of by omission. At call time the
consumer resolves the *user's* bands through the shared band catalog and feeds
`BandSpec.wavelength_um` (optical) or the SAR center frequency (DOFA's existing convention:
5.405 for S1 bands) to the model — replacing TerraTorch's private `waves_list`.

## 6. The band catalog and resolver

### 6.1 Catalog (`spectral/catalog.py`)

The catalog holds canonical `BandSpec` definitions per sensor — Sentinel-2 (13 bands, above),
Sentinel-1 C-band (VV/VH/HH/HV + TerraTorch's ASC_/DSC_ orbit composites + VV_VH ratio), generic
RGB aliases, Landsat TM/ETM/OLI-TIRS (fixing the latent SSL4EO-L KeyError), and HLS thermal
bands. One function replaces every lookup dict:

```python
canonicalize_band(name: str | Enum, *, sensor: str | None = None) -> BandSpec | None
```

- accepts enums (uses `.value`) and dotted prefixes (`"SENTINEL2.B02"` — the convention
  `get_pretrained_bands` already strips) as sensor hints;
- case-insensitive alias matching (`B4`/`b04`/`R`/`RED` → the S2 RED spec);
- ambiguity policy: un-padded `B1`–`B8` collide between Sentinel-2 and Landsat; with a sensor
  hint (which the weight registry always supplies) resolution is exact; unhinted resolution
  defaults to Sentinel-2 with a warning (open question §10).

### 6.2 Resolver (`spectral/resolve.py`)

```python
resolve_bands(requested_bands: Sequence[str | Enum], spec: SpectralSpec, *, strict: bool = False)
    -> BandResolution

@dataclass(frozen=True)
class BandResolution:
    requested: tuple[str, ...]            # canonicalized requested names, input order
    indices: tuple[int | None, ...]       # per requested band: index into spec.bands, None if absent
    matched: tuple[str, ...]
    missing: tuple[str, ...]              # -> xavier-init downstream, as today
    means: tuple[float, ...] | None       # per requested band; 0.0/1.0 for missing; None if spec
    stds: tuple[float, ...] | None        #   carries no normalization
    pretrained_band_names: tuple[str, ...]  # all spec bands, spec order — drop-in
                                            # `pretrained_bands` for select_patch_embed_weights
```

One call answers both questions the boundary has: *which checkpoint channels correspond to my
bands* (`indices`/`pretrained_band_names`) and *how must those channels be normalized*
(`means`/`stds`). Worked example for issue #928:

```python
spec = spec_for_weights(ResNet50_Weights.SENTINEL2_RGB_MOCO)
res = resolve_bands(["R", "G", "B"], spec)
# res.indices == (0, 1, 2)          R->B4 (checkpoint channel 0), G->B3, B->B2
# res.means   == (0.0, 0.0, 0.0)
# res.stds    == (10000.0, 10000.0, 10000.0)
```

### 6.3 Normalization provenance: registry constants, not transform introspection

`weights.transforms` is an opaque callable whose framework has already changed across TorchGeo
versions (Kornia `AugmentationSequential` in 0.7.x, torchvision `nn.Sequential` in 0.8.x).
Walking its children to extract mean/std would be brittle and impossible for weights whose
normalization is expressed differently (SeCo's three chained Normalizes, Satlas' clamp). The
prototype instead records explicit constants in the per-weight registry — ~5 auditable lines per
weight — and pins them with a **black-box numerical equivalence test** against the real bound
transform: constant images are invariant under any resize/crop geometry, so feeding two constant
tensors through the pipeline recovers its effective per-channel affine exactly, framework- and
version-independently (§11). The registry cannot silently drift from TorchGeo. When TorchGeo
ships `meta['spectral_spec']` (phase 2, §9), the constants live next to the transform definition
and drift becomes structurally impossible.

## 7. SAR extension: representation-aware specs round-trip today

The schema can describe weights no current backbone can consume, so the information is preserved
rather than erased at ingest time. Demonstrated in the prototype tests:

```python
slc_spec = SpectralSpec(
    bands=(
        BandSpec(name="VV", modality=Modality.SAR, sensor_code="VV",
                 sar=SARInfo(Polarization.VV, SARFrequencyBand.C,
                             Representation.COMPLEX, center_frequency_ghz=5.405)),
        BandSpec(name="VH", modality=Modality.SAR, sensor_code="VH",
                 sar=SARInfo(Polarization.VH, SARFrequencyBand.C,
                             Representation.COMPLEX, center_frequency_ghz=5.405)),
    ),
    source="demo:ResNet50_Weights.SENTINEL1_SLC_DEMO",
)
register_weight_spec("ResNet50_Weights.SENTINEL1_SLC_DEMO", slc_spec)

assert SpectralSpec.from_json(slc_spec.to_json()) == slc_spec        # JSON round-trip
# and via the exact channel a future torchgeo WeightsEnum would use:
fake = Mock(meta={"spectral_spec": slc_spec.to_dict()})
assert spec_for_weights(fake) == slc_spec
assert spec_for_weights(fake).bands[0].sar.representation is Representation.COMPLEX
```

A future complex-valued backbone reads `representation` and refuses (or adapts) when handed dB
statistics; an amplitude model composed with a dB dataset can detect the mismatch instead of
training on garbage. Frequency band makes C-vs-L-band transfer explicit.

## 8. Consumption in TerraTorch (proof of concept)

Prototyped in this branch, all backward-compatible:

1. **ResNet path (#928 fix).** `load_resnet_weights` resolves `pretrained_bands` through
   `spec_for_weights` first, then legacy `get_pretrained_bands` (now catalog-backed and
   warn-don't-crash on unknown names), then `[]`. The user's `model_bands` are canonicalized
   through the same catalog — without this, spellings like `"R"` never string-match the
   canonical pretrained names and every conv1 column is silently xavier-reinitialized, so the
   KeyError fix alone would only be cosmetic. The `look_up_table` dict and the five
   enum-meta-mutating `weights.meta['bands'] = [...]` overwrites are deleted.
   `["R","G","B"]` against `SENTINEL2_RGB_MOCO` now builds, with R mapped to the B4 checkpoint
   channel (regression-tested at the conv1-weight level, before/after in §11).
2. **Normalization recovered, opt-in.** `ResNetEncoderWrapper` exposes `spectral_spec` and
   `input_normalization` (a `BandResolution`) and accepts `apply_input_normalization=True`
   (YAML: `backbone_apply_input_normalization: true`; the factory's `backbone_` prefix
   stripping already forwards it) to prepend `(x − mean)/std` in `forward`. Default `False`:
   behavior is bit-identical to today. Equivalence with TorchGeo's bound transform is asserted
   numerically in tests (`w.transforms(x)` vs Kornia geometry + spec constants).
3. **DOFA.** `get_wavelengths` resolves bands through the catalog (`BandSpec.wavelength_um` /
   SAR center frequency), fixing the two `waves_list` bugs (old spellings kept as aliases) and
   giving unknown bands a descriptive error; `get_wavelenghts` remains as a deprecated alias.
4. **ViT/Swin.** Their copy-pasted `get_pretrained_bands` delegate to the same catalog, and
   both load paths get the same `spec_for_weights` ladder — the Swin call site was previously
   fully unguarded (`weights.meta["bands"]` with no fallback at all).

## 9. Upstreaming plan (two maintainer groups, three phases)

The point of the design is that each phase is independently shippable and no coordinated release
is ever required:

- **Phase 1 (this prototype, TerraTorch only).** Self-contained `terratorch/spectral/` package;
  registry entries on the TerraTorch side bridge unmodified TorchGeo weights. TerraTorch stops
  crashing on TorchGeo's band spellings and stops discarding normalization. No TorchGeo change.
- **Phase 2 (TorchGeo, data-only PRs).** Add `'spectral_spec': {...}` dicts to `WeightsEnum.meta`
  entries — pure data, one weight per reviewable chunk, next to the transform definitions they
  describe. Optionally adopt the schema module itself as `torchgeo.spectral` (it is stdlib-only
  and lifts verbatim). Fallback-ladder step 2 means every TerraTorch already in the wild picks
  these up automatically.
- **Phase 3 (TerraTorch cleanup).** Delete TerraTorch-side registry entries as TorchGeo ships
  specs; eventually import the schema from `torchgeo.spectral` and drop the vendored copy.

If TorchGeo maintainers prefer not to host the module, phases 2–3 degrade gracefully: the meta
dict convention alone (a documented key + JSON shape) is enough for interop, and the module can
live in a thin shared package or stay vendored in TerraTorch.

## 10. Alternatives considered / open questions

**Rejected alternatives**

- *Introspecting `weights.transforms`* — brittle across Kornia versions; cannot express SeCo's
  chained normalizes or Satlas' clamp; nothing to introspect for weights normalized outside the
  transform. Registry constants + equivalence tests are auditable and drift-proof.
- *Extending TorchGeo dataset classes (`all_bands`, `wavelengths`)* — identity only, wrong
  granularity: normalization is per-weight, not per-sensor; and model meta ≠ dataset naming
  today (the very bug being fixed).
- *Per-band mean/std on `BandSpec`* — couples reusable sensor physics to run-specific
  statistics; forces catalog duplication per weight.
- *Fixing #928 with more `weights.meta['bands']` overwrites* — perpetuates global enum mutation
  and the padded/un-padded split; fixes one weight at a time forever.

**Open questions**

1. Unhinted un-padded codes (`B1`): default to Sentinel-2 with a warning (current prototype), or
   require a sensor hint and error? (S2-default matches legacy behavior.)
2. Is `representation: db` correct for the SSL4EO-S12 S1 weights? (Inferred from the clearly
   log-domain statistics; needs TorchGeo maintainer confirmation.)
3. Should `meta['spectral_spec']` be the dict (proposed: yes, keeps `meta` JSON-able) or a
   `SpectralSpec` instance?
4. Naming: `spectral_spec` vs `band_spec` vs `sensor_spec` for the meta key.
5. Where should per-weight *geometric* expectations (input size 224/256) live? Out of scope
   here, but the same meta-key pattern would work.

## 11. Before / after (issue #928) — measured on this branch

Regression tests: `TestIssue928Regression` and `TestInputNormalization` in
`tests/test_torchgeo_resnet.py`; equivalence tests in `tests/test_spectral_torchgeo_specs.py`.
Environment: torchgeo 0.8.1, torch 2.13 CPU.

**Before** (base commit `8497c0f`) — building the #928 configuration:

```python
ssl4eos12_resnet50_sentinel2_rgb_moco(model_bands=["R", "G", "B"], pretrained=True, ...)
```

```
  File ".../terratorch/models/backbones/torchgeo_resnet.py", line 796, in load_resnet_weights
    pretrained_bands = get_pretrained_bands(weights.meta["bands"]) if "bands" in weights.meta else []
  File ".../terratorch/models/backbones/torchgeo_resnet.py", line 98, in get_pretrained_bands
    model_bands = [look_up_table[x.split('.')[-1]] for x in model_bands]
KeyError: 'B4'
```

(the un-padded names in `SENTINEL2_RGB_MOCO.meta['bands']` miss the padded-only
`look_up_table`; issue #928 reports the same failure class as `KeyError: 'B1'`.
Through `EncoderDecoderFactory` the KeyError is additionally masked as
"The model ssl4eos12_resnet50_sentinel2_rgb_moco could not be instantiated from any source.")

**After** — the same call, through the full `EncoderDecoderFactory` path:

```
--- build succeeded ---
encoder type: ResNetEncoderWrapper
spectral_spec bands: ('RED', 'GREEN', 'BLUE')
input_normalization means: (0.0, 0.0, 0.0) stds: (10000.0, 10000.0, 10000.0)
```

with the checkpoint's `conv1` channels actually selected (R→B4 checkpoint channel 0, G→B3,
B→B2 — asserted at the weight-tensor level, including a reordered-bands variant; previously the
user's `"R"/"G"/"B"` spellings never string-matched the canonical pretrained names, so even
without the KeyError every conv1 column would have been silently xavier-reinitialized).

**Normalization preserved** — effective per-channel affine constants extracted from the real
`weights.transforms` pipelines (black-box, via constant images) vs the registry:

```
SENTINEL2_ALL_MOCO:  transforms: mean=[0.0]*13,          std=[10000.0]*13
                     registry:   mean=[0.0]*13,          std=[10000.0]*13
SENTINEL2_RGB_MOCO:  transforms: mean=[0.0]*3,           std=[10000.0]*3
                     registry:   mean=[0.0]*3,           std=[10000.0]*3
SENTINEL1_ALL_MOCO:  transforms: mean=[-12.59, -20.26],  std=[5.26, 5.91]
                     registry:   mean=[-12.59, -20.26],  std=[5.26, 5.91]
```

Exact match on all three target weights (also DINO/DECUR variants; pinned in CI by
`TestNormalizationEquivalenceVsTorchgeo`, which fails if upstream constants ever drift).
Opting in via `backbone_apply_input_normalization: true` applies `(x − mean)/std` in `forward`,
asserted equal to manually normalizing then running the plain model.
