# Volume 05 — calibration and quantization, end to end

This volume follows quantization information from the first calibration
sample to the bytes the backend writes into the BModel command stream.  It
documents every pass that reads, creates, rewrites or consumes quantization
state, and it explains why each stage exists and what the next stage expects.

The study was done against the source tree at commit `0f2f7d6bb`.  Line
numbers drift; function names are the stable anchor.

> **Boundary.** TPU-MLIR builds the quantized graph, the integer weights and
> the per-op parameter structs.  The final TIU/GDMA instruction words are
> emitted by the closed-source chip backend (`third_party/nntoolchain/lib/
> libbackend_<chip>.so`).  Everything up to the backend call is read from
> source here.  What happens inside the backend is described only as far as
> the parameter structs and the bit-exact CPU interpreter fix it.

---

## 1. The whole pipeline on one page

Quantization information changes form four times.  Knowing which form you are
looking at is the key to reading every file below.

```text
 FORM 1: text table          FORM 2: MLIR calibrated type     FORM 3: MLIR uniform type + int payload     FORM 4: bytes + param struct
 ───────────────────         ────────────────────────────     ───────────────────────────────────────     ────────────────────────────
 run_calibration.py          --import-calibration-table       --convert-top-to-tpu                        --weight-reorder / codegen
 "conv1 4.12 -3.2 4.12"  ──► tensor<..x!quant.calibrated   ──► tensor<..x!quant.uniform<i8:f32,       ──► coeff = [quant|bias|filter]
  (threshold, min, max)       <f32<-4.12:4.12>>>                0.0324>>  + filter_i8 / bias_int32         conv_global_spec_t{...}
                                                                + multiplier[], rshift[] attrs             → libbackend_*.so → TIU/GDMA cmds
```

The complete command chain for a normal INT8 deployment:

```text
model_transform.py ─► <m>.mlir (TOP_F32) + <m>_top_origin_weight.npz
        │
run_calibration.py <m>.mlir --dataset ... -o <m>_cali_table        (+ optional qtable)
        │      ├── ShapeOps / MatchPattern       -> default float layers (shape_pattern_qtable)
        │      ├── [SmoothQuant | SoftmaxCorrection | WeightEqualization]  (rewrite Top MLIR / weight NPZ in place)
        │      ├── ActivationCalibrator          -> per-tensor statistics -> thresholds
        │      ├── SimpleTuner                   -> refine thresholds (KL/ACIQ only)
        │      ├── [BiasCorrection]              -> rewrite conv bias in weight NPZ
        │      └── [search_* ]                   -> mixed-precision qtable via simulated lowering
        │
model_deploy.py --quantize INT8 --calibration_table ... [--quantize_table ...]
        │
        ├── tpuc-opt <m>.mlir                                         (mlir_shell.lowering_options)
        │     --processor-assign="chip=.. mode=INT8 .. high_precision=.."   (Top) module mode/chip
        │     --import-calibration-table="file=.. asymmetric=.."            (Top) FORM 1 -> FORM 2, TOP_CALIBRATED
        │     --processor-top-optimize                                       (Top) chip-specific graph prep
        │     --convert-top-to-tpu="qtable=.. weightFileName=.. asymmetric=.. q_group_size=.. ..."
        │                                                                    FORM 2 -> FORM 3, TPU_LOWERED
        │     --canonicalize  --weight-fold
        │   -> <prefix>_tpu.mlir + <prefix>_tpu_weights.npz
        │   -> validate: interpreter(tpu.mlir) vs top fp32 outputs, using --tolerance
        │
        └── tpuc-opt <prefix>_tpu.mlir                                  (mlir_shell.mlir_to_model)
              --strip-io-quant=...                                          boundary casts
              --processor-tpu-optimize                                      requant fusion, splits
              --dev-parallel --weight-reorder                               FORM 3 -> FORM 4 (merged coeffs)
              --subnet-divide --op-reorder --topo-sort --layer-group ...
              --core-parallel --after-layergroup-weight-reorder --address-assign
              --codegen                                                     spec structs -> backend -> cmds
            -> <prefix>.bmodel, <prefix>_final.mlir
            -> validate: bmodel outputs vs interpreter(tpu.mlir), using --correctness
```

---

## 2. Vocabulary and invariants

### 2.1 Numbers

| Term | Meaning | Where computed |
|---|---|---|
| threshold `th` | symmetric clipping magnitude for an activation: values outside `[-th, th]` saturate | calibration |
| min / max | observed (or percentile-clipped) range of an activation | calibration |
| scale `s` | real value of one integer step: `real = s * (q - zp)` | `module::getScaleAndZeroPoint` |
| zero point `zp` | integer that represents real 0; 0 in symmetric mode | same |
| multiplier `m`, rshift `r` | fixed-point form of a real rescale factor: `x * f ≈ (x * m) >> r` | `get_scale_and_shift*`, `QuantizeMultiplier` |
| requant | converting an int32 accumulator (or int8 of another scale) into the output's int8 scale | lowering + `tpu.RequantInt*` |

### 2.2 Scale and zero-point formulas (`lib/Support/Module.cpp`)

`getScale(threshold, sign, bitwidth)`:

| bits | signed | unsigned |
|---|---|---|
| 8 | `th / 127` | `th / 255` |
| 4 | `th / 7` | `th / 15` |
| 16 | `th / 32767` | `th / 65535` |

`getScaleAndZeroPoint(Value v, ...)` on a calibrated tensor:

```text
sign = (cali.min < 0)
symmetric : zp = 0; scale = getScale(max(|min|,|max|), sign, bits)
            (Compare/CompareConst outputs force scale = 1.0)
asymmetric: getScaleAndZeroPoint(min, max): scale = (max-min)/(qmax-qmin),
            zp = round(qmin - min/scale), then nudged into [qmin,qmax]
```

There is one exception: if the producer is an INT4 Conv/MatMul that carries
`out_int8_scale`/`out_int8_zp`, those attributes win.

Consequence worth remembering: a tensor whose calibrated min is `>= 0` (for
example after ReLU) becomes **uint8 with scale `th/255`**, not int8.  It gains
one bit of resolution for free.  This is why so many lowerings branch on
`isSign`.

### 2.3 MLIR types that carry the state

| Type | Introduced by | Meaning |
|---|---|---|
| `f32` | frontend | no quantization info |
| `!quant.calibrated<f32<min:max>>` | `--import-calibration-table` | "this tensor will be quantized with this range" |
| `!quant.uniform<i8:f32, scale:zp>` / `<u8:...>` / `<i4...>` / `<i16...>` | `--convert-top-to-tpu` | real integer storage with scale/zp |
| `si32` / `i32` | lowering | raw accumulator (no scale in the type) |
| `f16` / `bf16` / `f8E4M3FN` / `f8E5M2` | lowering | float modes; the calibrated wrapper may survive with an f16 storage type |

### 2.4 Module state machine (`include/tpu_mlir/Support/Module.td`)

```text
TOP_F32 ──import-calibration-table──► TOP_CALIBRATED ──convert-top-to-tpu──► TPU_LOWERED
TOP_QUANTIZED (TFLite int8 / TpuLang "quantized") ───convert-top-to-tpu──────► TPU_LOWERED
                                    ──weight-reorder──► TPU_REORDERED ──subnet-divide──► TPU_DIVIDED ──address-assign──► TPU_ADDRESSED
```

`ImportCalibrationTablePass` refuses to run unless the state is `TOP_F32`.
`ConvertTopToTpu` treats `TOP_QUANTIZED` specially: it forces asymmetric and
uses each op's `LoweringQuantized`, because scales already exist in the types.

### 2.5 Quantize modes (`Module_Mode` in `Module.td`)

| Family | Modes | Weights | Activations |
|---|---|---|---|
| Integer static | `INT8`, `UINT8`, `INT4`, `W4INT8` | int8/int4 | int8/int4, calibrated |
| Float | `F32`, `F16`, `BF16` | same float | same float |
| FP8 static | `F8E4M3`, `F8E5M2` | fp8 + per-channel scale | fp8, calibrated (max) |
| Weight-only | `W8F16`, `W8BF16`, `W4F16`, `W4BF16` | int8/int4 per-group + scale/zp | f16/bf16 |
| Dynamic | `INT8F16DYN`, `INT4F16DYN`, `F8E4M3F16DYN`, `F4F16DYN`, `MXF4F16DYN` and BF16 twins | per-group | quantized at run time; only MatMul is quantized, everything else F16/BF16 |

### 2.6 Requant and round modes (`TpuOps.td`)

| `RequantMode` | Integer semantics |
|---|---|
| `MultiplierShift` | `y = round((x * m) >> r)` — the default on BM1684X/BM1688/BM1690 |
| `OnlyShift` | `y = round(x >> r)` — BM1684 style |
| `TFLite` / `QDM` | `y = round(round_half_up((x * m) >> 31) >> shift)` (gemmlowp style) — CV18xx, TFLite |
| `TFLite_LShift` | as TFLite with a left shift first |
| `OnlyScale` | float scale, used for FP8 and BM1684X2 per-tensor MatMul |

`RoundMode`: `HalfAwayFromZero`, `HalfUp`, `HalfDown`, `HalfToEven`,
`HalfToOdd`, `TowardsZero`, `Down`, `Up`.  The reference implementation is
`RightShiftRound` in `lib/Support/MathUtils.cpp`.

---

## 3. Stage 1 — calibration (`python/tools/run_calibration.py`)

### 3.1 Order of operations

`run_calibration.py` is a driver.  Its order matters because several steps
mutate the Top model before statistics are collected:

1. Mode fix-ups: `wi4ai4_wi8ai8`/`wi4ai8_wi8ai8` force an INT4-capable chip
   (`bm1688` if the chip is not in `{bm1688, cv186x, bm1690}`); `wf4*` modes
   force `sg2262`.
2. `DataSelector` chooses samples (`--dataset` + `--input_num`, or
   `--data_list`); an optional `--tune_list` provides tuning samples.
3. `ShapeOps.run()` (`shape_ops.py`) finds shape-computation subgraphs
   (`Shape`, `Size`, `TopK` 2nd output, `Arg` and everything fed only by them)
   that must stay float.
4. `MatchPattern.run()` (`transformer_pattern.py`) recognizes transformer
   blocks and model families and returns float layers, FP32 layers and ops that
   must be symmetric.
5. `gen_shape_pattern_qtable` writes those as a first quantize table
   (`shape_pattern_qtable` by default).  `--pre_qtable` replaces it.
6. Then either a **search** path (section 3.8) or the **simple** path:
   SmoothQuant (`--sq`) → softmax correction (`--smc`) → weight equalization
   (`--we`) → `ActivationCalibrator.run()` → bias correction (`--bc`), which
   re-calibrates afterwards.

### 3.2 Collecting statistics (`kld_calibrator.py: ActivationCalibrator`)

- The Top MLIR is loaded into `pymlir.module()` (or `pymlir.cuda()`), the
  same interpreter used for validation.  `pymlir.set_mem_mode("force_value_mem")`
  keeps every intermediate tensor available.
- Inputs are preprocessed with the preprocessing config stored on each
  `top.Input` (`utils/preprocess.py`).  Image batches are padded up to the
  model batch size by repeating the last sample.
- For each sample: `invoke()`, then `process_statistic` runs **per op in a
  thread pool** (`POOL_THREADS`).  Only tensors returned by
  `get_no_fused_tensors` are tracked; fused-op aliases are skipped.
- Per tensor it accumulates:
  - running `min_value`, `max_value`, and `abs_max`;
  - a 2048-bin histogram of `|x|` (zeros removed) — `cali_math.histogram`.
    When a later sample exceeds the current range, `combine_histogram` **extends
    the bin array at the same bin width** rather than rebinning, so earlier
    samples are not smeared;
  - per-sample MSE and ACIQ thresholds when those methods are active
    (averaged later);
  - sorted top-k tails for percentile methods (`sort_distr_percentile`).

Percentile is not one fixed value: op `i` uses
`99.99 + i * (99.999999 - 99.99) / num_ops`, so deeper layers clip less.

### 3.3 Threshold methods

| `--cali_method` | Threshold | Source |
|---|---|---|
| `kl` (default) | argmin KL divergence between the clipped 2048-bin distribution P and its 128-bin (int8) / 8-bin (int4) re-quantized Q | `bindings/kld/calibration_math.cpp: kl_diversity_hist` (multithreaded C, loaded via ctypes) |
| `mse` | fixed-point iteration `s ← Σ|x|>s / (|x≤s| / (3·4^b·u) + |x>s|)` (u=4 for non-negative tensors) | `cali_math.threshold_mse` |
| `max` | `abs_max` | — |
| `percentile9999` | interpolated percentile of `|x|`; min/max become the percentile min/max | `process_compute_threshold` |
| `aciq_gauss` | `α·σ`, σ estimated from range and N; α = 3.924 (8-bit), 2.559 (4-bit) | `cali_math.threshold_aciq_gauss` |
| `aciq_laplace` | `β·b`, b = mean absolute deviation; β = 9.897 / 5.029 | `cali_math.threshold_aciq_laplace` |

Special rules: every threshold is clamped to `abs_max` (`calibration_result.
canonicalize`); an all-zero tensor gets `1e-5` and a warning; for MSE/ACIQ,
model inputs use `abs_max`, and the last five tensors use the percentile value
when it differs from abs-max (the loop stops at the first Conv/MatMul).
`--part_asymmetric` marks SiLU/GELU outputs that feed only Conv/MatMul as
asymmetric, uses percentile min/max for them, and emits a `#asym_op` section.

KL histogram bins are padded to a multiple of `dst_bins` and halved (width
doubled) while they exceed `64*2048` bins, to bound the C helper's thread count.

### 3.4 Threshold tuning (`simple_tunner.py: SimpleTuner`)

Tuning runs only when `tune_num > 0` **and** the method is KL or ACIQ (it is
skipped for `max`, `mse`, `percentile9999`, int4, and an input table).
Because `--tune_num` defaults to 5, a default KL run is tuned.

For each op in topological order, and for each of its producer tensors:

1. A second interpreter instance gets `fake_quant_weight()` — weights are
   fake-quantized in place.
2. Candidate thresholds are swept linearly from the KL threshold to `abs_max`
   (`tune_steps = 21`; `find_lower_th` also searches below).
3. The input is fake-quantized with `import_quant_bias` (`round(x·127/th)`,
   clamped to [-128, 127], then divided back), the consumer is run with
   `invoke_at`, and the distance `‖y_fp32 − y_q‖₂ / ‖y_fp32‖₁` is averaged over
   tuning samples.
4. The best threshold is kept only if it is **larger** than the one already in
   the table (a producer with several consumers keeps the largest).

Tuning therefore optimizes a tensor's threshold for the op that consumes it,
using real fake-quantized weights.

### 3.5 The calibration table format

`dump_cali_table` writes (and `CalibrationTable.parse` reads):

```text
# mlir version: ... cali version 1.0
# quantization threshold calculation method: kl
# histogram number: 2048
# sample number: 100
###
# tune number: 5
###
# op_name    threshold    min    max
input        2.6400000 -2.1179039 2.6400000
conv1_Conv   9.8123457 -11.2000000 14.7000000
...
#int4_th                       <- optional: int4 activation thresholds (min = -th*8/7)
...
#asym_op                       <- optional: op names quantized asymmetrically
...
# symmetric_op                 <- optional, informational
```

Other sections understood only by the C++ importer: `#weight_scale` /
`#weight_scale_fp8` (`name N s0 s1 ...`, produced by QAT tooling),
`#tpu-mlir-fp8` / `mqbench-fp8` (FP8 thresholds), `#int4_op` (explicit INT4 op
list).  FP8 calibration (`debug_cmd fp8`) forces the `max` method and writes the
`#tpu-mlir-fp8` header **before** the data lines, so the whole body is parsed as
FP8 thresholds.

Two subtleties that explain many "why is my INT8 bad" questions:

- **Symmetric mode uses only the threshold**; the min column only decides the
  sign.  **Asymmetric mode ignores the threshold** and uses raw min/max (except
  for `percentile9999`, which writes clipped min/max).  KL clipping does not
  apply in asymmetric mode.  `mlir_shell.py` still carries the comment
  "always using symmetric, as asymmetric not good".
- The table key is the **tensor name** (output location name), not the op
  type.  Renaming a tensor in the frontend silently drops its threshold; the
  importer then leaves the tensor un-calibrated (min = max = 0 is skipped), and
  lowering will treat it as a float producer.

### 3.6 Graph-modifying pre-steps

These are calibration-time **model edits**, not just statistics.  They
overwrite the Top weight NPZ (`module.weight_file`), and some insert ops into
the Top MLIR.  Re-running `model_transform.py` undoes them.

| Flag | File | What it changes |
|---|---|---|
| `--sq` | `smoothquant.py` | inserts a `Mul` after LayerNorms that feed MatMul+Add; for each smoothable producer searches α∈[0,1] (SmoothQuant) and an OS+ style range clip, minimizing the MSE of fake-quantized `x/s @ (s·W)`; divides producer weights/bias by `s` and multiplies consumer MatMul weights by `s` |
| `--smc` | `softmax_correction.py` | for `Softmax → MatMul`, measures per-head mean of softmax row-max and inserts `Mul` ops so the softmax output uses the int8 range better; the calibrator then pins that `Mul`'s threshold to 1.0 |
| `--we` | `mix_precision.py: weight_equalization` | cross-layer equalization of consecutive conv pairs, `s = r1 / sqrt(r1·r2)` per channel |
| `--bc` | `mix_precision.py: run_bias_correction` | runs float and INT8 models (through real lowering), compares per-channel output means of each conv, and folds the difference into the float bias |

### 3.7 The quantize table (qtable)

Format: `op_name  MODE  [group_size]` per line, `#` comments allowed.  The
mode is any `Module_Mode` spelling (`F32`, `F16`, `BF16`, `INT8`, `INT4`,
`W4F16`, ...).  `ConvertTopToTpu::init_qtable` also accepts an inline string
`name:MODE,name:MODE`.  The optional third column populates
`LoweringConfig::group_map` (per-op group size).

### 3.8 Mixed-precision search tools

All search tools share one idea: **build real quantized models with
`mlir_lowering()` and run them in the interpreter**.  `MixQuantModel`
(`mix_precision.py`) lowers `<m>.mlir` to `<m>.mlir.<mode|mix>.tune.mlir` with a
candidate qtable and calibration table, loads it in `pymlir`, and compares
outputs with the float model by cosine, SNR, SQNR or MSE.

| `--search` | Class | Strategy |
|---|---|---|
| `search_threshold` | `SearchThreshold` | picks among per-method tables the one with the best network output |
| `search_qtable` | `SearchQtable` | generates tables for every method in `--quantize_method_list`; if the all-INT8 model already reaches `--expected_cos`, exits; otherwise measures per-layer sensitivity (per method, best threshold per op), then selects up to `--max_float_layers` float layers (`--cluster` uses clustering) and greedily adjusts |
| `fast_search` | `SearchQtableFast` | cheaper sensitivity estimate |
| `mix_search` | `MixPrecSearcher.run` | layer-wise mixed precision search |
| `search_qtable` + 4-bit mix modes | `SearchQtable4Bit` + `Int4SearchStrategy` / `W4A8SearchStrategy` / `F4F8SearchStrategy` | decides which Conv/MatMul layers can run 4-bit; maintains `#int4_th`/`#int4_op` blocks |

Because these tools call `tpuc-opt` and the interpreter, **every change to
lowering numerics changes search results**.  The interpreter is the oracle for
both calibration search and deploy validation.

---

## 4. Stage 2 — deploy orchestration (`model_deploy.py`, `mlir_shell.py`)

`DeployTool.lowering()` calls `mlir_lowering()`, which builds one `tpuc-opt`
command from `lowering_options()`:

| Option | Pass | Quantization role |
|---|---|---|
| `--processor-assign="chip mode num_device num_core addr_mode high_precision"` | `top::ProcessorAssignPass` | records the target and the **global** mode on the module |
| `--import-calibration-table="file asymmetric"` | `top::ImportCalibrationTablePass` | only when a table is given |
| `--processor-top-optimize` | `top::ProcessorOptimize` | chip-specific Top rewrites before lowering |
| `--fuse-preprocess=...` | optional | fuses input normalization (int8 input) |
| `--convert-top-to-tpu="qtable weightFileName asymmetric doWinograd q_group_size q_symmetric matmul_perchannel gelu_mode"` | `ConvertTopToTpu` | the actual quantization |
| `--canonicalize`, `--weight-fold` | | constant folding of quantized constants |

Deploy-side defaults that matter:

- Output names: `<model>_<chip>_<quantize>[_sym|_asym]_tpu.mlir`; target
  weights go to `..._tpu_weights.npz` (`..._qtable_weights.npz` with a qtable).
- `--q_group_size` (default 64) is forwarded **only** for `W4/W8 x F16/BF16`
  and `*DYN` modes; otherwise 0.
- Default `--tolerance` (lowered MLIR vs. float reference): INT8/W8/W4 →
  `0.90,0.80`; F16/BF16 → `0.99,0.90`; F32 → `0.99,0.99`; FP8 → `0.80,0.50`.
- Default `--correctness` (BModel vs lowered MLIR): `0.99,0.90`
  (`0.98,0.80` with a qtable).

`tpu_opt_options()` and `tpu_ada_options()` build the second command (section 7).

---

## 5. Stage 3 — Top-dialect passes

### 5.1 `--processor-assign` (`lib/Dialect/Top/Transforms/ProcessorAssign.cpp`)

Sets `module::setChip`, `setMode` (the global quantize mode), device/core
counts, address mode and `setHighPrecision`.  On CV18xx it also converts
integer `top.Input` types to f32, because CV18xx inputs are always float.
After this pass every `module::isBF16Modes()`, `isF8Modes()`, `isCV18xx()`, ...
query works, so it must come first.

### 5.2 `--import-calibration-table` (`lib/Dialect/Top/Transforms/ImportCalibration.cpp`)

1. Parses the table into maps: INT8 (`calibration_map`), INT4, FP8, per-channel
   weight scales, the asymmetric op list and the INT4 op list.  An `inf`
   entry is replaced by `6.5e4` with a warning.
2. Walks every op that implements `InferenceInterface` (plus `top.Input`):
   - SiLU/GELU in `#asym_op` → `output_asym = true`; Conv/MatMul in it →
     `input_asym = true`;
   - for each non-integer result: looks up the threshold (INT4 entry for INT4
     ops), computes `[min, max]` with `getMinMax`, and retypes the value to
     `!quant.calibrated<f32<min:max>>`;
   - CV18xx forces `min = -max`;
   - `top.Weight` users with a `#weight_scale` entry get a `scale` attribute.
3. For INT4 ops it pre-computes boundary attributes: `out_int8_scale`,
   `out_int8_zp` (when a consumer is not INT4) and `in_int4_scale`,
   `in_int4_zp` (when the producer is not INT4).
4. Sets state `TOP_CALIBRATED`.

`getMinMax` encodes op knowledge that calibration statistics cannot:

| Op | Range |
|---|---|
| `Abs` | `[-th, th]` |
| non-log `Sigmoid`, `Softmax` | fixed `[0, 1]` |
| `Mul` of identical inputs, or of Sigmoid/ReLU outputs | `[0, th]` |
| `Sub` | `[-th, th]`, or `[-1e-5, th]` if observed min ≥ 0 |
| symmetric (default) | `[-th, th]` if observed min < 0, else `[0, th]` |
| asymmetric | raw `[min, max]` |
| anything with `do_relu` (except Sub) | min forced to 0 |

### 5.3 Pre-quantized inputs: TFLite, TpuLang, QAT

- **TFLite int8** and **TpuLang quantized** graphs are imported with state
  `TOP_QUANTIZED`; their tensors already have `!quant.uniform` types.  No
  calibration table is used.  `ConvertTopToTpu` sets asymmetric mode and calls
  each lowering's `LoweringQuantized`, which typically emits the TFLite
  requant modes.
- **QAT (fake-quant ONNX)**: `python/tools/deploy_qat.py` strips
  FakeQuantize nodes, writes a calibration table from their scales/zero
  points, and records per-channel weight scales in the table's `#weight_scale`
  section.  The model then follows the normal calibrated path.  When a
  `top.Weight` has a `scale` attribute, Conv/MatMul INT8 lowering **uses that
  scale instead of computing max/127** (`filterOp.getScale()` in `Conv.cpp`
  and `MatMul.cpp`).
- `--convert-qdq-to-calibrated-dialect` (`QDQConvert.cpp`) converts
  `top.QuantizeLinear`/`DequantizeLinear` pairs into calibrated types and
  weight scales.  It is registered, but no Python driver in the tree invokes
  it.

### 5.4 `--processor-top-optimize`

Chip-specific Top rewrites (`lib/Dialect/Top/Transforms/ProcessorOptimize/
Optimize{BM1684,BM1684X,CV18XX}.cpp`) run **after** calibration import.
Newly created ops must therefore carry or derive a calibrated type.  If a new
op ends up with a plain f32 output, lowering treats it as float and inserts
casts around it.

---

## 6. Stage 4 — `--convert-top-to-tpu` (`lib/Conversion/TopToTpu/`)

### 6.1 Order inside `ConvertTopToTpu::runOnOperation`

```text
1. record group-quant info (q_group_size, q_symmetric), weight file name, FLOPs
2. TryInsertTile*  (broadcast fix-ups for binary ops, MatMul)
3. init_qtable()                 -> LoweringConfig::quantize_map (+ high_precision rules)
4. set_add_before_softmax_fp32() -> automatic float for badly balanced Adds
5. set_sub_float()
6. TOP_QUANTIZED ? isQuantized=true, asymmetric=true
                 : setAsymmetric(option); calibration_process()   (threshold propagation)
7. dynamic-quant mode marking (non-MatMul -> F16/BF16, MatMul -> per-op DYN mode)
8. chip/mode-specific automatic mixed precision (FP8, INT8: matmul_perchannel,
   YOLO head detection, MatMul->Softmax gets output_int16, ...)
9. shape-op lowering, Device2Host, control-flow (If/Loop) lowering, Host2Device
10. integer-subnet lowering (TopIntLowering)
11. MAIN: populateTopToTpuConversionPatterns(chip); greedy, maxIterations = 1
12. ForwardType patterns for Reshape/Squeeze/Unsqueeze/Tile (copy output type)
13. cast_process()   -> insert tpu.Cast wherever producer/consumer types disagree
14. relu_process()   -> relu_limit = -1 for quantized fused-ReLU ops
15. weight_process() -> placeholder weights cloned to F16/BF16
16. state TPU_LOWERED; abort if any non-Tpu op remains
```

### 6.2 Deciding the mode of each op

`LoweringConfig` (`TopLowering.h`) is a process-global table:

- `quantize_map[name] → Mode` comes from the qtable, from `--high_precision`
  (forces `AvgPool`, `Softmax`, `LayerNorm` to F32), and from the automatic
  rules in steps 4, 5 and 8.
- `getOpQuantMode(op)` returns the qtable mode or the global mode, with
  normalization: F16 becomes BF16 on CV18xx/CV184X/SGTPUV8 (no F16 there), and
  INT4 becomes INT8 for anything that is not Conv/MatMul.
- `init_qtable` normalizes chip limitations: on CV18xx, F32/F16 → BF16; on
  BM1684, any non-INT8 entry → F32; mixing `WxF16` with a BF16 global mode (or
  the reverse) aborts.

Automatic precision rules worth knowing:

| Rule | Where | Effect |
|---|---|---|
| Add inputs with threshold ratio > 64 | `set_add_before_softmax_fp32` | Add → F16 (F32 on BM1684) |
| MatMul whose only user is a float Softmax | step 8 | MatMul gets `output_int16`; its INT8 lowering emits an int16 output with scale `out_scale·255/65535` |
| YOLO detection head | step 8 | multi-MaxPool + Concat pattern triggers float handling of the head |
| `--matmul_perchannel` | step 8 (BM1684X/BM1688/CV184X) | sets `matmulPerchannelQuant` on every MatMul |

### 6.3 Threshold propagation (`calibration_process`)

The calibration table gives each tensor its own threshold.  Many ops cannot
change scale for free in hardware, or would waste a requant.  So before
lowering, the pass rewrites calibrated types so that "transparent" ops have
identical input and output scale:

| Pattern | Direction | Ops | Reason |
|---|---|---|---|
| `ForwardCalibartion` | output ← input | Reshape, Permute, Squeeze, Unsqueeze, Slice, Tile, Pad, Reverse, Upsample, MaxPool, ReLU, Min/MaxConst, LeakyReLU, PReLU (not CV18xx), Abs, MulConst, Arg; AvgPool on BM1684 | data movement and order-preserving ops are exact if scale is unchanged |
| `BackwardCalibartion` | input ← output (only if the input has one use) | ReLU, MaxPool, Depth2Space, Abs, PReLU/LeakyReLU (keep min) | lets the producer's requant write directly in the consumer's scale |
| `BackwardMutiInSingleOut` | all inputs ← output | Concat, Min, Max | concat must not requant each branch |
| `KeepSignPattern`, `KeepAddSignPattern`, `KeepMulSignPattern`, `Set*SignPattern` | sign fix-ups | AvgPool, MaxPool, Abs, Add, Mul, Sub, AddConst, SubConst | "backend does not support in/out of different sign" |
| `BackwardAddThToMuls` | BM1684X/BM1690 | Add | shares thresholds into Mul producers |
| `CompareCalibartion`, `SelectiveWhere`, `SelectiveMaskedFill` | | compare / select | keep boolean and selected branches consistent |

Rule of thumb: after this step, an INT8 Reshape → MaxPool → ReLU chain needs
zero requant ops.

### 6.4 Dispatch: `TopLowering<OpTy>::matchAndRewriteImpl`

Every op lowering derives from `TopLowering<OpTy>` and overrides
`LoweringINT8`, `LoweringINT4`, `LoweringF16`, `LoweringBF16`, `LoweringF32`,
`LoweringF8`, `LoweringQuantized`.  The base class removes `round_mode`
attributes (except on Cast) and switches on `getOpQuantMode(op)`:

```text
INT8, W4INT8                    -> LoweringINT8   (Conv: set doWinograd)
INT4                            -> LoweringINT4   (defaults to LoweringINT8)
F16, W8F16, W4F16, *F16DYN      -> LoweringF16
BF16, W8BF16, W4BF16, *BF16DYN  -> LoweringBF16
F8, F8E4M3, F8E5M2              -> LoweringF8
default                         -> LoweringF32
```

Weight-only and dynamic modes go to the float lowerings; the MatMul float
lowering then checks the mode and emits `tpu.A16MatMul` or a DYN MatMul.

Patterns are registered per chip family in `LoweringBM1684X.cpp`
(BM1684X/BM1688/BM1690/CV184X/SG2380/SGTPUV8 share it), `LoweringBM1684.cpp`
and `LoweringCV18xx.cpp`; the files under `BM1684X/`, `BM1684/`, `CV18xx/` are
named after the Top op.

### 6.5 Building the integer type

`getQuantInt8Type(v, asymmetric)` (`TopLowering.cpp`) is called by almost every
INT8 lowering:

```text
cali = calibrated type of v
scale, zp = getScaleAndZeroPoint(v, asymmetric)
cali.min >= 0 ? !quant.uniform<u8:f32, scale:zp> range [0,255]
              : !quant.uniform<i8:f32, scale:zp> range [-128,127]
```

`getQuantInt4Type`, `getQuantInt16Type`, `getQuantIntType(v, scale, zp, bits)`
and `getQuantBoolType` (scale 1, range [0,1]) follow the same pattern.
`lowering_common_int8<TpuOp>` replaces a Top op with the Tpu op of the same
name, keeps attributes and operands, and gives the result this type.  Ops that
are exact under shared scale (Reshape, Permute, MaxPool, Concat after
propagation, ...) use only that.

### 6.6 Canonical lowerings (BM1684X family)

#### Conv INT8 — `BM1684X/Conv.cpp: ConvLowering::LoweringINT8`

For an input with scale `sx, zx`, output `sy, zy`, filter `W` (shape
`[oc, ic/g, kh, kw]`) and float bias `b`:

```text
non-weight filter            -> float lowering (F32; BF16 on CV184X/SGTPUV8)
fsign = (min(W) < 0 || has_bias)        // int8 weights unless all positive and no bias
if W is already integral (to_all_int8)  -> scale_w = 1/times, no requantization of W
for each output channel c:              // always per-channel
    scale_w[c] = weight scale attr (QAT)       if present
               | MSE int4 search                if weight_bits == 4 and enabled
               | max(|W[c]|)/127 (or /255, /7, /15), at least 1e-5
    f[c]       = scale_w[c] * sx / sy
    (m[c], r[c]) = get_scale_and_shift(f[c], bitwidth=32)
    Wq[c]      = to_int8(W[c] / scale_w[c])          (or uint8)
    bias_i32[c]= round(b[c] / (scale_w[c]*sx) - Σ Wq[c]*zx)   // zero point folded into bias
```

Result: `tpu.Conv2D` with operands `(input, filter_i8, bias_int32)`,
attributes `quant_mode = MultiplierShift`, `multiplier = [m]`,
`rshift = [r]`, and an `!quant.uniform<i8>` output.  If `sx ≤ 1e-5` (an
all-zero calibration), `sx` is re-derived from bias or weight range to avoid
bias overflow.  3D convs instead produce an int32 conv followed by
`tpu.RequantIntAxis` with a per-channel quant tensor.

`get_scale_and_shift` brute-forces `n ∈ [-32, 31)` and keeps the `n` with the
smallest `|round(f·2ⁿ)/2ⁿ − f|` under the multiplier limit (INT_MAX for
bitwidth 32), which normally means the largest legal shift.

#### MatMul INT8 — `BM1684X/MatMul.cpp: MatMulLowering::LoweringINT8`

| Case | Weights | Requant |
|---|---|---|
| constant right operand (also through Reshape) | per-tensor `max/127` by default; per-column with `matmulPerchannelQuant` or a QAT `scale` attribute | per-tensor: one `(m, r)`; per-channel: one multiplier per N **with a common shift**, found by lowering the shift limit until no multiplier overflows `2³¹`; the multipliers become a `multi_int32` weight operand and `fuse_rq = true` |
| two activations (`A @ B`) | — | one `(m, r)` from `sx·sw/sy`; bias quantized with `sw·sx` |
| batch > 1 with input zero point, or asymmetric two-activation case | — | falls back to F32 |
| constant left operand | — | falls back to F16 (BF16 on CV184X/SGTPUV8) |

`output_int16` (set when the user is a float Softmax) widens the output to
int16.  On BM1684X2 per-tensor MatMul uses `OnlyScale` with an `out_scales`
float attribute.

#### Add INT8 — `BM1684X/Add.cpp`

Each input gets its own `(multiplier_i, rshift_i)` from
`coeff_i · s_i / s_out` via `get_scale_and_shift_positive` (bitwidth 8, so the
multiplier fits in int8).  Constant inputs are quantized with their own
abs-max.  The hardware evaluates `Σ (x_i · m_i) >> r_i` and saturates, so no
int32 tensor is materialized.

#### Nonlinear activations — lookup tables (`lib/Support/LutFunc.cpp`)

SiLU, Sigmoid, GELU, Tanh, Exp, ... are lowered in INT8 to `tpu.Lut` with a
256-entry table built by `create_lookup_table(in, out, func)`:

```text
for q in [-128, 127] (or [0, 255]):
    table[q mod 256] = sat( func((q - zx)·sx) / sy + zy )
```

The input scale, the output scale and the function are baked into the table,
so any change to either threshold changes the table bytes.

#### Ops without INT8 kernels

Their `LoweringINT8` delegates to a float lowering.  Examples: `LayerNorm`
lowers to F16 (BF16 on CV184X/SGTPUV8); `Softmax` with `log`, rank > 4 or
axis ≠ 1 goes F16/BF16.  The surrounding int8 tensors are then bridged by
casts (section 6.8).  An "INT8 model" is therefore a mixed-precision graph by
construction.

#### INT4 (`LoweringINT4`)

Only non-depthwise Conv and MatMul run INT4 (`module::isInt4Op`).  Weights are
quantized to `[-8, 7]`, stored in int8 containers, and packed two per byte at
weight reorder (`compact_coeff_for_int4`).  The importer-provided
`in_int4_scale/zp` and `out_int8_scale/zp` attributes define the INT8↔INT4
boundaries.  `do_cast` checks whether **all** users of a value are INT4-capable
before deciding the cast target.

#### FP8 (`LoweringF8`, BM1690 family)

E4M3: `sx = th_in / 448`, `sy = th_out / 448`; weights are cloned to
`f8E4M3FN` with per-channel scales (`clone_f8e4m3`); the bias is divided by
`sx·scale_w[c]`; the conv gets `quant_mode = OnlyScale` and
`out_f8_scales[c] = sx·scale_w[c] / sy`.  E5M2 uses unit scales.

#### Weight-only W4A16 / W8A16 and A16MatMul

`W4F16`, `W8BF16`, ... lower MatMul through `LoweringF16/BF16`, producing
`tpu.A16MatMul(input, weight, scale, zp, bias)`:

- Pre-quantized checkpoints (AWQ/GPTQ/AutoRound, `LLM_QUANTIZED` platform)
  already arrive as `top.A16MatMul`; `BM1684X/A16MatMul.cpp` only retypes the
  weight, scale (to BF16/F16), zp and bias.
- For float weights, `lib/Dialect/Tpu/Canonicalize/A16MatMul.cpp:
  computePerGroupParam` quantizes per group of `q_group_size` along K:
  asymmetric by default (`scale = (max−min)/(2ᵇ−1)`,
  `zp = clamp(−round(min/scale))`), symmetric with `--q_symmetric`; FP8/FP4
  group variants use `abs_max/448` or `abs_max/6`.
- `WeightReorder/BM1684X/A16MatMul.cpp` interleaves scale and zp per NPU lane
  (`[zp, scale]` pairs, or packed `[31:16] scale_bf16 | [15:0] zp_bf16`) so
  the kernel dequantizes on the fly.

### 6.7 Chip-family differences

| Family | Weights | Requant | Notes |
|---|---|---|---|
| BM1684 (`BM1684/Conv.cpp`) | int8, **per-layer** | `OnlyShift`: one `rshift` from `calRightShiftNum`; bias int16, shift reduced while > 3 % of bias overflows | no multiplier; qtable non-INT8 entries become F32 |
| CV18xx (`CV18xx/Conv.cpp`) | int8 per-channel | threshold-based `qscale = (max_w·th_x)/(127·th_y)`, forced < 1 (no left shift); `getRShiftAndMultiplierFromQScale` (QDM) | always symmetric signed; float = BF16 |
| BM1684X / BM1688 / BM1690 / CV184X / SG2380 / SGTPUV8 | int8 per-channel, int4 on BM1688/CV186X/BM1690 | `MultiplierShift` per channel | requant words: BM1684X `{m, −r, zp}` (3×int32); BM1688/SG2380/CV184X/SGTPUV8 `{m, (−r & 0xff) | zp<<16}` (2×int32) |
| BM1684X2 | as above | per-tensor MatMul `OnlyScale` | |

### 6.8 Casts between precisions (`cast_process`)

After all ops are lowered, every operand's type is checked against what its
consumer expects:

1. The consumer's `TypeInterface::type_verify` (or `type_verify_case_same`,
   or for `return` the function result type) gives a target type and a
   `TypeCastMode` (`lib/Interfaces/TypeInterface.cpp`):
   - float → quantized: `DO_QUANTIZE`;
   - quantized → float: `DO_DEQUANTIZE`;
   - float → FP8: `DO_QUANTIZE`; FP8 → float: `DO_DEQUANTIZE`;
   - otherwise `DO_CAST`.  Integer-to-integer inputs of Add/Sub/Mul/Shift are
     accepted as-is, because their own multipliers absorb scale differences.
2. `ConvertTopToTpu::do_cast` reuses an existing cast if one already produces
   the needed type, otherwise inserts `tpu.Cast` (named
   `<value><user>_<type>`).  The Cast carries the uniform type, so its scale
   and zero point come from the types, not from attributes.
3. Weights and `quant_inner_requant` operands are skipped.

`relu_process` sets `relu_limit = -1` on quantized fused-ReLU ops (no upper
clip, the int8 saturation is the clip).  `weight_process` materializes
placeholder weights in F16/BF16.

Integer-to-integer scale changes inside the graph use
`do_transfer` (→ `tpu.MulShift` or `tpu.RequantInt`) or `do_transfer_fp`
(→ `tpu.RequantFp`), both in `TopLowering.cpp`.

### 6.9 Worked example: one INT8 conv channel

Inputs: input threshold 4.0 (signed), output threshold 8.0 (signed), channel
weight abs-max 0.5, bias 0.1.

```text
sx      = 4.0 / 127 = 0.0314961
sy      = 8.0 / 127 = 0.0629921
scale_w = 0.5 / 127 = 0.0039370
f       = scale_w * sx / sy = 0.25 / 127 = 0.0019685
(m, r)  ≈ (2113665, 30)          since 2113665 / 2^30 = 0.0019685
Wq      = round(W / 0.0039370)   in [-127, 127]
bias_q  = round(0.1 / (0.0039370 * 0.0314961)) = 806
y_q     = sat_int8( RightShiftRound((Σ x_q·Wq + 806) · 2113665, 30, HALF_UP) + 0 )
y_real  = y_q · 0.0629921
```

These exact numbers are what the interpreter computes
(`tpu::Conv2DOp::inference`) and what the backend's requant stage is expected
to reproduce bit for bit.

---

## 7. Stage 5 — Tpu-dialect passes that touch quantization

### 7.1 `--weight-fold`, `--canonicalize`, Deinit

`--weight-fold` folds ops whose inputs are all constants, now in integer form.
When the lowering command finishes, `tpuc-opt`'s Deinit pass calls
`module::saveWeight()`, so `_tpu_weights.npz` contains `filter_i8`,
`bias_int32`, `table`, `multi_int32`, `quant` ... entries and drops unused F32
originals.

### 7.2 `--strip-io-quant` (`StripIOQuant.cpp`)

By default the BModel accepts F32 and returns F32, so the boundary
`tpu.Cast` ops (F32↔INT8, F32↔F16) are compiled into the model.  With
`--quant_input`, `--quant_output`, `--quant_input_list`, `--quant_output_list`,
`--quant_input_int8`, `--quant_output_int8` or `--quant_output_bf16`, the pass
deletes those casts and retypes `top.Input` / the function results.  The
application then supplies or receives integer (or half) data and must apply
the scale itself.  The scale is visible in the BModel tensor info.

### 7.3 `--processor-tpu-optimize` (`ProcessorOptimize/Optimize<chip>.cpp`)

Quantization-relevant patterns on BM1684X-family chips include:

| Pattern | Effect |
|---|---|
| `ConvMergeRequant` | `Conv2D(int32 out) → RequantInt[Axis]` merged into one conv with multiplier/rshift; detects int32 accumulator or bias overflow risk and splits the bias when that preserves the result |
| `MatMulRequantIntFusion` | `MatMul → RequantIntAxis(fuse_rq_axis)` merged into the MatMul (BM1684X/BM1688) |
| `SplitQuantizedMLP2Pattern`, `SplitMixedQuantizedMLPPattern` | split large quantized MLPs |
| `QuantizedSoftmaxPattern` | quantized softmax rewrite |
| `CastWithoutScalePattern` (BM1684) | simplifies casts that do not need scaling |

### 7.4 `--weight-reorder` (`WeightReorder/<chip>/Conv2D.cpp`, ...)

This is where FORM 3 becomes FORM 4.  For a BM1684X-family INT8 conv:

1. Filter re-layout: IC is padded to `IC_PARALLEL` (`BM168x::ic_num(1)` for
   int8, `ic_num(0.5)` for int4) and kh/kw may be folded into IC
   (`use_3ic_optimize`); stride > 15 uses a cell layout.  Channels are spread
   across NPUs (`reshape_coeff_for_broadcast_channel`).  4-bit weights are
   packed (`compact_coeff_for_int4`).
2. Requant words: `{m, −r, zp}` (BM1684X) or `{m, (−r & 0xff) | zp<<16}`
   (BM1688 family) per output channel.  FP8 convs store one float scale per
   channel.
3. One merged coefficient per output channel:

```text
row(oc) = [ requant words | bias int32 (aligned to 4 B) | filter bytes (aligned to EU_BYTES) ]
shape   = [1, oc_per_lane, 1, merge_w]   (split further when > MAX_TPU_DIM)
```

4. `rshift`, `multiplier`, `out_f8_scales` attributes are **removed** and
   `coeff_merged = true` is set.  After this pass the requant parameters live
   only in the weight bytes.  `final.mlir` will no longer show them; read the
   `_tpu.mlir` instead.

If the conv output is int32 (a separate requant op follows), only bias and
filter are merged (`merge_with_requant = false`).

### 7.5 Layer group and address assignment

Neither pass reasons about scales, but both size buffers by storage type
(`BM168x::getFmtBytes(BM168x::getDataType(v))`): an INT8 tensor uses 1 byte
per element, an INT4 weight 0.5, F16 2, F32 4.  Quantization therefore changes
tiling decisions, how many layers fit in one local-memory group, and DDR
traffic.  This is a main source of INT8 speed-up besides the TIU arithmetic.
Local-memory buffer requests also depend on dtype; for example, a conv adds an
int32 partial-sum buffer (`getBufferSize_bm1684x`).

---

## 8. Stage 6 — code generation and instructions

### 8.1 From Tpu op to backend parameter struct

Each Tpu op implements `codegen_global_<chip>()` (whole tensor in DDR) and
`codegen_local_<chip>()` (one tile inside a layer group) in
`lib/Dialect/Tpu/Interfaces/<chip>/<Op>.cpp`.  For `tpu.Conv2D` on BM1684X
(`Interfaces/BM1684X/Conv2D.cpp`) the quantization-relevant fields of
`conv_global_spec_t` (`include/tpu_mlir/Backend/BM168x/Param.h`) are:

| Field | Value | Source |
|---|---|---|
| `common.ipad_value` | input zero point | input `!quant.uniform` type |
| `common.is_asym` | 1 for integer inputs | |
| `common.if_relu` | forced on when the output is unsigned | uint8 output encodes ReLU |
| `common.round_mode` | from `round_mode` attr | |
| `common.has_bias`, `bias_sign` | | |
| `common.kzp_value` | kernel zero point | |
| `common.weight_bits` | 4 or 8 | `weight_bits` attr |
| `common.use_3ic_optimize` | layout chosen at weight reorder | |
| `merge_coeff` | 2 = `(requant, bias, weight)` merged; 1 = `(bias, weight)`; 0 = separate | int32 output → 0/1 |
| input/output tensor specs | address, shape, `DTYPE_INT8/UINT8/INT4/...` | `BM168x::get_input_spec/get_output_spec` |

Ops and their backend entry points for the quantized path (BM1684X family):

| Tpu op | Backend symbol |
|---|---|
| `Conv2D` | `backend_api_conv_global` / `_local` |
| `MatMul` | `backend_api_fc_global`, `backend_api_fc_multi_core_global`, `backend_api_batch_matmul_global` |
| `A16MatMul` | `backend_api_a16_matmul_global` |
| `RequantInt`, `RequantIntAxis` | `backend_api_requant_int_global` |
| `RequantFp`, `Cast` F32→INT8 | `backend_api_requant_float_global` (`scale_value = 1/scale`, `offset_value = zp`) |
| `Cast` INT8→F32 | `backend_api_dequant_float_global` (`scale_value = scale`, `offset_value = zp`) |
| `Lut` | `backend_api_lut` |
| `Add` (int8) | `backend_api_bcbinary_global` with per-input multipliers/shifts |

### 8.2 From backend call to instruction words

`BM168x::call_global_func(name, &spec, size, in_specs, out_specs)`
(`lib/Backend/BM168x/BM168x.cpp`) resolves `name` with `dlsym` in the loaded
`libbackend_<chip>.so` (`third_party/nntoolchain/lib/`) and calls it.  The
backend function writes TIU (compute) and GDMA (DMA) command descriptors into
the command buffers that `BM168x` owns.  `Codegen` (`BM168xCodegen.cpp`)
collects those buffers per subnet together with the coefficient blob (the
merged weights from section 7.4, at the addresses from `--address-assign`) and
serializes the BModel.

What the backend does for an INT8 conv is not visible in this repository.
From the spec and from the interpreter, which is required to match it, the
computation is:

```text
GDMA  : load input tile (int8/uint8), merged coeff row(s)
TIU   : int8 × int8 conv, int32 accumulate, + bias_int32      (pad with ipad_value = zx)
TIU   : per-channel  (acc · m) >> r  with round_mode, + zy, saturate to int8/uint8
        (ReLU via unsigned output or if_relu)
GDMA  : store output tile
```

When the conv output is int32, the requant is a separate
`backend_api_requant_int_global` call.  Inside a layer group, the same
arithmetic is emitted by the `_local` entry points on local-memory addresses
assigned by the layer-group pass.

---

## 9. Verification: why the interpreter is the contract

Every Tpu op implements `inference()` (`lib/Dialect/Tpu/Interfaces/Common/
<Op>.cpp`) with **integer-exact** semantics.  For `Conv2D`:

```text
conv in float on integer-valued data (oneDNN)  ->  int32-exact accumulator
for each channel: v = applyMultiplierAndRShift(acc + bias, m, r, qmode, rmode) + zp
                  ReLU clamp at zp ; saturate(v, out_type)
rmode = HALF_AWAY_FROM_ZERO for TFLite/QDM modes, HALF_UP for MultiplierShift
```

`applyMultiplierAndRShift` (`MathUtils.cpp`) is the single definition of each
`RequantMode`.  CV18xx `MultiplierShift` is computed in float `(v·m)/2^r` with
`to_int`, which reflects that hardware's behavior.

This gives two validation stages with different meanings:

| Stage | Compares | Default threshold | A failure means |
|---|---|---|---|
| `validate_tpu_mlir` | interpreter(`_tpu.mlir`) vs Top F32 outputs (`ref_npz`) | `--tolerance` (INT8 `0.90,0.80`) | **quantization error**: thresholds, precision choice, or a lowering formula |
| `validate_model` | BModel outputs vs interpreter(`_tpu.mlir`) | `--correctness` (`0.99,0.90`) | **codegen/backend mismatch**: the two should be nearly bit-exact; this is not a calibration problem |

For `TOP_QUANTIZED` models, the BModel is compared with the reference directly.

Both stages call `npz_tool.py compare` (cosine and Euclidean similarity per
tensor).  `--compare_all` compares every intermediate tensor, and
`--fazzy_match` matches tensor names loosely.

The same interpreter also drives calibration (statistics on Top F32),
tuning (fake-quantized Top), and every search tool (real lowered models).  A
wrong `inference()` therefore misleads calibration, search and validation at
once, while a wrong backend kernel only shows up in `validate_model` or on
hardware.

---

## 10. How everything connects

### 10.1 Artifact chain

| Artifact | Producer | Consumers |
|---|---|---|
| `<m>.mlir`, `<m>_top_origin_weight.npz` | `model_transform.py` | calibration (may modify the NPZ), lowering |
| `<m>_cali_table` | `run_calibration.py` | `ImportCalibrationTablePass`; search tools |
| qtable | `run_calibration.py --search ...`, `shape_pattern_qtable`, the user | `ConvertTopToTpu::init_qtable`; search tools |
| calibrated types in IR | `ImportCalibrationTablePass`, then `calibration_process` | every `LoweringINT8` via `getScaleAndZeroPoint` |
| `!quant.uniform` types | lowering (`getQuantInt8Type`...) | cast insertion, interpreter, codegen (zero points, dtypes) |
| `filter_i8`, `bias_int32`, `table`, `quant`, `multi_int32` weights | lowering | weight reorder, interpreter |
| `multiplier`/`rshift` attributes | lowering | interpreter; weight reorder (merged and then removed) |
| merged coefficient bytes | weight reorder | address assign, codegen, BModel coeff section |
| `*_spec_t` structs | `codegen_*` | `libbackend_<chip>.so` |
| TIU/GDMA commands | backend | BModel, runtime |

### 10.2 Change-impact rules

| If you change ... | ... also check |
|---|---|
| a threshold method or table format | `ImportCalibration.cpp` parser, `CalibrationTable.parse`, search tools reading `_<method>` tables |
| `getScaleAndZeroPoint` / `getScale` | every lowering, `create_lookup_table`, cast codegen; regenerate regression baselines |
| an op's `LoweringINT8` formula | its `inference()` (must stay bit-exact), its weight reorder, its codegen spec |
| requant word packing | lowering (3D conv path), `WeightReorder/*/Conv2D.cpp`, `ConvMergeRequant`; the backend's expected layout |
| a new Top op | calibration propagation (is it transparent?), `LoweringINT8` or an explicit float fallback, `type_verify` |
| a new chip | `getOpQuantMode` normalization, `init_qtable` restrictions, requant packing, `run_calibration.py` chip list, `regression/chip.py` |
| `processor-top-optimize` patterns | new ops must carry calibrated types, otherwise they become float islands |

---

## 11. Debugging recipes

Run these inside `./tmp`, not in the source tree.

1. **Is the problem quantization or codegen?**  Check which validation stage
   failed (section 9).
2. **Find the first bad tensor.** Re-run deploy with `--compare_all`, then
   `npz_tool.py compare <prefix>_tpu_outputs.npz <m>_top_outputs.npz -vv`.  The
   first layer whose cosine drops sharply is the culprit; its threshold in the
   table and its precision in the qtable are the first things to look at.
3. **Try float for one layer.** Add `layer_name F16` to a qtable and redeploy;
   if accuracy recovers, run `run_calibration.py --search search_qtable` to
   find a minimal set.
4. **Check the threshold.** Compare the table's threshold with min/max on the
   same line: a threshold far below `max(|min|,|max|)` means heavy clipping
   (KL); try `--cali_method mse` or `percentile9999` for that model.
5. **Inspect quant parameters.** In `_tpu.mlir` (before weight reorder), read
   the output type `!quant.uniform<i8:f32, scale>` and the `multiplier`/
   `rshift` attributes of the op.  `final.mlir` no longer has them.
6. **Inspect the lowering.** `--debug-only=pattern-application,
   dialect-conversion` (`--patterns_count` in deploy) shows which patterns
   fired.
7. **Transformers**: try `--sq`, `--smc`, `--matmul_perchannel`, or keep
   LayerNorm/Softmax/Add-before-Softmax in F16 (the automatic rules usually do
   that already).

---

## 12. Defects observed during this study

These are reading findings, not test failures.  Each one should be confirmed
with a test before it is fixed.

1. **`MatMulRequantIntFusion` ignores its own early exit** —
   `lib/Dialect/Tpu/Transforms/ProcessorOptimize/OptimizeBM1684X.cpp:5093-5095`:
   `if (m_fuse_rq) failure();` discards the result instead of
   `return failure();`, so a MatMul that already fuses per-channel requant is
   still considered for fusion.  The following checks (single
   `RequantIntAxis` user with `fuse_rq_axis`) probably make it unreachable in
   practice, but the code does not do what the comment says.
2. **INT4 + ACIQ calibration crashes** — `python/calibration/kld_calibrator.py:442`
   and `:455` test `out not in self.aciq_g` / `self.aciq_l` (the 8-bit dicts,
   just filled above) instead of `self.aciq_g4` / `self.aciq_l4`.  The 4-bit
   list is therefore never created, and the `else` branch's
   `self.aciq_g4[out].append(...)` raises `KeyError` on the first sample when
   `int4` and `aciq_gauss`/`aciq_laplace` are combined.
3. **Dead mixed-precision rule with a side effect** —
   `lib/Conversion/TopToTpu/TopToTpuPass.cpp` around line 1694-1700 (INT8 mode,
   MatMul with a single Add user): the block creates an attribute `tmp` and
   never sets it.  The condition also uses
   `LoweringConfig::quantize_map[name]` outside the `find` guard, and
   `operator[]` **inserts** a default entry (`Mode::INT8`, value 0) for every
   such Add.  Later code that asks "is this op in the qtable?" with `find()`
   then sees an entry the user never wrote.

---

## 13. Reading order

```text
python/tools/run_calibration.py
python/calibration/kld_calibrator.py, cali_math.py, utils.py, simple_tunner.py
bindings/kld/calibration_math.cpp
python/utils/mlir_shell.py (lowering_options, mlir_lowering, tpu_opt_options, tpu_ada_options)
lib/Dialect/Top/Transforms/ProcessorAssign.cpp
lib/Dialect/Top/Transforms/ImportCalibration.cpp
lib/Support/Module.cpp (getScale, getScaleAndZeroPoint)
lib/Support/MathUtils.cpp (get_scale_and_shift*, QuantizeMultiplier, RightShiftRound, applyMultiplierAndRShift)
include/tpu_mlir/Conversion/TopToTpu/TopLowering.h (LoweringConfig, getOpQuantMode, TopLowering)
lib/Conversion/TopToTpu/TopLowering.cpp (getQuantInt8Type, do_requant*, do_transfer*)
lib/Conversion/TopToTpu/TopToTpuPass.cpp (runOnOperation, calibration_process, init_qtable, cast_process)
lib/Conversion/TopToTpu/BM1684X/{Conv,MatMul,Add,SiLU,Softmax,LayerNorm,A16MatMul}.cpp
lib/Support/LutFunc.cpp
lib/Interfaces/TypeInterface.cpp
lib/Dialect/Tpu/Interfaces/Common/Conv2D.cpp (inference)
lib/Dialect/Tpu/Transforms/StripIOQuant.cpp
lib/Dialect/Tpu/Transforms/ProcessorOptimize/OptimizeBM1684X.cpp (ConvMergeRequant, MatMulRequantIntFusion)
lib/Dialect/Tpu/Transforms/WeightReorder/BM1684X/Conv2D.cpp, A16MatMul.cpp
lib/Dialect/Tpu/Canonicalize/A16MatMul.cpp
lib/Dialect/Tpu/Interfaces/BM1684X/{Conv2D,MatMul,Cast,RequantInt,Lut}.cpp
include/tpu_mlir/Backend/BM168x/Param.h
lib/Backend/BM168x/BM168x.cpp (call_global_func)
```

For another chip, swap `BM1684X/` for `BM1684/` or `CV18xx/` in the
conversion, weight-reorder and interface paths and compare against
section 6.7.

