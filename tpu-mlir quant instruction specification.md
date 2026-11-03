# Volume 05a — the quantized data lifecycle: where the numbers actually change

Volume 05 explains *how* each quantization parameter is computed.  This
companion volume answers a different question: **at which exact moment do
real numbers turn into integers, which component does it, and what does the
quantization look like in each artifact on the way?**

It follows one small network from the ONNX file to TPU execution, showing for
every phase:

- what the IR or file looks like (the "instructions" that describe the
  quantization at that phase);
- whether any data bytes are changed, and if so which bytes and by whom;
- which code is responsible.

Studied at commit `0f2f7d6bb`.  Function names are the stable anchors; line
numbers drift.  IR snippets are abbreviated for reading (attributes dropped,
shapes shortened) but use the real op names, type syntax and weight-naming
rules from the code.  Numeric values are computed with the formulas quoted
from the source; the multiplier/shift values were reproduced with a Python
re-implementation of `get_scale_and_shift*` and should be confirmed with the
interpreter if you depend on the exact digits.

---

## 1. The short answer

There are two kinds of data, and they are quantized at completely different
times.

| Data | When it becomes integer | Who does it | How often |
|---|---|---|---|
| **Weights and biases** | during `--convert-top-to-tpu` (compile time) | each op's `LoweringINT8` (`lib/Conversion/TopToTpu/<chip>/<Op>.cpp`) via `top::WeightOp::create` | once per compilation |
| **Lookup tables** (SiLU, Sigmoid, ...) | same pass | `create_lookup_table` (`lib/Support/LutFunc.cpp`) | once per compilation |
| **Activations** (input, intermediate, output tensors) | **never at compile time** — only at inference time | a `tpu.Cast` (F32→INT8) running on the TPU, or the host application if `--quant_input` was used | every inference |

The compiler never sees activation *values*.  For activations it produces only
**instructions**: types that say "this tensor is int8 with scale s", Cast ops
that say "quantize here", and multiplier/shift words that say "rescale here".
The actual conversion of an activation from float to int8 happens when the
first `tpu.Cast` runs, either inside the CPU interpreter (`model_runner.py` on
`_tpu.mlir`) or inside the chip (BModel).

```text
                       COMPILE TIME                                              RUN TIME
  ┌───────────────────────────────────────────────────────────┐    ┌───────────────────────────────────┐
  weights  f32 ──(lowering)──► i8/i32 ──(reorder)──► merged bytes ──(codegen)──► coeff blob ──► DDR ──► TIU
  activations:   only types + Cast ops + requant words ────────────────────────► cmds ──► input f32 ──Cast──► i8 ──► ... ──Cast──► f32
```

The rest of this volume shows every intermediate step.

---

## 2. Every place where bytes change

This is the complete list of mutations on the INT8 path, in order.  "Value
change" means the numbers change; "layout change" means the same numbers are
moved, padded or packed.

| # | Phase | Data | Change | Code |
|---|---|---|---|---|
| M0 | `run_calibration.py` (optional) | float weights in `<m>_top_origin_weight.npz` | **value change, still float**: SmoothQuant scales MatMul weights, WeightEqualization rebalances Conv pairs, BiasCorrection adjusts conv bias | `python/calibration/smoothquant.py` (`np.savez(self.parser.module_weight_file, ...)`), `mix_precision.py: weight_equalization`, `run_bias_correction` |
| M1 | `--convert-top-to-tpu` | Conv/MatMul filter, bias, constant operands | **value change float → int**: `filter_i8 = to_int8(W/scale_w)`, `bias_int32 = round(b/(scale_w·sx) − Σ Wq·zx)` | `BM1684X/Conv.cpp: LoweringINT8`, `MatMul.cpp`, `Add.cpp` (constant inputs → `<name>_i8`) |
| M2 | same pass | nonlinear functions | **new data**: 256-entry int8/uint8 table with input and output scale baked in | `create_lookup_table` |
| M3 | Deinit of the lowering command | weight file | **selection**: unused float originals are dropped, new integer tensors written | `module::saveWeight()` (`lib/Support/Module.cpp`) → `<prefix>_tpu_weights.npz` |
| M4 | `--weight-reorder` | int8 filter, int32 bias, multiplier/rshift attributes | **layout change** (IC padding, NPU spreading, int4 packing) and **attribute → bytes** (requant words written into the coefficient, attributes deleted) | `WeightReorder/BM1684X/Conv2D.cpp` → `<conv>_merge` weight |
| M5 | `--address-assign` + `--codegen` | all weights | **placement**: each weight's bytes are `memcpy`'d into one coefficient blob at its assigned address | `BMCodegen::CreateCoeffMem` (`BM168xCodegen.cpp`) |
| R1 | inference | model input | **value change f32 → int8** | `tpu.Cast` → `backend_api_requant_float_global` on the chip; or host side `lowering()` in `python/utils/lowering.py` if the cast was stripped |
| R2 | inference | every intermediate tensor | **int → int with a new scale** (requant) | TIU kernels; interpreter: `applyMultiplierAndRShift` |
| R3 | inference | model output | **value change int8 → f32** | `tpu.Cast` → `backend_api_dequant_float_global`; or host side in `model_runner.py` |

Nothing else changes values.  Layer group, address assignment and codegen
move bytes; they never re-quantize.

---

## 3. The running example

```text
            x (f32, 1x2x1x1)
                 │
         conv1 1x1, ic=2 → oc=…, ReLU fused        (threshold 2.0, ≥ 0 → uint8)
                 │──────────────┐
         conv2 1x1                │                 (threshold 3.0, signed int8)
                 │                │
                add ◄─────────────┘                 (threshold 4.0, signed int8)
                 │
              sigmoid                                (range [0,1] → uint8)
                 │
              output (f32)
```

Concrete numbers for output channel 0 of `conv1`:

| Quantity | Value |
|---|---|
| input `x` | `[1.30, −0.75]` |
| input threshold | `4.0` |
| conv1 channel-0 weights `W` | `[0.50, −0.20]` |
| conv1 channel-0 bias `b` | `0.10` |
| conv1 output threshold | `2.0` (calibrated min ≥ 0 because of ReLU) |
| float reference of conv1 ch 0 | `relu(1.30·0.50 + (−0.75)(−0.20) + 0.10) = 0.90` |

---

## 4. Phase by phase

### Phase 0 — `model_transform.py`: pure float

**Artifacts:** `<m>.mlir` (state `TOP_F32`), `<m>_top_origin_weight.npz`,
`<m>_in_f32.npz`, `<m>_top_outputs.npz`.

```mlir
%x  = "top.Input"(%arg0)                       : (tensor<1x2x1x1xf32>) -> tensor<1x2x1x1xf32>
%w1 = "top.Weight"() loc("conv1_filter")       : () -> tensor<8x2x1x1xf32>
%b1 = "top.Weight"() loc("conv1_bias")         : () -> tensor<8xf32>
%c1 = "top.Conv"(%x, %w1, %b1) {do_relu = true, ...} : (...) -> tensor<1x8x1x1xf32>   loc("conv1")
%c2 = "top.Conv"(%c1, %w2, %b2) ...           -> tensor<1x8x1x1xf32>   loc("conv2")
%a  = "top.Add"(%c1, %c2)                      -> tensor<1x8x1x1xf32>   loc("add")
%s  = "top.Sigmoid"(%a)                        -> tensor<1x8x1x1xf32>   loc("sigmoid")
```

**Quantization instructions present:** none.  `do_relu = true` comes from
the Top canonicalization that fuses Conv + ReLU; it matters later because it
forces the calibrated minimum to 0.

**Data:** NPZ holds `conv1_filter = [[0.50, −0.20], ...]` as float32.

### Phase 1 — `run_calibration.py`: statistics, maybe float weight edits

The interpreter (`pymlir`) runs the **float** Top MLIR on the calibration
images.  `ActivationCalibrator` records min/max/histograms per tensor and
produces thresholds.

**Artifact (FORM 1):**

```text
# op_name    threshold    min     max
x            4.000000    -3.912   3.990
conv1        2.000000     0.000   2.731
conv2        3.000000    -2.950   2.402
add          4.000000    -3.700   3.880
sigmoid      1.000000     0.024   0.977
```

**Data changes:** only M0, and only if requested (`--sq`, `--we`, `--bc`).
They overwrite the float NPZ in place; the model stays float.  This is the
only phase that alters *float* values to make later quantization easier.

**Not changed:** activations.  Calibration never stores quantized
activations; the samples are discarded after statistics are gathered.

### Phase 2 — `--processor-assign`, `--import-calibration-table`: types only

`ImportCalibrationTablePass` rewrites **types**, not data (FORM 2).  The range
comes from `getMinMax`: because `conv1` has `do_relu`, its min becomes 0.

```mlir
%x  = "top.Input"(%arg0) -> tensor<1x2x1x1x!quant.calibrated<f32<-4.000000e+00:4.000000e+00>>>
%c1 = "top.Conv"(...)    -> tensor<1x8x1x1x!quant.calibrated<f32<0.000000e+00:2.000000e+00>>>
%c2 = "top.Conv"(...)    -> tensor<1x8x1x1x!quant.calibrated<f32<-3.000000e+00:3.000000e+00>>>
%a  = "top.Add"(...)     -> tensor<1x8x1x1x!quant.calibrated<f32<-4.000000e+00:4.000000e+00>>>
%s  = "top.Sigmoid"(...) -> tensor<1x8x1x1x!quant.calibrated<f32<0.000000e+00:1.000000e+00>>>
```

The sigmoid range is fixed to `[0, 1]` by `getMinMax`, independent of the
observed min/max.  Weights are untouched (`top.Weight` results are skipped,
except for receiving a QAT `scale` attribute when the table has a
`#weight_scale` section).

Module state: `TOP_CALIBRATED`.  The interpreter can still run this file and
produces exactly the float results, because a calibrated type is stored as
f32.

### Phase 3 — inside `--convert-top-to-tpu`, before lowering: range propagation

`calibration_process` may further rewrite calibrated types (Volume 05 §6.3).
In this example nothing transparent sits between ops, so ranges stay as above.
In a network with `conv → maxpool → reshape`, this is the step where the
MaxPool and Reshape inherit the conv's range so that they need no requant.

### Phase 4 — `--convert-top-to-tpu`: the compile-time data change

This is the single place where weight values become integers (M1, M2).  The
pass lowers each op, then `cast_process` inserts `tpu.Cast` ops at precision
boundaries.

#### 4.1 Types become storage types (FORM 3)

`getQuantInt8Type` turns each calibrated type into a uniform quantized type:

| Tensor | Calibrated range | Uniform type | scale |
|---|---|---|---|
| `x` | `[−4, 4]` | `!quant.uniform<i8:f32, 0.031496>` | `4/127` |
| `conv1` | `[0, 2]` | `!quant.uniform<u8:f32, 0.0078431>` | `2/255` |
| `conv2` | `[−3, 3]` | `!quant.uniform<i8:f32, 0.023622>` | `3/127` |
| `add` | `[−4, 4]` | `!quant.uniform<i8:f32, 0.031496>` | `4/127` |
| `sigmoid` | `[0, 1]` | `!quant.uniform<u8:f32, 0.0039216>` | `1/255` |

#### 4.2 Weights become integers (M1)

`ConvLowering::LoweringINT8` (`BM1684X/Conv.cpp`) for conv1, channel 0:

```text
scale_w      = max(|0.50|, |−0.20|) / 127                     = 0.0039370
Wq           = to_int8([0.50, −0.20] / 0.0039370)             = [127, −51]
bias_int32   = round(0.10 / (0.0039370 · 0.031496) − 0)       = 806          (zx = 0, symmetric)
f            = scale_w · sx / sy = 0.0039370·0.031496/0.0078431 = 0.0158100
(m, r)       = get_scale_and_shift(f, 32)                     = (4243973, 28)   4243973 / 2^28 = 0.0158100
```

`get_scale_and_shift` keeps the **first** shift with the smallest float32
error, so `r` is not necessarily the largest legal shift; here several shifts
tie and 28 wins.

New weights are created with `top::WeightOp::create(op, suffix, data, type)`.
The name is always `<op name>_<suffix>` (with a numeric disambiguator if it
already exists), and the bytes are added to the in-memory weight file at the
same moment:

```mlir
%w1q = "top.Weight"() loc("conv1_filter_i8")   : () -> tensor<8x2x1x1xi8>
%b1q = "top.Weight"() loc("conv1_bias_int32")  : () -> tensor<1x8x1x1xi32>
```

The float `conv1_filter` / `conv1_bias` lose their last user and disappear
from the IR.

#### 4.3 The lowered ops carry the requant instructions

```mlir
%x_i8 = "tpu.Cast"(%x) loc("x_<i8 type suffix>")
          : (tensor<1x2x1x1xf32>) -> tensor<1x2x1x1x!quant.uniform<i8:f32, 0.031496>>

%c1 = "tpu.Conv2D"(%x_i8, %w1q, %b1q) {
        do_relu = true, relu_limit = -1.0, with_bias = true,
        quant_mode = #tpu<rq_mode MultiplierShift>,
        multiplier = [4243973, ...], rshift = [28, ...]}
          -> tensor<1x8x1x1x!quant.uniform<u8:f32, 0.0078431>>        loc("conv1")

%c2 = "tpu.Conv2D"(%c1, %w2q, %b2q) {multiplier = [...], rshift = [...]}
          -> tensor<1x8x1x1x!quant.uniform<i8:f32, 0.023622>>         loc("conv2")

%a  = "tpu.Add"(%c1, %c2) {multipliers = [1, 3], rshifts = [2, 2]}
          -> tensor<1x8x1x1x!quant.uniform<i8:f32, 0.031496>>         loc("add")

%tb = "top.Weight"() loc("sigmoid_table") : () -> tensor<1x1x1x256xui8>
%s  = "tpu.Lut"(%a, %tb)
          -> tensor<1x8x1x1x!quant.uniform<u8:f32, 0.0039216>>        loc("sigmoid")

%o  = "tpu.Cast"(%s) loc("sigmoid_f32")
          : (...u8...) -> tensor<1x8x1x1xf32>
return %o
```

Where each quantization instruction lives in this file:

| Instruction | Representation | Created by |
|---|---|---|
| "quantize the input" | `tpu.Cast` f32 → `!quant.uniform<i8>`; scale is in the result type, no attribute | `cast_process` / `do_cast`, because `tpu.Conv2D` wants the quantized type from `type_verify` |
| "accumulate in int32 and rescale per channel" | `multiplier`/`rshift` arrays + output uniform type | `ConvLowering::LoweringINT8` |
| "rescale each Add input to the output scale" | `multipliers`/`rshifts` (8-bit multipliers) | `AddLowering::LoweringINT8`, `get_scale_and_shift_positive(·, 8)` |
| "apply sigmoid in the quantized domain" | `tpu.Lut` + 256-byte table weight | `create_lookup_table` |
| "dequantize the output" | `tpu.Cast` uniform → f32 | `cast_process` (function return type is f32) |
| "ReLU" | unsigned output type; `relu_limit = -1` | calibrated min 0 → u8; `relu_process` |

Add multipliers, from `scale_in / scale_out`:

```text
conv1 → add : (2/255) / (4/127) = 0.24902  → (m, r) = (1, 2)   = 0.25
conv2 → add : (3/127) / (4/127) = 0.75     → (m, r) = (3, 2)   = 0.75
```

The 8-bit multiplier limit makes the first factor slightly inexact (0.25 vs
0.24902); this is an inherent INT8 Add error and is visible in the
`validate_tpu_mlir` comparison, not in `validate_model`.

Sigmoid table (M2): `sigmoid` input is `add` (signed, scale 4/127), output is
u8 scale 1/255:

```text
table[q mod 256] = to_uint8( sigmoid(q · 4/127) · 255 , HALF_AWAY_FROM_ZERO ),  q ∈ [−128, 127]
table[255] (q = −1)  = round(sigmoid(−0.031496)·255) = round(125.49) = 125
table[0]   (q =  0)  = round(0.5·255)                = 128
```

Any change to the `add` or `sigmoid` threshold rewrites all 256 bytes.

#### 4.4 Saving (M3)

When `tpuc-opt` exits, the Deinit pass calls `module::saveWeight()`.  The
weight file is rewritten keeping only names that still have a `top.Weight`
in the IR:

```text
<prefix>_tpu_weights.npz
  conv1_filter_i8   int8   [8,2,1,1]    [[127,-51], ...]
  conv1_bias_int32  int32  [1,8,1,1]    [806, ...]
  conv2_filter_i8   int8   ...
  conv2_bias_int32  int32  ...
  sigmoid_table     uint8  [1,1,1,256]
```

Module state: `TPU_LOWERED`.  This file pair (`_tpu.mlir`,
`_tpu_weights.npz`) is the **numerical contract** of the model.

### Phase 5 — running `_tpu.mlir` on the CPU interpreter

`model_deploy.py` validates `_tpu.mlir` by running it through
`ModuleInterpreter` (`bindings/pymlir/host/ModuleInterpreter.cpp`).  Two facts
explain everything you see in the `_tpu_outputs.npz` dump:

1. **Every tensor is stored as `std::vector<float>`**, including int8 ones.
   An int8 activation is a float buffer holding integer values such as
   `41.0`.  Weights are loaded with `WeightOp::read_as_float()`.
2. After `invoke`, if `express_type` is true and the state is `TPU_LOWERED`,
   every uniform-quantized tensor is converted back in place with
   `(data − zp) · scale`.  `mlir_inference` passes `invoke(not out_fixed)`, so
   by default the dumped tensors are real-valued and comparable with the Top
   float reference.  `--out_fixed` shows the raw integers.

The run on our example, using the formulas of the `inference()` methods:

```text
R1  tpu.Cast   x_i8   = saturate(round(x · 1/0.031496))         = [41, −24]
                       (requant() in CastUtils.cpp: round(data·(1/scale)) + zp)
R2  conv1      acc    = 41·127 + (−24)(−51) + 806               = 7237
               y      = RightShiftRound(7237 · 4243973, 28, HALF_UP) = 114   → clamp [0,255]
               real   = 114 · 2/255 = 0.8941     (float reference 0.90)
    conv2      (same arithmetic)                                → say −40, real −0.9449
    add        lhs    = RightShiftRound(114 · 1, 2, HALF_UP)    = 29
               rhs    = RightShiftRound(−40 · 3, 2, HALF_UP)    = −30
               out    = saturate(29 + (−30))                    = −1,  real −0.0315
    sigmoid    out    = table[255]                              = 125
R3  tpu.Cast   o      = 125 · (1/255)                           = 0.4902   (float reference ≈ 0.4873)
```

These are the numbers `validate_tpu_mlir` compares against the Top float
outputs (`--tolerance`).  Every deviation from the float reference was decided
in phase 4.

### Phase 6 — `--strip-io-quant` and `--processor-tpu-optimize`

The second `tpuc-opt` command starts from `_tpu.mlir`.

- **Default:** the boundary Casts stay.  The BModel takes f32 and returns f32,
  and R1/R3 execute **on the TPU**.
- **`--quant_input` / `--quant_output`** (and the `_list`/`_int8`/`_bf16`
  variants): `StripIOQuant` deletes the boundary Casts and retypes the input
  and results.  R1/R3 move **to the host application**.  The model never sees
  float input; the application must quantize with the scale stored in the
  BModel (Phase 9).
- `processor-tpu-optimize` can merge `Conv2D(i32) → RequantInt` into one conv
  (`ConvMergeRequant`), or fuse a MatMul with its per-channel requant.  These
  rewrite *where* the requant instruction lives, never the numbers.

### Phase 7 — `--weight-reorder`: requant attributes become weight bytes (M4)

`WeightReorder/BM1684X/Conv2D.cpp` turns conv1's three pieces of quantization
data into one coefficient tensor, `conv1_merge`:

```text
requant words per output channel
  BM1684X                            : {m, −r, zp}                    3 × int32
  BM1688 / BM1690 family / CV184X /
  SG2380 / SGTPUV8 / BM1684X2 / SG2262: {m, ((−r) & 0xff) | (zp << 16)} 2 × int32

channel 0 on BM1684X:  { 4243973, −28, 0 }
channel 0 on BM1688 :  { 4243973, 0x000000E4 }            (−28 & 0xff = 0xE4, zp = 0)

row(oc) = [ requant words | bias_int32 (align 4 B) | filter int8, IC padded to IC_PARALLEL (align EU bytes) ]
```

Then the pass:

- removes `multiplier`, `rshift`, `out_f8_scales` attributes;
- sets `coeff_merged = true`;
- creates the weight `conv1_merge` and replaces the filter/bias operands.

After this point the requant parameters **exist only as bytes**.
`<prefix>_final.mlir` no longer shows them; look in `_tpu.mlir` to read them.
The values themselves (127, −51, 806, 4243973) are unchanged; only their
layout changes.

Add and Lut keep their attributes and table weights; the table is a plain
weight. Module state: `TPU_REORDERED`.

### Phase 8 — layer group and address assignment

No numbers change.  Both passes size buffers by storage type
(`BM168x::getFmtBytes`): the i8/u8 activations take 1 byte per element, the
int32 conv partial sum gets its own local-memory buffer, `conv1_merge` takes
its row size × channels.  Address assignment then gives every activation a
neuron address and every weight a coefficient address (state
`TPU_ADDRESSED`).  INT8 lets more layers fit into one local-memory group,
which is where much of the speed-up comes from.

### Phase 9 — `--codegen`: commands, coefficient blob, tensor metadata (M5)

Three independent outputs come from `BMCodegen` (`BM168xCodegen.cpp`) and the
per-op `codegen_*` methods.

**(a) Coefficient blob.**  `CreateCoeffMem` walks every `top.Weight`, reads its
bytes (`read_as_byte`) and `memcpy`s them into one buffer at
`address − coeff_addr`.  This blob — containing `conv1_merge`, `sigmoid_table`,
… — is stored in the BModel with a SHA-256.  Weight values are final here.

**(b) Per-op parameter structs → commands.**  `tpu::Conv2DOp::codegen_global_bm1684x`
fills `conv_global_spec_t`:

| Field | Value for conv1 | Meaning |
|---|---|---|
| input dtype | `DTYPE_INT8` | from `!quant.uniform<i8>` |
| output dtype | `DTYPE_UINT8` | from `!quant.uniform<u8>` |
| `ipad_value` | `0` | input zero point; padding uses it |
| `if_relu` | 1 | unsigned output |
| `merge_coeff` | 2 | requant + bias + weight merged |
| `round_mode` | from attribute | |

`BM168x::call_global_func` calls `backend_api_conv_global` in
`libbackend_<chip>.so`, which writes TIU/GDMA descriptors.  The same happens
for the boundary Casts (`backend_api_requant_float_global`,
`backend_api_dequant_float_global`), the Add (`backend_api_bcbinary_global`
with per-input multipliers and shifts) and the Lut (`backend_api_lut`).

**(c) Tensor metadata.**  For every input/output tensor, codegen writes
`scale` and `zero_point` into the BModel tensor table.  A subtle rule:

```text
for a uniform-quantized tensor:
    scale = qtype.scale
    if the tensor is produced by top.Input:  scale = 1 / qtype.scale     <-- inputs store the reciprocal
    zero_point = qtype.zero_point
```

So a stripped int8 input with scale `0.031496` is stored with scale `31.75`,
and an int8 output is stored with its real scale.  The host helpers rely on
this asymmetry (Phase 11).

### Phase 10 — inside the chip

For our example the command stream executes this sequence (the backend is
closed source; the sequence below is what the spec structs and the
interpreter fix):

```text
GDMA  load x (f32) from DDR
TIU   Cast: x · 31.75 → round → saturate int8                        x_i8 = [41, −24]          (R1)
TIU   Conv: int8 × int8, int32 accumulate, + bias_int32              acc = 7237
TIU         per-channel (acc · 4243973) >> 28, + zp, clamp u8        114                        (R2)
TIU   Add : (114·1)>>2 + (−40·3)>>2, saturate int8                   −1                         (R2)
TIU   Lut : table[−1]                                                125
TIU   Cast: (125 − 0) · 0.0039216 → f32                              0.4902                     (R3)
GDMA  store output (f32) to DDR
```

Inside a layer group, the loads/stores of intermediate tensors disappear (they
stay in local memory), but the arithmetic is the same.  `validate_model`
compares these results with the interpreter's from Phase 5; they should be
nearly bit-exact (`--correctness 0.99,0.90`).  A mismatch here is a
codegen/backend problem, not a calibration problem.

### Phase 11 — host side (`model_runner.py` on a BModel)

`model_runner.py` (`model_inference`) feeds the BModel through `pyruntime`:

- **Input:** if the BModel input is int8/uint8 (stripped Cast), the host calls
  `lowering(input, pdtype, pzero_point=i.qzero_point, pscale=i.qscale)`
  (`python/utils/lowering.py`), which computes
  `clip(round_away_from_zero(input · pscale + zp))`.  Because codegen stored
  the **reciprocal** scale for inputs, multiplying is the correct quantization.
  If the input is f32, the data is passed as-is and the on-chip Cast does R1.
- **Output:** for int8/uint8 outputs (unless `--out_fixed`) it computes
  `(data − qzero_point) · qscale`, the dequantization R3.

Your own application must do the same with the BModel tensor info when it
compiles with `--quant_input` / `--quant_output`.

---

## 5. One table: what the quantization looks like in each artifact

| Phase / artifact | Activation quantization appears as | Weight quantization appears as | Values changed here |
|---|---|---|---|
| `<m>.mlir` | nothing | float `top.Weight` | — |
| calibration table | `name threshold min max` text | optional `#weight_scale` (QAT) | float NPZ (M0, optional) |
| after import-calibration | `!quant.calibrated<f32<min:max>>` | unchanged | — |
| `_tpu.mlir` | `!quant.uniform<i8/u8:f32, scale:zp>`, `tpu.Cast`, `multiplier`/`rshift(s)` attrs, `tpu.Lut`, `tpu.RequantInt*` | `*_filter_i8`, `*_bias_int32`, `*_table`, `*_i8` weights | weights (M1, M2, M3) |
| `_tpu_outputs.npz` (interpreter) | dequantized floats (raw ints with `--out_fixed`) | — | activations simulated (R1–R3 on CPU) |
| after weight-reorder | uniform types, Casts; conv requant attrs gone | `*_merge` = `[requant words | bias | filter]` | layout only (M4) |
| `_final.mlir` | as above + addresses, layer groups | merged weights with addresses | — |
| BModel | tensor `scale`/`zero_point` (reciprocal for inputs); TIU commands | coefficient blob | placement only (M5) |
| runtime | real integer activations in DDR / local memory | read from blob | activations (R1–R3 on TPU) |

---

## 6. How the pieces depend on each other

```text
calibration thresholds ──► calibrated types ──► uniform scales (getScaleAndZeroPoint)
                                                    │
               ┌────────────────────────────────────┼──────────────────────────────┐
               ▼                                    ▼                              ▼
     conv/matmul multiplier,rshift        Cast quant/dequant scale            LUT table bytes
     bias_int32 (uses input scale)        (in the type, not an attribute)     (uses in + out scale)
               │                                    │                              │
               ▼                                    ▼                              ▼
     weight-reorder requant words         codegen requant/dequant spec        plain weight
     (attributes deleted)                 + BModel tensor scale               │
               └──────────────────────┬─────────────┴──────────────────────────────┘
                                      ▼
                        coefficient blob + TIU/GDMA commands
```

Practical consequences:

1. **One threshold feeds several byte locations.**  Changing the `add`
   threshold changes the Add multipliers, the sigmoid table, and (through
   `calibration_process`) possibly the producers' requant words.  You cannot
   patch one of them in the BModel by hand.
2. **The input scale enters the bias.**  `bias_int32 = b / (scale_w · sx)`.
   A wrong input threshold corrupts the bias, not only the activations; this
   is why `LoweringINT8` guards `sx ≤ 1e-5`.
3. **A ReLU is a type, not an op.**  Calibrated min 0 → u8 output →
   `if_relu = 1` in codegen.  If a graph rewrite loses the calibrated min,
   ReLU behavior changes.
4. **Float islands cost two Casts.**  An op with no INT8 kernel (LayerNorm,
   some Softmax) gets a dequantize Cast before and a quantize Cast after.  Each
   is a real runtime conversion (R3 then R1) inside the model.
5. **`_tpu.mlir` is the last readable form.**  After weight reorder the
   requant parameters are bytes; after codegen the instructions are opaque
   backend descriptors.  Debug quantization on `_tpu.mlir`; debug codegen by
   comparing the BModel with the interpreter.

---

## 7. How to observe each phase yourself

Run inside `./tmp`.

```bash
# Phase 0/1
model_transform.py --model_name m --model_def m.onnx --input_shapes [[1,2,1,1]] \
    --test_input in.npz --test_result m_top_outputs.npz --mlir m.mlir
run_calibration.py m.mlir --dataset ./data --input_num 100 -o m_cali_table

# Phase 2-4: stop after lowering and inspect types/attributes/weights
model_deploy.py --mlir m.mlir --quantize INT8 --calibration_table m_cali_table \
    --processor bm1684x --test_input m_in_f32.npz --test_reference m_top_outputs.npz \
    --compare_all --debug --model m_int8.bmodel
grep -n "tpu.Conv2D\|tpu.Cast\|tpu.Lut\|multiplier" m_int8_bm1684x_tpu.mlir   # prefix depends on deploy options
python -c "import numpy as np; w=np.load('m_int8_bm1684x_tpu_weights.npz'); print({k:(v.dtype,v.shape) for k,v in w.items()})"

# Phase 5: raw integers instead of dequantized values
model_runner.py --input m_in_f32.npz --model m_int8_bm1684x_tpu.mlir --output raw.npz --out_fixed --dump_all_tensors

# Phase 7-9: merged weights and final layout
grep -n "_merge\|coeff_merged" m_int8_bm1684x_final.mlir
model_tool --info m_int8.bmodel        # tensor dtypes, scales (input scale is the reciprocal)
```

The exact output file prefixes depend on `--model`, `--quantize` and
`--processor`; `--debug` keeps the intermediate files.

---

## 8. Reading order for this volume

```text
lib/Conversion/TopToTpu/BM1684X/Conv.cpp      (LoweringINT8: weight values change)
lib/Dialect/Top/Interfaces/Weight.cpp         (WeightOp::create / read / read_as_float)
lib/Support/LutFunc.cpp                       (create_lookup_table)
lib/Conversion/TopToTpu/TopToTpuPass.cpp      (cast_process, do_cast: Cast insertion and naming)
lib/Support/CastUtils.cpp                     (requant / dequant used by Cast inference)
lib/Dialect/Tpu/Interfaces/Common/{Cast,Conv2D,Add,Lut}.cpp   (inference = numerical contract)
bindings/pymlir/host/ModuleInterpreter.cpp    (float storage, express_type dequantization)
lib/Support/Module.cpp                        (saveWeight)
lib/Dialect/Tpu/Transforms/StripIOQuant.cpp
lib/Dialect/Tpu/Transforms/WeightReorder/BM1684X/Conv2D.cpp   (requant words, _merge)
lib/Dialect/Tpu/Transforms/Codegen/BM168xCodegen.cpp          (CreateCoeffMem, tensor scale)
python/tools/model_runner.py, python/utils/lowering.py        (host quantize/dequantize)
```
