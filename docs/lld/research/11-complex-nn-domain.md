# Stage 11 — Complex Neural Network: Domain Research

**Date:** 2026-04-24
**Target plan:** `docs/plans/active/11-complex-nn.md` (not yet created)
**Intent source:** `docs/intent/11-complex-nn.md`
**Baseline SHA:** `6ad2d7a` (branch `stage-11-complex-nn`)
**Torch version in pyproject.toml:** `>=2.7,<3` (cu128 on Linux, PyPI on macOS/Windows)

---

## Section 1 — Temporal architectures for hourly load forecasting: state of practice (2020–2026)

### Architecture family comparison table

| Family | Representative papers | Approx. parameter count (typical small config) | Training time order-of-magnitude (CPU / small GPU) | Published accuracy vs linear baseline on hourly / sub-daily data | Fits `fit(features, target)` protocol? |
|---|---|---|---|---|---|
| **Dilated causal CNN (TCN)** | van den Oord et al. 2016 (WaveNet, arXiv:1609.03499); Bai et al. 2018 (arXiv:1803.01271) | 50k–500k for 6–8 dilated layers × 64 channels | Minutes on CPU for ~8760 rows / seconds on GPU | Bai et al.: consistently beats LSTM on sequence benchmarks; load forecasting studies report MAPE 2–3% (vs ~3.4% for classical baselines) | Yes — forward pass is a standard `nn.Module`; no decoder or autoregressive inference step required |
| **Informer** | Zhou et al. AAAI 2021 (arXiv:2012.07436) | 10M–50M | 30–120 min CPU / 5–20 min GPU for long-horizon tasks | Designed for very long sequences (1000+ steps); at hourly demand horizon (24 h ahead) it is over-engineered; not benchmarked head-to-head against linear models on short horizons | Requires encoder–decoder setup with decoder start token; does not map cleanly to flat `fit/predict` |
| **Autoformer** | Wu et al. NeurIPS 2021 (arXiv:2106.13008) | 10M–50M | Similar to Informer | 38% relative improvement on 6 long-horizon benchmarks vs Informer; omits positional encoding by design; learns 24 h and 168 h lags automatically from auto-correlation | Encoder–decoder; same protocol mismatch as Informer |
| **DLinear / NLinear** | Zeng et al. AAAI 2023 (arXiv:2205.13504) | <10k | Seconds on CPU | DLinear beats all prior Transformer-based LTSF models "in almost all cases"; NLinear adjusts for distribution shift; both decompose or subtract before a single linear projection | Yes — a single `nn.Linear` per step; trivially protocol-compatible |
| **PatchTST** | Nie et al. ICLR 2023 (arXiv:2211.14730) | ~1M–5M for d_model=128, 3 layers | 10–60 min CPU / 2–10 min GPU | 21% MSE reduction vs best Transformer baseline; outperforms DLinear significantly on Electricity dataset (MSE 0.131 on Electricity with context=512); patch_len=16, stride=8, d_model=128, heads=16, 3 layers | Channel-independent: each variate is an independent univariate series; maps to protocol with a wrapper |
| **iTransformer** | Liu et al. ICLR 2024 Spotlight (arXiv:2310.06625) | ~1M–5M for D=256, L=2–4 | 10–60 min CPU / 2–10 min GPU | SOTA on multivariate benchmarks including ECL (Electricity Consuming Load): MSE=0.178, MAE=0.270 with lookback T=96; designed for multivariate cross-variate correlation | Inverts the token axis (each variate = one token); works well for many-variate datasets, less natural for univariate or few-variate demand |
| **TSMixer** | Chen et al. (Google) 2023 (arXiv:2303.06053); KDD 2023 lightweight variant (arXiv:2306.09364) | ~100k–1M | 5–30 min CPU / 1–5 min GPU | Outperforms DLinear and PatchTST on ETTh1/ETTh2/Electricity by 1–2% accuracy at 2–3x lower memory and runtime vs PatchTST; KDD variant specifically designed for multivariate with auxiliary info | Yes — all-MLP, no special decoder; straightforward protocol fit |

**2025–2026 state of the debate.** The "Are Transformers Effective?" (Zeng et al. 2023) challenge showed DLinear beats complex Transformers on standard benchmarks. PatchTST (Nie et al. 2023) partially answered it by demonstrating a patch-based Transformer does beat DLinear on large datasets (Electricity 321-variate). iTransformer (Liu et al. 2024, ICLR Spotlight) showed a further gain specifically for multivariate correlation. The TFB benchmark (PVLDB 2024 Best Paper Nomination, github.com/decisionintelligence/TFB) attempts comprehensive fair comparison but no single model dominates all regimes. For small univariate/few-variate short-horizon tasks of the scale this project uses (~8760 rows, single target), the DLinear/NLinear result is a genuine warning: a linear decomposition baseline may be hard to beat and should be included in the ablation.

Pedagogical risk flag: If Stage 11's temporal model fails to beat DLinear on the held-out period, the ablation table still tells a coherent story — it becomes an honest demonstration that temporal complexity does not automatically win, which is more instructive than a model that wins for unclear reasons.

---

## Section 2 — TCN vs small Transformer: decision-relevant trade-offs

### Trade-off table

| Dimension | TCN (dilated causal 1D-conv) | Small Transformer (encoder-only, d_model 64–128) |
|---|---|---|
| **Receptive field** | Determined by architecture: `1 + 2*(kernel_size-1)*(2^n_layers - 1)`. For kernel=3, 8 layers: 1 + 2*2*255 = 1021 steps. Covers 168 h weekly cycle easily with 6–8 layers. | Determined by sequence length fed as input. With seq_len=168 all positions attend to all others (O(L²) attention). No architectural constraint on receptive field; the constraint is the input window size. |
| **Training time on ~8760 rows, CPU** | Fast. Conv1d operations are cache-friendly and parallelise well on CPU. Expected order: 10–60 s per epoch for 8-layer × 64-channel, batch_size=64, seq_len=168. 30–50 epochs = 5–50 min total. | Slower. Self-attention is O(L²·d_model) per head. With L=168, d_model=128, 2–4 layers: expect 2–10x slower than TCN per epoch on CPU. 30–50 epochs = 20–200 min total on CPU. This is the dominant constraint for a 60-min demo slot. |
| **Parameter efficiency for ~8760 rows** | High. A 6-layer × 64-channel TCN has ~150k–300k parameters. Receptive field grows exponentially with layers. Low risk of overfitting on ~7900-row training set (post-val-split). | Moderate. A 2-layer d_model=64 Transformer has ~300k–600k parameters. Overfitting risk on small data is real (Transformers are known to underperform on small datasets: "when data is subsampled to 1%, bigger [Transformer] models are worse"). |
| **Ease of explanation in live demo** | High. Dilated convolution is visually intuitive: stack of sliding filters, each seeing a wider window. WaveNet diagram is a standard teaching tool. | High but different. Attention visualisation is the compelling demo moment. However, attention patterns on 168-step sequences can be noisy and hard to interpret without guided narration. |
| **Reproducibility on CPU** | Excellent. `nn.Conv1d` is deterministic under `torch.use_deterministic_algorithms(True)` on CPU — no known non-deterministic kernels for standard 1D convolution. | Requires care. `nn.MultiheadAttention` uses `scaled_dot_product_attention` internally (introduced PyTorch 2.0); on CUDA this selects backends non-deterministically. On CPU the math backend is deterministic but the `is_causal=True` parameter in `MultiheadAttention.forward` was documented as being silently ignored when `need_weights=True` in PyTorch 2.0 (GitHub issue #99282, opened April 2023 against 2.0.0 — fix status unclear in retrieved content). Always pass an explicit `attn_mask` rather than relying on `is_causal` for correctness. |
| **Attention weight visualisation** | N/A — convolution weights are filters, not sequence-level attention. Saliency maps are possible but require additional machinery. | Available. TFT-style per-timestep attention weights reveal which past hours the model uses. Published results show attention peaks at 24 h and 168 h intervals matching daily and weekly periods — genuinely illuminating for electricity demand. But this works best when `need_weights=True` is correctly implemented (see above bug). |
| **Causal structure enforcement** | Built in by construction: the padding scheme (`padding = (kernel_size-1)*dilation`, applied left-only) ensures no future leakage without any additional mask. This is the TCN's defining property (Bai et al. 2018). | Must be enforced explicitly via `attn_mask`. The standard pattern is `torch.ones(L, L).triu(1) * float('-inf')`. Omitting this mask is the single most common reproducibility/correctness bug in Transformer time series reproductions. |
| **Interface fit with Stage 4 protocol** | Natural. TCN is a `nn.Module` with a `forward(x)` taking `(batch, features, seq_len)`. The sequence dataset wrapper lives inside `fit`; the protocol surface is unchanged. | Natural. Same `nn.Module` shape. Both choices require a `SequenceDataset` helper that converts the flat `pd.DataFrame` from the harness into `(batch, seq_len, features)` tensors. |

---

## Section 3 — Sequence length and receptive field

### What the literature says

**168 hours as the natural upper bound.** The weekly cycle in electricity demand (168 h) is the established natural period. Autoformer (Wu et al. NeurIPS 2021) reports that on Traffic dataset (hourly), its auto-correlation mechanism learns lag intervals of exactly 24 h and 168 h, confirming these as the dominant periodicities. Studies on day-ahead forecasting routinely evaluate prediction lengths of 24, 48, 96, and 168 hours and use look-back windows matching these periods.

**The 168 h look-back as a seasonal naive baseline.** A 2020 study on day-ahead load forecasting (arXiv:2008.07025) uses `sNaive(168)` — the observed value from one week ago — as a seasonal naive benchmark. This establishes that a model conditioning on the last 168 hours "deserves" to beat a trivially reproducible 168-step seasonal pattern; if it does not, the temporal architecture adds no value.

**UniLF (2025, Scientific Reports)** evaluated input lengths of 48, 96, 168, and 336 hours on a short-term load forecasting task and found the best prediction accuracy at `in_len=168`. This is direct confirmation that 168 h is a practically defensible choice for GB hourly demand.

**PatchTST** (Nie et al. ICLR 2023) evaluated context lengths of 336 steps as the default for ETT/Electricity benchmarks. The paper notes "models that use a longer look-back window perform better" but with diminishing returns beyond 336. For the `~8760 h` annual dataset in this project, a 336-step window would reduce the number of training sequences by roughly 2× compared to a 168-step window, which may harm training on small datasets.

**Summary for the plan author.** Three defensible positions from the literature:

1. **168 h** — matches the weekly cycle exactly; aligns with sNaive(168) semantic anchor; confirmed by UniLF 2025 as best-accuracy input length for STLF.
2. **336 h** — used by PatchTST defaults; captures two full weekly cycles; but halves the number of non-overlapping training windows from ~52 to ~26, which is genuinely risky on ~7900 training rows.
3. **96 h** — common in long-horizon benchmarks as a round number; misses the full weekly cycle.

Pedagogical risk: if the plan author picks 168, they can defend it from the literature (weekly cycle anchor + UniLF). If they pick 336, they should halve it for CPU budget reasons. A configurable `seq_len` with default=168 is defensible.

---

## Section 4 — Exogenous feature handling

### The two canonical patterns

**Pattern A — In-sequence.** Weather features (temperature, wind, solar irradiance, etc.) are concatenated onto each time step of the input tensor: shape `(batch, seq_len, n_load_features + n_weather_features)`. Simpler to implement. Assumes the model learns the relationship between past weather and past demand implicitly, then extrapolates to future demand. The model sees the same weather-feature structure at every step.

**Pattern B — Side channel (forecast-known).** The sequence encoder sees only historical load (and optionally historical weather). The weather forecast for the target horizon (which is known at prediction time for day-ahead forecasting) is fed as additional context at the final layer or decoder. This is the design philosophy in the Temporal Fusion Transformer (Lim et al., *International Journal of Forecasting* 2021, arXiv:1912.09363): "known future inputs go into the decoder" while "past inputs are fed into the encoder". TFT explicitly distinguishes three input types: static covariates, known-future inputs (calendar, forecasted weather), and observed-past inputs (historical demand).

**Is one published as better for electricity load?** No head-to-head published result was found for GB national demand specifically comparing the two patterns. The TFT paper (Lim et al. 2021) shows strong results on electricity datasets using the side-channel architecture for known-future inputs, but the comparison is against LSTM and DeepAR, not against in-sequence concatenation of the same features.

**Practical implication for Stage 11.** For a day-ahead forecaster where the feature set (as established in Stage 5) already contains the target-hour weather forecast (available at prediction time), Pattern A and Pattern B are informationally equivalent at prediction time — the model receives the same features either way. The architectural difference matters for the training signal: in Pattern A, the model trains to use weather features as part of the temporal context; in Pattern B, weather forecasts are anchored as a global condition rather than a time-varying sequence component. Pattern A is simpler to implement within the existing flat-features-in → predictions-out protocol. Pattern B better reflects physical causality but requires a two-branch model architecture that complicates the `fit(features, target)` interface.

**Recommendation for the plan author (lay out, not prescribe).** Pattern A is the lower-complexity default. If the ablation shows the temporal model underperforms the MLP, Pattern B is the natural next diagnostic.

---

## Section 5 — Positional encoding for short sequences

### What each major architecture chose and why

**Autoformer (Wu et al. NeurIPS 2021):** Omits positional encoding entirely. The paper states: "Since the series-wise connection will inherently keep the sequential information, Autoformer does not need the position embedding, which is different from Transformers." The auto-correlation mechanism encodes temporal relationships implicitly.

**iTransformer (Liu et al. ICLR 2024):** Omits positional encoding explicitly. The paper states: "the position embedding in the vanilla Transformer is no longer needed here" because "the order of sequence is implicitly stored in the neuron permutation of the feed-forward network." This is the inverted-axis design consequence: the token axis is now variates, not time steps, so no time-step positional encoding is needed.

**PatchTST (Nie et al. ICLR 2023):** Uses learnable positional embeddings. Each patch position is assigned a learned embedding vector (standard BERT-style). The HuggingFace `PatchTSTConfig` defaults to `d_model=128`, `num_attention_heads=16`, `num_hidden_layers=3`. The patch-level positional encoding over 64 patches (context 512 / patch_stride 8) is a much shorter sequence than raw time steps, making learned embeddings practical and avoiding the length-extrapolation problem.

**Vanilla Transformer (Vaswani et al. 2017):** Sinusoidal. Fixed, no training required, generalises to longer sequences by construction. The positional encoding survey (arXiv:2502.12370, 2025) notes sinusoidal encodings "struggle with varying sequence lengths and dynamic temporal relationships" but are standard for short fixed-length sequences.

**Consensus for ≤168-step sequences.** For a fixed sequence length of 168 steps, all three options (sinusoidal, learned, omitted) are defensible:

- **Sinusoidal:** No parameters to fit; standard choice from original Transformer paper; appropriate when the sequence length is fixed and known a priori. For electricity demand there is a natural interpretation: sinusoidal frequencies corresponding to periods of 24 h and 168 h have direct physical meaning.
- **Learned:** Adds `seq_len × d_model` parameters (168 × 64 = ~10k). On small training data this risks overfitting positional assignments. Avoid unless training set is large (>50k sequences).
- **Omitted:** Appropriate if the architecture implicitly encodes order (as in iTransformer) or if a causal mask already enforces temporal direction (as in a causal Transformer encoder). For a non-causal encoder (where all positions attend to all past positions), omitting positional encoding removes the model's ability to distinguish "2 hours ago" from "10 hours ago" — this is a correctness risk.

**Defensible default for Stage 11:** Sinusoidal, fixed. It is the cheapest option (zero extra parameters), it has direct physical interpretability for hourly data, it does not add a parameter-fitting burden on a small training set, and it is what a meetup audience will recognise from the original Transformer paper. This can be stated confidently with the citations above.

---

## Section 6 — Training loop gotchas specific to temporal models

### Gradient accumulation / batch size

For a training set of ~7900 rows (after 10 % val split from 8760 h) and sequences of length 168 with stride 1, the number of non-overlapping training sequences is ~47 and with stride=1 (overlapping) it is ~7732. At `batch_size=64`, an epoch is ~120 batches (overlapping) or under 1 (non-overlapping). Overlapping sequences are the standard approach for temporal models; non-overlapping is for computational efficiency on very large datasets.

Gradient accumulation is unnecessary at this scale: the dataset fits in CPU RAM and a batch_size of 32–128 is tractable. The Stage 10 harness convention (`batch_size=256` for tabular) should be reduced for sequence training — sequences of 168 × n_features are larger tensors than single feature rows. Recommend `batch_size=32` or `64` as the default for sequence training; expose as config.

### Learning rate schedules

Standard for this class of model:

- **Flat Adam (lr=1e-3):** Works for TCN, validated across sequence benchmarks (Bai et al. 2018 used Adam lr=2e-3). Simplest and consistent with Stage 10.
- **Cosine annealing with warm restarts (`torch.optim.lr_scheduler.CosineAnnealingLR`):** Standard for Transformers on time series. iTransformer (Liu et al. 2024) uses Adam with learning rates in {1e-3, 5e-4, 1e-4} and early stopping with patience=5.
- **Warmup + cosine:** Standard for NLP Transformers; generally unnecessary for small time series Transformers at this scale (only ~50 epochs), but described as best practice for models with >3 layers.
- **ReduceLROnPlateau:** Adaptive; monitors val loss. Used in many load forecasting papers as the safe default. Works with the existing early-stopping infrastructure.

**Practical recommendation for the plan author:** Use `ReduceLROnPlateau` or flat Adam with early stopping (patience=5–10) for the TCN path. Use cosine annealing (T_max = max_epochs) for the Transformer path. Both are in `torch.optim.lr_scheduler` and require no new dependencies.

### Early stopping — temporal-specific concerns

The Stage 10 research (R4) documented Prechelt (1998) patience conventions. For temporal models specifically:

- The val tail must be a contiguous recent window (not shuffled); this is already enforced by Stage 10's internal val split (val fraction of the last 10% of time-ordered data). Shuffled validation would leak future information into the training set distribution.
- Sequence overlap between the end of the training window and the start of the validation window is a subtle leakage risk: if a training sequence ends at step T and a validation sequence starts at step T-10, the model has "seen" some of the validation context during training. Standard practice is to offset the train/val boundary by `seq_len` steps to prevent overlap.

### Attention masking for causal forecasting

**The most common implementation bug in time series Transformer reproductions.** A Transformer encoder without a causal mask attends to future time steps during training if the sequence includes the target time step. For a day-ahead forecasting model that conditions on the last 168 hours and predicts the 169th, the correct setup is:

- **Encoder receives past 168 steps.** No causal mask is needed if the encoder input genuinely contains only past data (i.e., the target hour is never in the input window). This is the correct setup for a point forecast Transformer.
- **Decoder or direct prediction head then maps the encoder output to the forecast.** With this architecture, no mask is needed in the encoder — it is attending over historical data only.

If the architecture instead uses a sequence-to-sequence formulation where the input sequence contains the target step, a causal mask is essential. The safe pattern in PyTorch:

```python
mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
attn_output = nn.MultiheadAttention(...)
output, weights = attn_output(q, k, v, attn_mask=mask.float() * float('-inf'))
```

**Do not use `is_causal=True` in `nn.MultiheadAttention.forward` as the sole mechanism.** PyTorch issue #99282 (opened April 2023, against PyTorch 2.0.0) documents that `is_causal=True` is silently ignored when `need_weights=True`. Since attention-weight visualisation (one of the Stage 11 "sometimes illuminating" features) requires `need_weights=True`, this bug directly affects the demo workflow. Pass `attn_mask` explicitly.

### Regularisation: dropout and normalisation placement

**LayerNorm vs BatchNorm for sequence models.** LayerNorm is the universal choice for Transformers and TCNs applied to time series. BatchNorm on time series data introduces information leakage because batch statistics include future time steps when batch includes temporally mixed samples, and it performs poorly at inference time for individual sequences with distributions far from the training mean. The practical consensus (supported by architecture choices in PatchTST, iTransformer, Autoformer, and TCN implementations) is: use `nn.LayerNorm` throughout.

For TCN with weight normalisation (`torch.nn.utils.weight_norm`): Bai et al. 2018 used weight normalisation on TCN kernels as the default. The reference `locuslab/TCN` implementation applies `weight_norm` to every Conv1d layer. Weight norm is applied to the kernel weights only and does not involve batch statistics, so it is safe for time series.

**Pre-norm vs post-norm.** The original Transformer (Vaswani 2017) uses post-norm (LayerNorm after residual addition). Pre-norm (LayerNorm before sublayer) is empirically more stable for small models and is the default in most modern implementations including PatchTST. For Stage 11's small (2–4 layer) Transformer, pre-norm is recommended and requires no additional tuning.

**Dropout.** Standard placement for a Transformer: after attention output and after FFN output, before residual addition. For TCN: after each residual block activation. Typical dropout rate for small models on small data: 0.1–0.2. Higher values (0.3+) risk under-fitting on ~7900 training sequences.

---

## Section 7 — PyTorch API surface for Stage 11

### APIs beyond Stage 10's set

| API | First available | Required minimum for Stage 11 | Reproducibility / gotcha |
|---|---|---|---|
| `nn.Conv1d` | PyTorch 1.0 | Already available in `>=2.7` (pyproject.toml pin) | CPU: fully deterministic under `use_deterministic_algorithms(True)`. No known issues. Weight norm via `torch.nn.utils.weight_norm(layer)` is safe on CPU. |
| `nn.ConvTranspose1d` | PyTorch 1.0 | If needed for decoder path — not required for a point forecast TCN | Not needed for Stage 11 scope. |
| `nn.LayerNorm` | PyTorch 1.0 | Available | CPU: deterministic. |
| `nn.TransformerEncoder` / `nn.TransformerEncoderLayer` | PyTorch 1.1 | Available | `is_causal` parameter on `TransformerEncoderLayer.forward` introduced in PyTorch 2.0 but has documented interaction issues (see Section 6). Use explicit `src_mask` instead. |
| `nn.MultiheadAttention` | PyTorch 1.1 | Available | `is_causal=True` silently ignored when `need_weights=True` in PyTorch 2.0 (issue #99282). The fix status was not confirmed in retrieved content. Mitigation: pass `attn_mask` explicitly. |
| `torch.nn.functional.scaled_dot_product_attention` (SDPA) | **PyTorch 2.0** (introduced as beta) | Available in `>=2.7` | **CUDA-specific non-determinism:** "may select a nondeterministic algorithm" on CUDA. Mitigated by `torch.backends.cudnn.deterministic = True`. On CPU: deterministic. `nn.MultiheadAttention` uses SDPA internally in PyTorch 2.0+. Use `torch.nn.attention.sdpa_kernel()` context manager to pin a specific implementation if strict reproducibility across hardware is required. |
| `torch.nn.utils.weight_norm` | PyTorch 1.0 | Available | CPU: deterministic. |
| `torch.nn.functional.pad` (for causal padding in TCN) | PyTorch 1.0 | Available | Deterministic on CPU. Pattern: `F.pad(x, (padding, 0))` followed by `nn.Conv1d(padding=0)` — removes right-side leak. |

**Key version note.** The project pins `torch>=2.7,<3`. All APIs listed above were introduced in PyTorch 1.0–2.0 and are available throughout the 2.7 line. No new API acquisition is needed beyond what exists in the current lock file.

**CUBLAS_WORKSPACE_CONFIG.** The Stage 10 research (R1) noted that for CUDA training under `use_deterministic_algorithms(True)`, `CUBLAS_WORKSPACE_CONFIG=:4096:8` must be set. This applies to Stage 11's GPU-optional path. The Dockerfile should set this environment variable if CUDA training is desired with deterministic mode.

---

## Section 8 — CPU-feasibility sanity check

No single paper provides exact CPU timing for the specific configurations described. The following back-of-envelope is grounded in the available evidence.

### TCN: 8-layer × 64-channel, kernel=3, batch_size=64, seq_len=168, 30–50 epochs

**Training data:** ~7900 usable rows → ~7900 overlapping sequences → 123 batches of 64.

**Per-batch FLOPs estimate.** A TCN with 8 layers, kernel=3, 64 channels, input_dim ≈ 50 features:
- Residual block per layer ≈ 2 × Conv1d(64, 64, kernel=3) ≈ 2 × 64 × 64 × 3 × seq_len ≈ 2.4M FLOPs per sample.
- 8 layers × 64 samples per batch ≈ ~1.2B FLOPs per batch.

**CPU throughput.** A modern laptop CPU (Intel Core i7/i9 or AMD Ryzen 7/9) delivers approximately 50–200 GFLOPS for dense matrix operations via PyTorch's MKL/BLAS backend. At 100 GFLOPS and ~1.2B FLOPs per batch: ~12 ms per batch. At 123 batches per epoch: ~1.5 s per epoch. For 50 epochs: ~75 s ≈ **1–2 minutes total**.

This is well within the pedagogical budget. A 30-epoch training run with live loss plot should complete in under 90 s on a 2024-era laptop.

### Small Transformer: depth=2, d_model=128, heads=4, seq_len=168, batch_size=32, 30–50 epochs

**Self-attention FLOPs.** Multi-head attention per layer ≈ `4 × batch × heads × seq_len² × (d_model/heads)` ≈ `4 × 32 × 4 × 168² × 32` ≈ 1.16B FLOPs. With 2 layers and FFN (≈ equal cost): ~5B FLOPs per batch. At ~200 GFLOPS (CPU): ~25 ms per batch.

**Batches per epoch.** 7900 / 32 = 246 batches. Per epoch: ~6 s. For 50 epochs: ~300 s ≈ **5 minutes total**.

**Caution.** These are idealised estimates. PyTorch's overhead (Python dispatch, memory allocation, gradient bookkeeping) typically adds 2–5× on small batch sizes on CPU. Realistic estimate: **TCN 3–8 minutes; Transformer 15–40 minutes** for 50 epochs on a 2024-era laptop. The Transformer estimate is borderline for a 60-minute demo slot that also includes notebook setup, data loading, and evaluation.

**If these numbers do not fit the pedagogical budget, the architecture choice is forced toward the cheaper option (TCN).** The plan author should benchmark on the actual hardware before committing to a Transformer default. A `%%time` cell in the notebook will surface the truth immediately.

**A configuration that keeps the Transformer within budget:** depth=2, d_model=64, heads=4, seq_len=96 (dropping to 96 from 168). This roughly halves attention cost relative to seq_len=168 and is still defensible from the literature (96 h = 4 days).

---

## Section 9 — The ablation table: prior art

### Have published papers produced a pedagogical ablation of this shape?

The search found ablation tables in load forecasting literature, but they are component-ablations (remove module X, show metric drop) rather than cross-model ablations (one row per model family, same held-out period, same metrics). Examples found:

- Frontiers Energy Research (2024): ablation of LSTM + TCN + attention combinations showing MAPE/RMSE per configuration. Published as Table 6 and Figure 14. Pedagogically similar to what Stage 11 needs but for a different dataset and not progressive across model stages.
- TFTformer (ScienceDirect 2025): "comprehensive ablation study by computing MSE and MAPE for every possible combination of elements across seven datasets." Progressive but structural (one model family with components enabled/disabled), not cross-model.
- Deep learning survey on STLF (arXiv:2408.16202, 2024): comprehensive comparison tables but each paper is evaluated on its own dataset and period — not a unified held-out split.

**Conclusion.** No published paper was found that produces a pedagogically-shaped cross-model ablation table (naïve → linear → MLP → TCN/Transformer, same held-out GB demand split, same metrics) matching the Stage 11 intent. The plan author will design it from scratch. The closest structural precedent is the GEFCom2014 load forecasting competition results table, which has the same shape (one row per method, same test split, same metric) but is a competition leaderboard rather than a pedagogical artefact.

**Design suggestion for the ablation.** Columns: Model name | MAE (MW) | RMSE (MW) | MAPE (%) | Training time (s, CPU) | Parameters. One row per stage: naïve, linear, SARIMAX, scipy parametric, simple MLP, temporal model (Stage 11). The "Training time" column is the most illuminating comparison: it quantifies the cost-accuracy trade-off that underpins the pedagogical arc.

---

## Canonical sources

| Source | Summary |
|---|---|
| [van den Oord et al. 2016 — WaveNet (arXiv:1609.03499)](https://arxiv.org/abs/1609.03499) | Original dilated causal convolution for sequences; dilation doubling schedule 1,2,4,...,512; receptive field grows exponentially with depth |
| [Bai, Kolter, Koltun 2018 — TCN (arXiv:1803.01271)](https://arxiv.org/abs/1803.01271) | Empirical demonstration that TCN outperforms LSTM across sequence benchmarks; Adam lr=2e-3; weight normalisation on Conv1d |
| [Zhou et al. AAAI 2021 — Informer (arXiv:2012.07436)](https://arxiv.org/abs/2012.07436) | ProbSparse attention for long sequences; Best Paper AAAI 2021; over-engineered for day-ahead single-horizon forecasting |
| [Wu et al. NeurIPS 2021 — Autoformer (arXiv:2106.13008)](https://arxiv.org/abs/2106.13008) | Decomposition Transformer; omits positional encoding; learns 24 h and 168 h lags automatically; 38% improvement over Informer on long-horizon benchmarks |
| [Zeng et al. AAAI 2023 — DLinear/NLinear (arXiv:2205.13504)](https://arxiv.org/abs/2205.13504) | "Are Transformers Effective for Time Series Forecasting?"; DLinear beats all Transformer LTSF models; < 10k parameters |
| [Nie et al. ICLR 2023 — PatchTST (arXiv:2211.14730)](https://arxiv.org/abs/2211.14730) | Patch-based Transformer; d_model=128, patch_len=16, stride=8, 3 layers, 16 heads; learnable positional encoding; 21% MSE improvement vs best Transformer baseline; outperforms DLinear on large Electricity dataset |
| [Liu et al. ICLR 2024 — iTransformer (arXiv:2310.06625)](https://arxiv.org/abs/2310.06625) | Inverted Transformer; variate tokens; omits positional encoding; D=256, L=2–4; ECL MSE=0.178; ICLR Spotlight |
| [Chen et al. 2023 — TSMixer (arXiv:2303.06053)](https://arxiv.org/abs/2303.06053) | All-MLP time/feature mixer; outperforms DLinear and PatchTST by 1–2% at 2–3x lower memory/runtime |
| [Lim et al. 2021 — TFT (arXiv:1912.09363)](https://arxiv.org/abs/1912.09363) | Temporal Fusion Transformer; canonical reference for known-future inputs as side channels; static/known/observed input taxonomy |
| [Phan et al. 2025 — UniLF (Scientific Reports)](https://www.nature.com/articles/s41598-025-88566-4) | Best accuracy at in_len=168 for STLF across multiple settings; direct justification for 168 h sequence length |
| [PyTorch MultiheadAttention docs](https://docs.pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html) | `is_causal` parameter reference; use `attn_mask` explicitly |
| [PyTorch issue #99282 — is_causal ignored when need_weights=True](https://github.com/pytorch/pytorch/issues/99282) | Bug: causal mask silently ignored when `need_weights=True` in MHA; opened against PyTorch 2.0.0; fix status uncertain |
| [PyTorch SDPA docs](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html) | Non-determinism on CUDA; `torch.backends.cudnn.deterministic=True` for control; `sdpa_kernel()` context manager for backend selection |
| [TFB benchmark — PVLDB 2024](https://github.com/decisionintelligence/TFB) | Comprehensive and fair time series forecasting benchmark; no single model dominates all regimes |
| [MDPI Applied Sciences 2020 — TCN for energy forecasting](https://www.mdpi.com/2076-3417/10/7/2322) | TCN applied to energy time series; confirms TCN receptive field / dilated conv advantage over LSTM |
| [Frontiers Energy Research 2024 — LSTM-TCN ablation](https://www.frontiersin.org/journals/energy-research/articles/10.3389/fenrg.2024.1384142/full) | TCN better than LSTM for trend fitting; LSTM better for local detail; table/figure ablation showing component contributions |
| [arXiv:2502.12370 — Positional Encoding survey 2025](https://arxiv.org/abs/2502.12370) | Survey of eight PE methods; sinusoidal struggles with long variable-length sequences but adequate for short fixed-length sequences |

---

## Known pitfalls, CVEs, and deprecations

1. **`is_causal=True` silently ignored when `need_weights=True`** (PyTorch issue #99282, PyTorch 2.0.0). Mitigation: always pass explicit `attn_mask`. This is a correctness bug, not a deprecation.
2. **SDPA non-determinism on CUDA.** `torch.nn.functional.scaled_dot_product_attention` (used internally by `nn.MultiheadAttention` in PyTorch 2.0+) selects between Flash Attention, memory-efficient attention, and standard C++ backends on CUDA. Different backends produce slightly different outputs due to floating-point operation ordering. On CPU: deterministic. Mitigation: set `torch.backends.cudnn.deterministic = True` and optionally use `torch.nn.attention.sdpa_kernel()` to pin the backend.
3. **BatchNorm leakage on time series.** Using `nn.BatchNorm1d` in a temporal model trained with temporally ordered batches introduces information leakage (batch statistics include future timestep information). Use `nn.LayerNorm` throughout.
4. **Causal convolution padding.** The naive `nn.Conv1d(padding=(kernel_size-1)*dilation)` pads symmetrically (both left and right). For a causal TCN, padding must be left-only. The correct pattern is `F.pad(x, ((kernel_size-1)*dilation, 0))` before a `nn.Conv1d(padding=0)` layer. Alternatively, apply the full padding and then strip the right-side `padding` elements from the output. Getting this wrong introduces future leakage that is invisible in training metrics but wrong in production.
5. **Attention weight visualisation requires `need_weights=True`.** `nn.MultiheadAttention` defaults to `need_weights=True` but this is expensive for production use. The `need_weights=False` path uses a faster SDPA fused kernel. For the demo's attention-weight visualisation cell, pass `need_weights=True` explicitly and remember that this triggers the PyTorch 2.0 `is_causal` bug (see item 1 above).
6. **Train/val boundary leakage in sequence datasets.** The last `seq_len` steps of the training window overlap with the first `seq_len` steps of the validation window when a SequenceDataset is built from the full time series without a gap. Offset the train/val split by `seq_len` steps to prevent this, or assert in the test suite that no training sequence has any future-index in common with the validation set.
7. **`torch.use_deterministic_algorithms(True)` on GPU path.** Requires `CUBLAS_WORKSPACE_CONFIG=:4096:8` environment variable for cuBLAS operations (Stage 10 R1). The Dockerfile should set this if CUDA training with deterministic mode is enabled.

---

## Version and compatibility notes

- **torch>=2.7** (pyproject.toml pin): All APIs needed — `nn.Conv1d`, `nn.MultiheadAttention`, `nn.TransformerEncoder`, `nn.LayerNorm`, `torch.nn.functional.scaled_dot_product_attention`, `torch.nn.utils.weight_norm` — available since PyTorch 1.0–2.0. No version acquisition needed.
- **`scaled_dot_product_attention`:** Introduced as beta in PyTorch 2.0; used internally by `nn.MultiheadAttention` in 2.0+. Available in 2.7. Non-determinism note above applies.
- **`is_causal` parameter on `MultiheadAttention.forward`:** Introduced in PyTorch 2.0. Bug present in 2.0.0 when `need_weights=True`. The pin `>=2.7` means the fix may or may not have landed (fix confirmation was not retrieved). Use explicit `attn_mask` as the safe path.
- **PatchTST / iTransformer are not in-scope architectures** for Stage 11 (they are external references, not code dependencies). The architectures to be implemented (TCN or small Transformer) use only PyTorch core modules.
- **No new runtime dependencies are needed** beyond `torch>=2.7` already declared. `torch.nn.LayerNorm`, `torch.nn.Conv1d`, `torch.nn.TransformerEncoder` are all in `torch.nn` core.

---

## Open questions returned to the plan author

1. **Architecture choice.** The trade-off table (Section 2) lays out TCN (simpler, faster, pedagogically clean causal structure) vs small Transformer (attention visualisation, more transferable pattern, 10–40× slower on CPU per epoch). If the live demo must train in under 10 minutes, TCN is forced unless the Transformer is very small (depth=2, d_model=64, seq_len=96). The plan author should benchmark on target hardware before specifying the architecture in the plan.
2. **DLinear inclusion.** The literature strongly suggests including DLinear (< 10k parameters, seconds to train) as a temporal baseline row in the ablation table. It is the "are Transformers needed?" sanity check and makes the ablation table intellectually honest. Requires `nn.Linear` only; trivially protocol-compatible. Should Stage 11 implement DLinear alongside the primary temporal architecture?
3. **Sequence length.** The plan author should choose between 96, 168, or 336 h. The literature defends 168 most directly (weekly cycle; UniLF 2025 best-accuracy result). The plan should document the choice with a citation from this document.
4. **Train/val gap.** Stage 10's implementation uses the last 10% of training data as a contiguous val tail. For a 168-step sequence model, the last 168 steps of the training window overlap with the start of the val window. The implementer must decide whether to (a) offset the val split by `seq_len` steps or (b) accept the overlap on the grounds that the held-out test set (managed by the harness, not inside `fit`) is already fully separated. This is an implementation-level decision with correctness implications; it should be specified in the plan.
5. **Attention visualisation scope.** The intent flags attention visualisation as "sometimes illuminating, not worth building an interpretability stage around." Given the `need_weights=True` / `is_causal` interaction bug, the notebook's attention cell should be explicitly documented as requiring `attn_mask` rather than `is_causal`. The plan should specify whether a failed attention cell (confusing weights) is acceptable as a "negative result" demo moment or whether it should be omitted if weights are uninterpretable.
6. **SequenceDataset helper scope.** A `SequenceDataset(features_df, target, seq_len, stride)` class is required for both TCN and Transformer paths. This is new infrastructure not present in Stage 10. The plan should specify whether it lives in `models/` (model-internal, private), `features/` (shared feature engineering), or is a standalone utility. The Stage 4 protocol does not require it to be exposed publicly.
7. **Dispatcher and config extension.** Stage 11 adds a sixth model type. The plan should decide whether Stage 11 consolidates the registry/train-CLI dispatcher at this point or defers further.
