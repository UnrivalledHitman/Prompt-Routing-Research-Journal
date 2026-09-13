# Experimental Protocol — ARBITER

**CONFIDENTIAL. Patent filing pending. Do not publish or make this repository public.**

Version 0.1, 2026-09-11. This document is the specification the code implements and the source text for
the paper's Methodology and Experimental Setup sections. It is written so that a competent researcher
could reproduce the study from this document alone, without reading the code.

Open decisions that need Indrajeet's input are marked **[DECISION]**. Assumptions I made in order to keep
moving are marked **[ASSUMPTION]** and can be overturned cheaply at this stage.

---

## 1. Research question and hypotheses

Users of tiered LLM services choose a model and a reasoning setting manually, and they choose badly in two
opposite directions. They start too low and escalate after failure, paying for every discarded attempt.
Or they leave a heavyweight configuration selected from a previous task and pay a premium on trivial
prompts. Both are misallocation. Neither has been measured.

The study asks whether a small predictor placed in front of the pool can select the cheapest configuration
that will succeed, in one shot, and whether the saving survives being charged for the predictor itself.

**H1 (escalation tax exists and is large).** Under plausible manual-escalation policies, the expected cost
per solved prompt exceeds the cost of the minimum sufficient configuration by a margin that is
significant in tokens, latency and joules.

**H2 (one-shot prediction beats escalation).** A router that predicts the configuration before any
generation attains a lower expected cost at matched quality than any manual-escalation policy, and than a
reactive cascade that must generate before it can decide.

**H3 (router cost is not free, and ordering changes when it is charged).** Ranking router families by
gross saving gives a different ordering than ranking them by saving net of their own latency, VRAM and
joules. Specifically, an LLM-based router is expected to lose to a classical model once charged.

**H4 (reasoning effort has a distinct energy signature).** Joules per solved query for the thinking mode
is not a fixed multiple of the non-thinking mode; the ratio varies by model size and task family, and
there exist regions of the grid where thinking costs more energy than moving up a size.

**H5 (misallocation is asymmetric).** The cost of routing one tier too low (a failure, then a retry) and
one tier too high (waste) differ by an order of magnitude, so accuracy against the oracle is a poor
objective and threshold-tunable probabilistic routing dominates argmax classification.

H4 and H5 are the ones I would defend hardest. H1 and H2 are the headline; H3 is the methodological
contribution.

---

## 2. System under study

### 2.1 The configuration grid

The routed pool is a two-dimensional grid of **model size x reasoning mode**, not a one-dimensional
ladder. Qwen3 is used because it is the only open family offering a clean size ladder with a documented
thinking/non-thinking switch and a controllable thinking budget (`qwen3`, arXiv:2505.09388).

| Cell | Model | Mode | Approx. weights (Q4_K_M) |
|------|-------|------|--------------------------|
| c1 | Qwen3-0.6B | no_think | ~0.5 GB |
| c2 | Qwen3-0.6B | think | ~0.5 GB |
| c3 | Qwen3-1.7B | no_think | ~1.2 GB |
| c4 | Qwen3-1.7B | think | ~1.2 GB |
| c5 | Qwen3-4B | no_think | ~2.5 GB |
| c6 | Qwen3-4B | think | ~2.5 GB |
| c7 | Qwen3-8B | no_think | ~5.0 GB |
| c8 | Qwen3-8B | think | ~5.0 GB |

Eight cells. Only one is resident at a time; the 8 GB VRAM budget forbids co-residency of the larger
cells, and the labelling loop is therefore organised model-major to avoid reload thrash (§9).

**[ASSUMPTION]** Q4_K_M quantisation throughout, for VRAM headroom and because it is the realistic
consumer deployment. Quantisation is held constant across cells so it cannot confound the size axis.
The cost of that choice is that results describe quantised models, which must be stated as a limitation.

### 2.2 Cost of a cell

Cells are not ordered by parameter count alone. Cost is measured, not assumed:

```
cost(c, q) = alpha * tokens_in(c,q) + beta * tokens_out(c,q)      [token cost, for API analogy]
latency(c, q) = wall-clock seconds, first token to last
energy(c, q)  = integral of GPU power over the generation window, joules  [§6]
```

Thinking mode inflates `tokens_out` by emitting a reasoning trace before the answer. Whether it also
inflates joules proportionally is exactly H4 and must not be assumed.

---

## 3. Formalising the escalation tax

This is the paper's central construct and the one no located prior work defines. It must be airtight.

Let `C = {c_1 ... c_K}` be the grid ordered by measured cost. For a prompt `q`, let `s(q,c) in {0,1}`
denote whether cell `c` produces an acceptable answer under the success criterion of §5.

**Minimum sufficient configuration.**
```
c*(q) = argmin_{c in C : s(q,c) = 1} cost(c, q)
```
undefined when no cell succeeds. Such prompts are retained and reported separately as the *unservable
fraction*; discarding them would flatter every policy.

**Manual escalation policy.** A policy `pi` is a starting cell `c_start` plus an ordered ladder
`L = (l_1, l_2, ...)`. The user submits `q` to `c_start`; on perceived failure the full prompt is
resubmitted to `l_1`, and so on. Total cost accrues over every attempt, including failed ones:

```
Cost(pi, q) = sum over attempted cells a of [ alpha*tokens_in(a,q) + beta*tokens_out(a,q) ]
```

Note `tokens_in` is paid in full at every attempt. Context resubmission is the mechanism that makes
escalation superlinear, and it is the part practitioners complain about and the literature ignores.

**The tax decomposes into two opposite failures.**

```
Tax(pi, q) = Cost(pi, q) - cost(c*(q), q)
           = Escalation(pi, q) + Stickiness(pi, q)
```

- **Escalation component** — cost of attempts strictly cheaper than `c*(q)` that necessarily failed.
  This is "I started on Haiku and had to climb."
- **Stickiness component** — excess of the finally-used cell over `c*(q)` when the user began above the
  minimum sufficient cell and never needed to move. This is "I still had Opus selected from yesterday."

Reporting these separately matters, because they respond to different fixes and because they map exactly
onto the two scenarios that motivated the project.

### 3.1 Escalation policies to evaluate

**[ASSUMPTION]** Four policies, spanning optimistic to pessimistic for the baseline. We are constructing
our own comparator, so it must be constructed generously — a strawman baseline is the fastest way to lose
a reviewer.

| Policy | Start | Ladder | Rationale |
|--------|-------|--------|-----------|
| P1 `frugal-climb` | c1 | c1 -> c3 -> c5 -> c7 -> c8 | Cost-conscious user, sizes first, thinking last |
| P2 `think-first` | c1 | c1 -> c2 -> c4 -> c6 -> c8 | User reaches for "thinking" before a bigger model |
| P3 `sticky-high` | c8 | none (no escalation needed) | Heavyweight left selected from a previous session |
| P4 `sticky-mixed` | drawn from the empirical distribution of previously used cells | P1 ladder | Realistic mixture |

**Failure detection.** The escalating user must notice failure. Two regimes are reported:
- *Oracle detection* — the user always notices. This is **optimistic for the baseline** and gives a
  lower bound on the tax.
- *Imperfect detection* — the user accepts a wrong answer with probability `p_accept` and re-tries a
  correct one with probability `p_retry`. **[DECISION]** Defaults of `p_accept = 0.15`, `p_retry = 0.05`
  are placeholders; ideally these are grounded in a citation or a small user study, otherwise they must
  be swept as a sensitivity analysis and clearly labelled as assumed.

The headline number should quote oracle detection, because it is the conservative one.

---

## 4. Corpus construction

Four task families, chosen so that three are programmatically gradable and the fourth reflects the traffic
that motivates the product problem.

| Family | Sources | Grading | Pilot n | Full n |
|--------|---------|---------|---------|--------|
| Math | GSM8K, MATH | exact match on normalised final answer | 200 | 800 |
| Code | HumanEval, MBPP | unit-test execution in a sandbox | 200 | 800 |
| Knowledge/MCQ | MMLU, ARC-Challenge | option-letter match | 200 | 800 |
| Open-ended | instruction-following and summarisation prompts | LLM judge, §5.4 | 200 | 800 |

**[DECISION]** The open-ended source is unsettled. It must be redistributable and license-clean. Options
are an Arena-style prompt set, or a held-out slice of an instruction dataset. I will confirm licence and
availability before committing, and will not cite a dataset I have not downloaded.

**Sampling.** Stratified by source dataset and by an a-priori difficulty proxy (MATH level, MMLU subject
group, MBPP task id band) so the corpus is not concentrated at one difficulty. Seeded and recorded.
Deduplicated by normalised text hash. Prompts longer than 2048 tokens are excluded so that context length
does not confound the size axis.

**Splits.** 60/20/20 train/validation/test by prompt, stratified by family and by oracle label. The test
split is touched exactly once, at the end. Router hyperparameters are selected on validation only.

---

## 5. Oracle labelling

Every prompt is run through **all eight cells**. This is the expensive part and the corpus that results is
a genuine contribution.

### 5.1 Decoding parameters

**[ASSUMPTION]** Qwen3 documents different recommended sampling settings for the two modes, and greedy
decoding is discouraged in thinking mode. Forcing greedy everywhere would be cleaner statistically but
would misrepresent thinking mode, so instead: use each mode's recommended settings, fix the seed, and draw
`k = 3` samples per (prompt, cell). Success is by majority. Seeds, sampling parameters and the exact
chat template are recorded per run.

`max_new_tokens` is capped per mode (`no_think` 1024, `think` 4096 including the reasoning trace). Hitting
the cap counts as a failure and is logged as such, because in deployment it is one.

### 5.2 Success criteria

- **Math** — normalise (strip commas, units, LaTeX wrappers), compare final answer. Both boxed and plain
  formats accepted.
- **Code** — extract the function, execute the reference unit tests in a subprocess sandbox with a
  timeout and no network. Pass/fail on all tests.
- **MCQ** — parse the selected option; unparseable output is a failure, not a resample.
- **Open-ended** — §5.4.

### 5.3 The label

```
y(q) = c*(q)  the cheapest cell with majority success
```
plus the full success vector `s(q, c_1..c_8)` and the measured cost/latency/energy of every cell. The
vector matters more than the scalar label: it is what lets us train per-cell success predictors (§7) and
sweep the quality threshold at test time.

### 5.4 Judging open-ended outputs, and validating the judge

A judge drawn from the routed pool would be circular, so the primary judge is **out-of-family and outside
the pool**: `llama3.1:8b-instruct-q4_K_M` (Llama 3.1 8B Instruct, 4.58 GiB), fixed on 2026-09-12 when
Indrajeet approved the download. It is loaded only after labelling, never alongside a routed model, and it
will not judge while a routed model is loaded. Settings are in `configs/judge.yaml`; the code is
`src/arbiter/eval/judge.py` and `scripts/run_judge.py`.

Scoring is a rubric on a 1-5 scale against the Dolly reference, with a success threshold of >= 4. The judge
sees the request (with any Dolly context), the reference and the visible answer; never the think trace,
the model or the mode. For open requests such as brainstorming or creative writing, the rubric treats the
reference as one acceptable answer among many, and it asks for content to be scored, not length.
Position and verbosity biases are mitigated by scoring single responses rather than pairs, in a seeded
random order. Decoding is greedy (temperature 0, fixed seed) and constrained to a JSON schema
`{reason, score}`, so scores are reproducible. A reply whose score cannot be read is retried once with a
longer output budget and counts as a failure if still unreadable. An answer that hit its token cap, or has
no visible text, fails without being judged. Each judge run records the rubric's hash, the model digest and
the counts of unreadable replies and shortened references (`judge_runs.jsonl`).

**Judge validation.** Cheap judges are the standard attack surface for this kind of paper, so the judge is
itself evaluated (`scripts/sample_judge_validation.py`, `scripts/judge_agreement.py`):
1. ~200 judged items, stratified by Dolly category and judge verdict, are re-scored by a hosted reference
   model with the same rubric and schema (`scripts/judge_openai.py`). **Resolved 2026-09-13:** the model
   is `gpt-5-mini` and Indrajeet authorised **$0.36** for it. That authorisation is separate from the
   $8.00 of the hosted arm (§14), and the two are not fungible -- the hosted arm closed at $7.07 and its
   $0.93 of headroom is not spent here.

   `--estimate` prints an **upper bound**, not the expected spend: it prices every request at the full
   prompt budget (~9,000 tokens, §5.4 `num_ctx`) and the full reply budget, which bounds 200 items at
   about $1.25 on `gpt-5-mini` against a realistic ~$0.30, since an open-ended request is 83 tokens at
   the median and a judged answer a few hundred. The ceiling is therefore enforced by `--max-usd`, which
   charges the next item its full prompt and reply budget and refuses it if that could carry the running
   spend past the ceiling, rather than by the estimate. The guard counts in dollars, the unit the
   authorisation is given in, so no exchange rate stands between the two; `--inr-per-usd` only adds a
   rupee figure to the log for a credit priced that way. If the guard stops the run short of 200 items,
   the sample is reported at the size reached and any increase goes back to Indrajeet before a further
   request is sent.
2. Indrajeet scores ~100 of those items blind, from a sheet that shows the request, reference and answer
   but not the model, mode or judge score, giving a human anchor.

Agreement is Cohen's kappa on the success decision and quadratic-weighted kappa on the 1-5 scores, each
with a 10,000-resample percentile bootstrap interval.

**Pre-registered rule (fixed 2026-09-12, before any judge output existed).** The deciding statistic is the
judge-human kappa on the success decision:
- kappa >= 0.60: the open-ended family is reported alongside the gradable families;
- 0.40 <= kappa < 0.60: it is reported with an explicit caveat wherever its numbers appear;
- kappa < 0.40: it moves to a secondary analysis and is left out of headline numbers.

The cut-offs are the moderate and substantial boundaries of Landis and Koch (1977). Judge-frontier kappa is
reported but does not decide. The rule is recorded before any routing result exists, so it cannot become a
post-hoc rescue. For the mentor: confirm the cut-offs before the human labels are collected.

---

## 6. Energy instrumentation

The energy contribution is what survives of the sustainability claim (see the prior-art register), so the
measurement has to be defensible.

**Method.** NVML via `nvidia-ml-py`. A sampler thread polls `nvmlDeviceGetPowerUsage` at 100 ms during
each generation and integrates over the window by the trapezoidal rule to obtain joules. Reported are:
- **Total energy** — the raw integral.
- **Dynamic energy** — total minus the idle baseline measured over 60 s immediately before the run.

Dynamic energy is the honest number for attributing cost to a query; total is reported alongside because
data-centre accounting usually quotes it.

**Carbon.** `gCO2e = (joules / 3.6e6) * grid_intensity`, computed by `arbiter.energy.carbon`
and reported by `analysis.carbon_table` across the intensities listed in `configs/analysis.yaml`
(50, 250, 475 and 700 gCO2e/kWh) rather than at one value. Those are round values spanning the
range grids occupy, labelled as assumed wherever they appear; the conversion is linear, so a
reported figure rescales to any other intensity. This follows the accounting style of
`luccioni2024powerhungry` and `lacoste2019quantifying`.

**The router is measured with the identical harness.** Every router family's own joules, latency and peak
VRAM are recorded the same way and subtracted from the saving. This is the thing GreenServ, EcoThink and
the QoS-aware edge router all omit, and it is the reason H3 is worth testing.

### 6.1 Gate: NVML power telemetry may not exist on this GPU

**This is a hard risk and must be checked before anything else.** Some laptop GPUs do not expose board
power through NVML, and `nvmlDeviceGetPowerUsage` returns `NVML_ERROR_NOT_SUPPORTED`. On an RTX 4060
Laptop this is genuinely uncertain.

If unsupported, fallbacks in order of preference:
1. `nvmlDeviceGetTotalEnergyConsumption` — a monotonic millijoule counter, present on some parts where
   instantaneous power is not.
2. An external wall-socket power meter, measuring whole-system draw with the idle baseline subtracted.
   Coarser, but honest, and arguably closer to real environmental cost.
3. Drop measured energy to a secondary result and lead on tokens and latency, stating plainly that energy
   was estimated rather than measured.

**Option 3 costs the paper a contribution.** The check is one short script and must run first.

### 6.2 Gate 1 result (run 2026-09-11)

Script `scripts/gate1_nvml_telemetry.py`. Raw output in `results/processed/gate1_nvml_telemetry.json`
and `results/processed/gate1_nvml_samples.csv`. Hardware: RTX 4060 Laptop GPU, driver 616.56,
NVML 13.616.56, enforced power limit 115 W. Procedure: 10 s idle before any CUDA context, 8 s of fp16
2048x2048 matrix multiplication at ~90% utilisation (40 MB peak allocation), 8 s cooldown, 100 ms polling.

| Source | Supported | Idle | Load | Cooldown | Verdict |
|--------|-----------|------|------|----------|---------|
| `nvmlDeviceGetPowerUsage` | yes | 2.43 W | 75.1 W mean, 88.4 W max | 10.5 W | **Usable** |
| `nvmlDeviceGetTotalEnergyConsumption` | yes, monotonic | implies 88.3 W | implies 127.6 W | implies 256.9 W | **Rejected — non-physical** |

**Instantaneous power passes.** It is plausible at P8 idle, tracks a step load within about a second, and
never exceeds the enforced limit.

**The energy counter fails, and would have silently corrupted every energy result.** It is supported and
strictly monotonic, so a naive check accepts it, and the script's first-pass verdict did exactly that.
Its implied mean power is 88 W with the GPU idle at P8, 128 W under load, which is above the 115 W enforced
limit, and 257 W during cooldown, which is higher than under load. A units error would scale all three
windows by the same factor and preserve their ordering; this ordering does not follow real draw at all.
Only the cross-check between the two sources exposed it: 41% disagreement under load and a factor of 36
at idle. The verdict rules now require agreement with sampled power and a plausibility bound against the
enforced limit before the counter can be trusted.

**Temporal resolution constrains the design.** Reported power refreshes every 0.5 s and is smoothed.
Under a step load it climbs 10.5, 31.8, 63.4, then 88 W over roughly a second, and it still reads 88 W at
the start of the cooldown window. A 0.6B no-think answer to a multiple-choice question can finish inside a
single refresh period, so its energy cannot be attributed from its own window.

**Amendment to §6 Method.** Energy is integrated over *batches*, not individual queries. Each
(cell, batch) window runs many prompts back to back and is extended by a 2 s tail after the last
generation to capture the smoothed decay, with the idle baseline subtracted. Energy is apportioned to
queries within the batch by each query's share of generation wall time, and that apportionment is
validated against token share on a subsample. Routers are measured the same way.

**Confound recorded.** NVML reports whole-board power, so any other GPU workload during a labelling run
contaminates the measurement. Runs require an otherwise idle GPU; the harness records utilisation
throughout and flags any window where it is inconsistent with the scheduled work.

**Status: Gate 1 passed, for instantaneous power sampling only.** The energy contribution survives. The
cumulative counter is excluded from the study.

---

## 7. Router families

All routers predict the **per-cell success vector**, not the argmax label:

```
f: q -> (p_1 ... p_8),  p_i = estimated P(s(q, c_i) = 1)
```

then select `argmin cost(c_i)` over cells with `p_i >= tau`. Decoupling prediction from policy means one
trained model yields a whole cost-quality curve by sweeping `tau`, matching the test-time-tunable quality
knob of `ding2024hybridllm`. Direct 8-way classification is retained as an ablation to show the
decoupled form is better.

**R1 — Classical ML.** (a) TF-IDF over word and character n-grams into logistic regression; (b) hand
features (token count, digit and operator density, code fences, imperative mood, question type, readability)
into gradient boosting (XGBoost, LightGBM); (c) frozen sentence-transformer embeddings into logistic
regression. Cheap, CPU-viable, and the honest baseline that an LLM router must beat.

**R2 — Fine-tuned encoder.** DistilBERT and DeBERTa-v3-small with eight sigmoid heads, trained with
class-weighted BCE. Small enough to train on 8 GB in minutes.

**R3 — LLM-as-router.** Qwen3-0.6B prompted zero-shot and few-shot to emit a configuration, with
constrained decoding; plus a LoRA-tuned variant. Deliberately the smallest pool member, because a router
larger than the models it routes to is self-defeating.

**Cost-sensitive training.** Because misallocation is asymmetric (H5), a variant weights errors by their
realised cost consequence rather than treating all misroutes equally.

**Implementation (2026-09-12).** `src/arbiter/routers/neural.py`, configured under `routers.neural` in
`configs/analysis.yaml`. Weights are read from the local Hugging Face cache at pinned revisions.
- R1c: normalised all-MiniLM-L6-v2 embeddings into per-cell logistic regression, over the R1a `C` grid.
- R2: first-token pooling into eight sigmoid outputs. Per-cell balanced class weights, multiplied by
  the cost-sensitive weights in the `-cs` variant. AdamW with linear warm-up and decay, bfloat16
  autocast on the GPU, grid over learning rate and epochs. Decisions run on the CPU and are timed and
  charged like R1. DeBERTa-v3's vocabulary is read with `sentencepiece` directly, because the
  Transformers tokenizer needs protobuf or tiktoken.
- R3-LoRA: adapters (rank 16, alpha 32) on every attention and MLP projection. Cross-entropy over the
  eight letters at the first answer position of the zero-shot prompt, with thinking off. The label
  is `c*` in mean-cost order, or the costliest cell for unservable prompts. No cost-sensitive
  variant, since the weights are per cell. Charged the zero-shot probe's metered cost per prompt, an
  upper bound (the probe decodes up to 8 tokens), and left out until that probe exists. Decisions
  are computed in bfloat16, while the probe ran Q4_K_M.
- Training waits for an otherwise idle GPU with 5 GB free, using the labelling harness's guard.

---

## 8. Baselines and metrics

**Baselines.** Always-c1 through always-c8 (eight static policies); random; **oracle** `c*` (unreachable
upper bound); the four manual-escalation policies of §3.1; and a reactive cascade in the style of
`aggarwal2024automix`, which generates on the small model and escalates on a self-verification signal.
That last one is the sharpest comparator: it is the strongest published thing that also pays for failed
attempts, and beating it requires the *predictive* advantage, not just cost accounting.

**Primary metrics.**
- Cost-quality Pareto frontier over `tau`, in tokens, seconds and joules.
- Area under the frontier, normalised between the weakest and strongest static policies.
- **Net saving**, router cost included, against each baseline — the headline for H3.
- **Joules per solved query** — energy normalised by successful outcomes, which penalises a cheap cell
  that fails and forces a retry. This is the escalation tax denominated in energy and it links H1 to H4.
- **Regret decomposition** — over-routing cost and under-routing failure rate reported separately, never
  summed into a single accuracy figure.
- Router overhead: p50/p95 latency, peak VRAM, joules per decision.

**Statistics.** Bootstrap 95% confidence intervals (10,000 resamples) on every headline number. Paired
comparisons across prompts, Holm-Bonferroni corrected across the policy family. Three training seeds per
router; mean and spread reported. No result is claimed without an interval.

---

## 9. Execution plan and compute budget

Runs are **model-major**: load one model once, sweep both modes and all prompts, unload. This turns eight
model loads into four and avoids reload thrash on a memory-tight GPU.

| Phase | What | GPU time | Gate |
|-------|------|----------|------|
| 0a | NVML power telemetry check (§6.1) | minutes | **Blocks the energy contribution** |
| 0b | Mode-separation validation (§10) | 1 h 54 min (actual) | **Passed 2026-09-11**, see §10.1 |
| 0c | Per-cell throughput calibration, budget re-estimate | answered by gate 2 data | Budget re-estimated in §10.1 |
| 1 | Pilot labelling, 800 prompts x 8 cells x k=3 | planned ~8-12 h; **measured rate implies ~75 h** | Superseded by §13 |
| 1' | Smoke stage, 6 prompts x 0.6B/1.7B | 5 min (actual) | **Passed 2026-09-11**, §13.3 |
| 1'' | Pilot labelling, 300 gradable prompts x 8 cells x k=1 | ~9.3 h estimated from gate 2 rates | Started 2026-09-11 |
| 2 | Router training and evaluation on pilot | ~2-4 h | First real numbers |
| 3 | Main labelling, 900 further gradable prompts | ~28 h | After the pilot is clean |
| 3a | Noise subset, 75 prompts x 2 extra samples | ~4.7 h | — |
| 3b | Cap sensitivity, think cap 8192 on cap-hit prompts | ~5 h | — |
| 3c | Open-ended family, 400 prompts, plus judge | ~14 h + ~4.5 h | Downloads approved 2026-09-12; judge confirmed 2026-09-13 |
| 4 | Full router sweep, ablations, seeds | ~6-10 h | — |
| 5 | Public-benchmark arm (no local inference) | 4 min (actual) | **Done 2026-09-12**, see §13.5 |
| 6 | Hosted-frontier arm (no local inference, $8 cap) | minutes (batch) | **Started 2026-09-12**, see §14 |

GPU use for the study was granted on 2026-09-11. The harness still waits for an otherwise idle GPU
before loading a model and reruns any window touched by another workload, so the user's own GPU jobs are
never co-run with labelling.

---

## 10. Threats to validity

**Mode separation may not be real.** `wang2025demystifying` (arXiv:2510.12680) finds hybrid-thinking
models achieve only partial separation — reasoning leaks into no-think mode. Their improved recipe cut
contaminating reasoning tokens on MATH500 from 5917 to 522, better but not clean. If `/think` and
`/no_think` are not distinct operating points on our checkpoints, the reasoning-mode axis is confounded
and every energy result inherits it.

*Mitigation, phase 0b:* run a stratified sample through every size in both modes and measure realised
output length, thinking-token count and reasoning-marker frequency. Report separation empirically, with
effect sizes. **If separation is weak at some sizes, that is a finding and gets published as one** — it
is not a reason to quietly proceed.

### 10.1 Gate 2 result (run 2026-09-11)

Script `scripts/gate2_mode_separation.py`, config `configs/gate2_mode_separation.yaml`. Summary in
`results/processed/gate2_mode_separation.json`, per-generation metrics (no text) in
`results/processed/gate2_records.csv`, model digests in `gate2_models.json`. Procedure: 60 prompts,
10 each from GSM8K, MATH-500, HumanEval, MBPP, MMLU and ARC-Challenge. Each prompt ran once per size in
both modes, 480 generations in total, with the **same 4096-token cap in both modes** and mode order
alternated ABBA across prompts. Wall time 1 h 54 min. All four models ran fully in VRAM, with no errors
and no retries.

| Size | AUC | Paired dominance | Median tokens, think / no_think | Median ratio (95% CI) | Leak rate | Marker density ratio | Think cap-hit | Status |
|------|-----|------------------|----------------------------------|-----------------------|-----------|----------------------|---------------|--------|
| 0.6B | 0.964 | 1.000 | 1444 / 195 | 13.9 (9.4–19.7) | 0 | 0.040 | 21.7% | separated |
| 1.7B | 0.954 | 1.000 | 2053 / 184 | 9.1 (7.6–11.9) | 0 | 0.034 | 31.7% | separated |
| 4B | 0.928 | 0.983 | 1429 / 149 | 7.0 (5.3–11.5) | 0 | 0.042 | 25.0% | separated |
| 8B | 0.953 | 1.000 | 1814 / 180 | 9.7 (7.7–17.0) | 0 | 0.045 | 21.7% | separated |

Wilcoxon signed-rank p < 2e-11 at every size. No no_think response had a non-empty thinking field or a
literal `<think>` tag. Reasoning-marker density in no_think answers is 3–5% of the density in think traces.

**Status: Gate 2 passed at all four sizes.** The size x mode grid stands. The leakage reported by
`wang2025demystifying` does not appear on these checkpoints under Ollama's `think=false`, at least by
these three measures.

**Limits of the check.** Sixty prompts per size. The marker lexicon is a proxy for reasoning-style
language, not a semantic judgement. Think lengths are right-censored at 4096, which works against
separation, so the ratios above are underestimates. On one 4B prompt the no_think output was at least as
long as the think output; the cause has not been examined.

The run also produced three findings that change the labelling plan.

**Finding 1 — the §9 compute budget is about 6-7x too low.** Mean wall time per prompt, one sample:

| Size | no_think | think |
|------|----------|-------|
| 0.6B | 0.9 s | 7.6 s |
| 1.7B | 2.0 s | 15.8 s |
| 4B | 4.0 s | 27.6 s |
| 8B | 6.6 s | 47.6 s |

All eight cells together take 112 s per prompt, and the think cells account for 88% of that. At this rate
the pilot (800 prompts x 8 cells x k=3) needs **~75 h**, against 8-12 h planned. Full labelling (3,200
prompts) needs **~300 h**, against 35-50 h. With k=1 the figures are ~25 h and ~100 h. These projections
use the gate 2 family mix and exclude the open-ended family.

**[DECISION]** Options, not mutually exclusive:
- (a) k=1 instead of k=3, reporting single-sample label noise as a limitation. This cuts the time by 3x.
- (b) A smaller full corpus, for example 1,600 prompts. Router training needs are modest, so this is
  probably viable.
- (c) Concurrent requests (`OLLAMA_NUM_PARALLEL`), which raise aggregate throughput at the cost of KV-cache
  VRAM. This is compatible with batch-level energy integration (§6.2, added in PR #4) but must be measured
  first.
- (d) Keep k=3 only for a subsample, to estimate label noise.

My recommendation is (a), (b) and (d) together. That is ~25 h for the pilot and ~50 h for a
1,600-prompt corpus, with label noise measured rather than assumed.

**Finding 2 — at a 4096 cap, truncation causes most think-mode failures.** Every think response that hit
the cap was graded wrong. Capped responses account for 7 of 23 think-mode errors at 0.6B, 11 of 17 at
1.7B, 8 of 13 at 4B and 7 of 9 at 8B. Graded accuracy on math and MCQ:

| Size | think | think, uncapped only | no_think |
|------|-------|----------------------|----------|
| 0.6B | 0.42 | 0.52 | 0.33 |
| 1.7B | 0.57 | 0.79 | 0.70 |
| 4B | 0.68 | 0.84 | 0.90 |
| 8B | 0.78 | 0.94 | 0.95 |

Under the §5.1 rule that a cap hit is a failure, think cells will rarely be the minimum sufficient
configuration for math and MCQ at 1.7B and above. The label will then reflect the cap as much as the
model. The rule is defensible, because a truncated answer is useless in deployment, but the result depends
on the cap and the paper must say so. Caveats: n = 40 graded responses per mode per size, approximate
grading, and code was not executed.

**[DECISION]** Keep 4096 and report it as a deployment constraint; raise the think cap (8192 roughly
doubles think-cell time); or keep 4096 and add a cap-sensitivity sweep on a subsample. I recommend the third.

**Finding 3 — the planned no_think cap of 1024 would truncate some correct-length answers.** 2, 3, 5 and 3
of 60 no_think responses (by size) exceeded 1024 tokens. **[DECISION]** Raise the no_think cap to 2048.
No_think cells are 12% of wall time, so the cost is small.

**Operational note — warm-up.** Prompt prefill stalled twice in the whole run, 5.5 s and 6.2 s at the
second and fourth generations of the first model, and never after. Loading a new model did not bring the
stall back. The labelling and energy harnesses must therefore run one think and one no_think warm-up
generation per Ollama server session and exclude them from measurement. Throughput in tokens/s,
think / no_think: 0.6B 269 / 312, 1.7B 146 / 162, 4B 72 / 76, 8B 44 / 46.

**Quantisation confound.** All cells are Q4_K_M, so size comparisons are between quantised models.
Constant across cells, but it bounds external validity and is stated as a limitation.

**Small-model pool.** The largest cell is 8B. Frontier tiers behave differently, which is precisely why
the public-benchmark arm exists.

**Judge bias.** Addressed by out-of-family judging plus dual validation (§5.4).

**Assumed human behaviour.** The escalation policies are constructed, not observed. This is the most
attackable part of the paper. It is handled by reporting a family of policies rather than one, by making
the headline figure the optimistic-for-baseline case, and by sweeping the detection parameters. **A small
observational study of real escalation behaviour would strengthen this more than any additional
experiment** — worth discussing with your mentor.

**Single-machine, single-GPU.** One hardware configuration. Energy figures do not transfer to other
silicon and will not be claimed to.

---

## 11. Reproducibility

- Every experiment driven by a YAML config under `configs/`; no constants in code.
- All seeds set and recorded (Python, NumPy, PyTorch, and the sampler seed per generation).
- A run manifest per experiment: git commit, config hash, package versions, driver and CUDA versions,
  GPU model, timestamp.
- Raw generations stored as JSONL with per-record token counts, latency and joules.
- `logging`, not `print`. Shapes and types validated at module boundaries.

---

## 12. Open decisions

1. **Open-ended dataset** — resolved 2026-09-12: Dolly-15k (§13.2).
2. **Judge model identity** — resolved 2026-09-12, confirmed by Indrajeet 2026-09-13:
   `llama3.1:8b-instruct-q4_K_M` (§5.4, §13.2).
3. **Validation reference model and its spend** — resolved 2026-09-13: `gpt-5-mini` under a $0.36 ceiling
   authorised by Indrajeet, enforced by `--max-usd` (§5.4).
4. **`p_accept` / `p_retry`** — resolved 2026-09-11: swept over a grid and labelled as assumed, with the
   headline tax under oracle detection (§13).
5. **Title-block details** — author name, institution, department, city/country, mentor's co-authorship
   and ORCID. Needed for the LaTeX template, not for any experiment.

Items 1-4 are resolved; 1, 2 and 4 are recorded in §13. Item 5 and a code/data availability statement
remain with Indrajeet and the mentor.

---

## 13. Adopted decisions (2026-09-11)

These defaults were adopted to run the study within the measured compute budget. Each is recorded before
any routing result exists, and each can be revisited; where a choice could move a headline number, a
sensitivity analysis is planned.

### 13.1 Budget-driven changes to §4 and §5

**Samples per cell: k = 1, with a label-noise subset.** Gate 2 measured 112 s to answer one prompt in all
eight cells, so k = 3 on 3,200 prompts would take about 300 GPU-hours. Instead each (prompt, cell) is
sampled once, and a noise subset of 25 prompts per family is sampled twice more. On that subset the
disagreement between samples bounds the label noise, and the majority label is compared with the
single-sample label to show how often k = 1 changes `c*`.

**Corpus: 400 prompts per family, 100 of them in a nested pilot.** The pilot is the first 100 prompts
selected per family, so pilot results are a strict subset of the final data. Gate 2 prompts are excluded.
Code takes all 154 remaining HumanEval problems plus 246 MBPP problems. Selection is in
`configs/corpus.yaml`; the manifest, without prompt text, is `results/processed/corpus_manifest.csv`.

**Token caps.** Think mode keeps 4,096 tokens including the trace. No-think mode is raised from 1,024 to
2,048 so that truncation does not stand in for failure on code and multi-step math. A cap hit is a
failure. Because the think cap could make larger models look worse than they are, a cap-sensitivity stage
reruns prompts that hit the cap with an 8,192-token cap, and the paper reports how many oracle labels
change.

### 13.2 Open-ended family and judge (downloads approved 2026-09-12)

Source: `databricks/databricks-dolly-15k` (CC BY-SA 3.0, 12.5 MB), stratified by category, with the
human-written responses as references. Judge: `llama3.1:8b-instruct-q4_K_M` (4.58 GiB, which `ollama list` shows as 4.9 GB; out-of-family), run
after labelling and never loaded alongside a routed model (§5.4). Indrajeet approved both downloads on
2026-09-12 and confirmed the judge on 2026-09-13; it is already present in Ollama. The family is added by enabling `dolly` in `configs/corpus.yaml`. Each source draws from its
own seeded stream, so the 1,200 gradable prompts should not change; the rebuilt manifest is checked against
the existing one. Until the family's labels and the judge validation exist, every result states that the
open-ended family is not yet included.

### 13.3 Energy windows, from the smoke stage

The smoke stage (6 prompts at 0.6B and 1.7B) showed:

- Reported power takes about 5 s to decay after generation stops. The tail of each window is therefore
  6 s, not the 2 s planned in §6.2, so a window's decay is not charged to the next window.
- One 588 W sample, between 1.2 W neighbours, appeared as telemetry started. The GPU cannot draw that, so
  readings above 200 W are dropped as read errors.
- Idle power with the model loaded is about 1.5 W once settled. The baseline is measured for 60 s after
  warm-up and 30 s at the end of each model session, and dynamic energy subtracts their mean.
- A window is rerun if utilisation at the end of its tail shows another workload, or if any generation in
  it had to reload the model.

### 13.4 Resolving the open decisions of §12

**`p_accept` / `p_retry` (§3).** Swept over a grid and labelled as assumed. The headline tax figure uses
oracle detection, which is the conservative case for the baseline. One point, `p_accept = 0.15` and
`p_retry = 0.05`, is shown as an illustration only.

**Recovery after a router misroute.** Primary: a failed routed answer escalates straight to c8. Sensitivity:
it continues up the frugal-climb ladder (P1) from the routed cell.

**Reactive cascade baseline.** AutoMix-style: the small model answers, the same model self-verifies, and the
query escalates when verification confidence is below a threshold tuned on the validation split. The POMDP
meta-verifier of `aggarwal2024automix` is omitted, which favours no method, and this is stated.

**Starting distribution of the sticky-mixed policy (P4).** Uniform over cells as the primary; a skewed
distribution with 50% of queries starting at c8 as a sensitivity. Both are labelled as assumptions.

### 13.5 Public-benchmark arm, stage 5 (run 2026-09-12)

The source is LLMRouterBench (`NPULH/LLMRouterBench`, MIT licence), read from the local Hugging Face
cache at revision `0e5af1b8`; `bench-release.tar.gz` has SHA-256 `b79f8cde...f648c8f1`. Nothing is
downloaded and nothing is generated: `scripts/build_public_arm.py` streams the archive, keeps the outcome
fields of the pool and writes `data/processed/public_arm_records.csv.gz` plus a manifest naming the
subset chosen per dataset.

- **Pool and queries.** The thirteen models of the performance-cost setting, over the nine datasets all of
  them were run on. `tau2` is the tenth and is dropped: `gpt-5-chat` was never run on it, and the pool has
  to stay fixed across datasets. Where a dataset ships several evaluated subsets, the largest one every
  pool model shares is taken, identified by the release's own `data_fingerprint`. That gives 12,168
  queries and 158,184 records.
- **Success.** A full score. Scores are binary except on ArenaHard, where a judged tie against the baseline
  scores 0.5; a tie is not a correct answer. The 0.5 threshold is run as a sensitivity and moves nothing.
- **Cost.** Dollars, as metered by the providers, and prompt plus completion tokens. No per-query latency
  and no energy is published for these pools, so those units are not reported here.
- **Ladder.** The grid's size-and-mode order has no counterpart in a pool of thirteen models, so the ladder
  is the cost-accuracy frontier of the pool in cost order: `deepseek-v3-0324`, `qwen3-235b-a22b-2507`,
  `gpt-5`, `gemini-2.5-pro`. Its top rung takes the role of c8. P1, P3 and P4 carry over; P2 does not,
  the pool having no mode axis, and the two Qwen3-235B-A22B releases are compared instead.
- **Caveat.** The frontier is a subset of the pool, so a climbing policy can fail where an off-frontier
  model would have answered (29.8% of queries are answered most cheaply off the frontier). Such a policy
  can show a negative tax by underpaying and failing, so the tax is reported beside the success rate and
  cost per solved query carries the comparison.
- **Bootstrap.** 2,000 resamples rather than 10,000: the prompt set is forty times the local one, and the
  resample matrix is what bounds memory. `analysis._mean` and `analysis._ratio` now consume that matrix in
  blocks, which changes no statistic.
- **Result.** P3 costs 7.96x the minimum sufficient cost and answers 58.8% of queries; P1 costs 2.69x and
  answers 76.2%; the oracle answers 82.0%. Per solved query, $102.08, $57.38 and $25.47 per 1,000. The
  tax survives the move to a frontier pool.

### 13.6 Cap, noise and open-ended stages (run from 2026-09-13)

The local grid's main stage finished on 2026-09-13. The remaining GPU stages run back to back, in the
order cap, noise, the open-ended family (pilot, then main), the judge, and the probes, after a restart of
the machine. What changed across that restart, and what did not:

- **Driver.** 616.56 for gate 1, the pilot and the main stage; 616.92 from the cap stage on. The update
  arrived with the restart and was not chosen. Every run record carries its driver version, and the noise
  stage re-samples prompts and cells the main stage measured, so it doubles as a check that energy per
  token did not move with the driver.
- **Unchanged.** Ollama 0.34.0, started from the same Startup shortcut with no `OLLAMA_*` variables set;
  the four Qwen3 digests, compared with those recorded at the main stage; and the 1,200 gradable prompts,
  which the corpus rebuild that adds Dolly reproduces record for record.
- **Cap reruns.** 219 capped generations, up to 40 per cell and all of them where a cell had fewer
  (`scripts/select_cap_reruns.py --per-cell 40` on the full 9,600-generation grades), bounded at 5.7 GPU
  hours. Thinking hit the 4,096-token cap on 16-20% of prompts at every size and non-thinking on about 1%.

## 14. Hosted-frontier arm, stage 6 (run 2026-09-12)

The local grid stops at 8B and the public arm (§13.5) has no mode axis. This arm restores both at once: a
hosted ladder whose rungs span 25x in price, crossed with a reasoning-effort axis, over prompts the local
run has already labelled, so each hosted `c*` is paired with a local one. No GPU is used and nothing is
downloaded.

**Spend.** $8.00 authorised by the repository owner on 2026-09-12, with $0.25 of that held as reserve and
never planned against. `arbiter.hosted.pricing.Meter` keeps a JSONL ledger (`results/processed/hosted_arm/
ledger.jsonl`) and refuses to submit a batch whose **worst case** -- every request emitting its full output
cap -- would carry the cumulative total past the cap. An in-flight batch is charged at its worst case until
its actual cost is known, so the guard holds across invocations and cannot be defeated by a crash. Actual
cost is computed from the provider's own reported token counts, never from an estimate.

- **Pool.** Six cells, `h1`..`h6`, size-major and mode-minor as `c1`..`c8` are: `gpt-5-nano`, `gpt-5-mini`
  and `gpt-5`, each at `minimal` and `high` reasoning effort. List prices per million tokens, read from
  developers.openai.com/api/docs/pricing on 2026-09-12, are $0.05/$0.40, $0.25/$2.00 and $1.25/$10.00 for
  input and output; the ladder therefore spans a factor of 25 in price against the local grid's much
  narrower spread.
- **Requests.** One user message carrying exactly the prompt text the local pool saw, no system message.
  Caps mirror the local protocol: 4,096 output tokens at high effort and 2,048 at minimal. On a reasoning
  model `max_output_tokens` counts reasoning tokens, so the cap bounds the same thing the local cap bounds,
  the trace included. A truncated response is a failure, as a local cap hit is.
- **Two accuracies, because the cap binds hardest where reasoning is dearest.** A high-effort
  generation spends most of its output budget on reasoning tokens, so the 4,096 cap is a real
  constraint there and none at all at minimal effort, where the trace is empty. Each cell is
  therefore reported twice: the headline accuracy, in which a truncated generation is wrong, and
  the accuracy over the generations that stopped on their own. The first is what a user of that
  cell would actually get; the second says whether the cell answered badly or merely ran out of
  room. Without the pair they are indistinguishable, and the local grid's cap-sensitivity stage
  (§13.1) has no hosted counterpart the budget can afford -- re-running a cell at a larger cap
  costs money, whereas conditioning on the generations that finished costs nothing.
- **Not a matched comparison of decoding.** The reasoning models reject the temperature, top-p, top-k and
  min-p the local protocol sets per mode, so sampling is the provider's default. The arm claims a matched
  *prompt* and a matched *cap*, not matched decoding.
- **The effort axis is close to the mode axis, but is not the same object.** `/think` and `/no_think` are a
  binary switch on one model's chat template; `reasoning_effort` is a four-valued knob the provider owns.
  Measured on the pilot, the two ends behave alike: across all 120 `minimal` responses, every one of the
  three models reported exactly zero reasoning tokens, so `minimal` is reasoning off in practice rather
  than merely reasoning reduced. That is a stronger analogy than anticipated before the run, and it is
  reported as a measurement, not assumed: the setting is the provider's and could change under the same
  name. The paper states this rather than treating the two axes as interchangeable.
- **Batching.** Requests go through the Batch API at 50% of list price, a 24 h completion window. The
  provider refuses a batch naming more than one model, so a stage runs as one batch per model, each
  carrying that model's two efforts -- grouped this way rather than one batch per cell so that a size is
  never half-run. The budget is checked once against the whole stage before any of its batches is sent.
- **Pairing.** The hosted prompts come from the local main subset, so each prompt carries a `c*` from
  both ladders. The two differ by orders of magnitude in absolute price, so they are compared by rank,
  not by cell: which size step a prompt needs on its own ladder, and whether it needs the costly mode.
  Both ladders are size-major and mode-minor, so one rank reading serves for `c1`..`c8` and `h1`..`h6`.
  The pairing also counts what the hosted ladder rescues -- prompts no local cell answers, and the
  hosted rung that does -- which is the one thing the local grid cannot report about itself. The local
  per-prompt `c*` is read from the analysis' own `splits.csv`; a unit that table does not carry fails
  loudly rather than joining to nothing and looking like agreement.
- **Policies.** `P1`, `P3`, `P4` and `random` are the public arm's, built from the pool's own cost
  frontier. On the 40-prompt pilot that frontier held only minimal-effort rungs; on the main stage it
  admits one high-effort rung, `h4` (see §14.2). `P2` is not discoverable from the frontier either way
  and is named instead: start at the cheapest model in the cheap mode, then take
  the costly mode at each size in turn. The rungs are derived from the pool's shape rather than
  hard-coded, and the derivation reproduces the local grid's `P2` exactly when given eight cells, so
  the two arms report the same policy rather than two policies with one name.
- **Grading.** The local graders, unchanged: `arbiter.eval.grading.grade` reads the hosted record because
  the record carries `content` and `done_reason` under the local names. Analysis reuses
  `arbiter.eval.public` in full, so pool summary, `c*` distribution, policy table and mode comparison are
  computed by exactly the code that produced them for the public arm.
- **Losses against the local grid.** The provider returns the reasoning-token *count* but not the trace, so
  the leak-rate and marker-density checks of §10 have no counterpart here. There is no energy: the arm
  reports dollars and tokens only.

| Stage | Prompts | Requests | Worst case | Actual |
|-------|---------|----------|-----------|--------|
| smoke | 1 | 1 | $0.0004 | $0.00002 |
| pilot | 40 | 240 | $1.5334 | $0.2639 |
| main | 827 in 10 waves | 4,962 | $31.6978 | $6.8066 |
| **total** | **868** | **5,203** | | **$7.0706** of $8.00 |

The actual column is the sum of the per-batch costs the ledger records from each batch's own returned
usage, over 35 settled batches with none outstanding. Indrajeet reconciled it against the provider's own
usage figure for the key on 2026-09-13, which read $7.07: the two agree to the cent, so the cost this
paper reports is not a self-accounted number alone.

The pilot exists to measure what `high` effort actually spends, which is the one quantity the local run
cannot supply; the main batch is sized from its measured cost per prompt, not from a projection.

### 14.1 Pilot result (run 2026-09-12)

40 prompts x 6 cells = 240 responses, 0 errors, 5 cap hits (all at `h2`). Cost **$0.2639** against a
worst case of $1.5334, that is 17.2% of the bound, so the guard is sound but loose -- which is what
motivated the wave campaign in 14.2.

| cell | model | effort | accuracy | output tok | reasoning tok | cap hits | $/1,000 prompts |
|------|-------|--------|---------:|-----------:|--------------:|---------:|----------------:|
| h1 | gpt-5-nano | minimal | 62.5% | 108 | 0 | 0.0% | 0.02 |
| h2 | gpt-5-nano | high | 85.0% | 1,738 | 1,664 | 12.5% | 0.35 |
| h3 | gpt-5-mini | minimal | 97.5% | 147 | 0 | 0.0% | 0.16 |
| h4 | gpt-5-mini | high | 97.5% | 824 | 675 | 0.0% | 0.84 |
| h5 | gpt-5 | minimal | 100.0% | 132 | 0 | 0.0% | 0.73 |
| h6 | gpt-5 | high | 100.0% | 887 | 760 | 0.0% | 4.50 |

Three findings, all provisional on 40 prompts:

1. **`minimal` emits zero reasoning tokens** on all three models, over 120 responses. See the corrected
   caveat in 14.
2. **The cost-accuracy ladder is `h1 -> h3 -> h5`: every rung is a minimal-effort cell.** High effort is
   off the frontier at all three sizes. `h3` beats `h2` on both cost and accuracy; `h6` costs 6.2x `h5`
   for no accuracy gain. 92.5% of prompts are cheapest-solved at a minimal cell, against 88.3% no-think
   on the local grid -- the local headline replicates on hosted frontier models.
3. **Relative tax: P1 0.34x, P3 10.60x, P4-uniform 17.41x, random 17.81x.** Same ordering as the local
   grid, wider spread, as expected from a ladder spanning 25x in price rather than ~13x in parameters.

Weakness: `gpt-5` answers 100% of these 40 prompts, so the top of the ladder is untested and no prompt
is unservable. The main run exists to find the prompts that need the upper rungs.

### 14.2 Wave campaign (main stage, run 2026-09-12)

The guard approves only the worst case, but measured generations came in at 17% of it, so a single
worst-case-sized batch would have bought 195 prompts and left most of the budget unusable. The main
stage therefore runs as a sequence of waves: each wave is sized to the worst case the *remaining*
budget allows, submitted under the same check, and then settled at its measured cost, which frees the
difference for the next wave. The guard is never loosened -- at every instant, committed spend is the
worst case of everything unsettled.

Waves share one records file and one ledger, and each wave's prompts are withheld from later waves by
prompt id, so the campaign is resumable: an interrupted run resumes from what its own input files show
was already paid for. The campaign stops when the budget left would fund a wave below a floor of 20
prompts, so that a final wave is never too small to be worth a batch round trip.

**Outcome.** Ten waves, 827 of the 860 eligible prompts, $6.8066 spent against a worst case of
$31.6978 -- 21.5% of the bound, close to the pilot's 17.2%. Wave sizes fall geometrically as the
remaining budget shrinks: 195, 153, 121, 94, 73, 58, 47, 36, 27, 23. Wave 11 drew 21 prompts, of which
the budget allowed 17, below the floor, and the campaign stopped with $0.9294 unspent. The 33 prompts
never submitted are the price of the guard, and the paper says so rather than implying full coverage.

4,960 responses were returned of 4,962 requested. The two missing are content-filter refusals, one at
`h3` (`math500-204`) and one at `h5` (`mbpp-66`); both prompts are dropped whole, because a refusal is
not a wrong answer, leaving **825 prompts** in the analysis. 243 responses hit their cap. No request
errored.

Results, over 825 prompts (`results/processed/hosted_arm/main/`):

| cell | model | effort | accuracy | acc. finished | output tok | reasoning tok | cap hits | $/1,000 |
|------|-------|--------|---------:|--------------:|-----------:|--------------:|---------:|--------:|
| h1 | gpt-5-nano | minimal | 74.2% | 74.5% | 142 | 0 | 0.5% | 0.03 |
| h2 | gpt-5-nano | high | 77.8% | 96.0% | 1,908 | 1,845 | 18.9% | 0.38 |
| h3 | gpt-5-mini | minimal | 89.5% | 89.6% | 168 | 0 | 0.1% | 0.18 |
| h4 | gpt-5-mini | high | 92.8% | 96.5% | 1,010 | 886 | 3.8% | 1.02 |
| h5 | gpt-5 | minimal | 91.8% | 91.9% | 146 | 0 | 0.1% | 0.79 |
| h6 | gpt-5 | high | 91.9% | 97.3% | 1,140 | 1,042 | 5.6% | 5.77 |

1. **The ladder is `h1 -> h3 -> h5 -> h4`.** High effort earns a rung at the `mini` size on 825 prompts,
   which it did not on the pilot's 40 -- the pilot was too easy to separate `h3` from `h4`. `h2` and
   `h6` stay off the frontier. `h6` costs 5.6x `h4` and answers 0.9 points *fewer* (91.9% vs 92.8%):
   **the costliest cell in the pool is not the most accurate.**
2. **Relative tax: P1 0.22x, P2 1.41x, P3 4.35x, P4-uniform 6.93x, random 7.24x.** Same ordering as the
   pilot and the local grid, much narrower spread, because the pilot's 40 easy prompts made the oracle
   unrealistically cheap. Per solved query: oracle $0.25/1,000, P1 $0.35, P3 $1.10.
3. **The pilot's weakness is resolved.** 2.5% of prompts are unservable by all six cells, and the upper
   rungs are the cheapest sufficient choice for a measurable share, so the top of the ladder is tested.
   `c*` in dollars: h1 74.2%, h3 16.0%, h2 3.4%, h5 2.2%, h4 1.3%, h6 0.4%.
4. **Effort is priced as a multiple, and what it buys shrinks with model strength.** High costs 11.8x,
   5.45x and 7.26x minimal at nano, mini and gpt-5, while the share of prompts minimal fails and high
   answers is 15.2%, 6.2% and 4.1%. At `gpt-5` the headline accuracy moves 91.8% -> 91.9% and *falls*
   on mcq, 94.9% -> 93.4%.
5. **The cap binds hardest where reasoning is dearest, exactly as §14 anticipated.** `h2` hits its cap
   on 18.9% of prompts and answers 96.0% of the generations that finish against a 77.8% headline. On
   code, where its reasoning runs longest (3,075 output tokens), high effort is worse than minimal
   outright, 65.3% vs 82.5%.

The local pairing is not yet computed: it needs the local main analysis, which needs the labelling grid
to finish. `run_hosted_analysis.py --stage main --local <dir>` fills it in; `summary.json` currently
carries `"pairing": null`.
