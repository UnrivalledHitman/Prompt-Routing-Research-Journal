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
the pool**. **[DECISION]** Candidate: a Llama- or Gemma-family instruct model at 4-bit that fits the
8 GB budget. To be fixed once VRAM headroom is measured against the largest routed cell.

Scoring is a rubric on a 1-5 scale against a reference where one exists, with a success threshold of >= 4.
Position and verbosity biases are mitigated by randomising presentation order and by scoring single
responses rather than pairs.

**Judge validation.** Cheap judges are the standard attack surface for this kind of paper, so the judge is
itself evaluated:
1. A stratified sample of ~200 judged items is graded by the frontier OpenAI judge, and agreement is
   reported as Cohen's kappa. This is what the ~INR 1000 credit is for; it is enough for this and not for
   bulk grading.
2. Indrajeet hand-labels ~100 items blind, giving a human anchor and a judge-human kappa.

If kappa against the human anchor is poor, the open-ended family is reported with an explicit caveat or
demoted to a secondary analysis. That decision is made before seeing routing results, and recorded here,
so it cannot become a post-hoc rescue.

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

**Carbon.** `gCO2e = (joules / 3.6e6) * grid_intensity`, with the grid intensity factor and its source
stated explicitly, and results reported across a range of regional factors rather than one number. This
follows the accounting style of `luccioni2024powerhungry` and `lacoste2019quantifying`.

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
| 0b | Mode-separation validation (§10) | ~1-2 h | **Blocks the whole design** |
| 0c | Per-cell throughput calibration, budget re-estimate | ~1 h | — |
| 1 | Pilot labelling, 800 prompts x 8 cells x k=3 | ~8-12 h | Validates the pipeline end to end |
| 2 | Router training and evaluation on pilot | ~2-4 h | First real numbers |
| 3 | Full labelling, 3,200 prompts | ~35-50 h, overnight | Only after phase 1 is clean |
| 4 | Full router sweep, ablations, seeds | ~6-10 h | — |
| 5 | Public-benchmark arm (no local inference) | CPU only | — |

Every GPU phase is requested individually, with expected VRAM and duration, before it starts.

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

1. **Open-ended dataset** — source, licence, redistributability. Blocks §4.
2. **Judge model identity** — pending VRAM headroom measurement. Blocks §5.4.
3. **`p_accept` / `p_retry`** — cite, measure, or sweep and label as assumed. Blocks the headline tax number.
4. **Title-block details** — author name, institution, department, city/country, mentor's co-authorship
   and ORCID. Needed for the LaTeX template, not for any experiment.
