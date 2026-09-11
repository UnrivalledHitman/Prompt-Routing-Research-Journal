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
| 0b | Mode-separation validation (§10) | 1 h 54 min (actual) | **Passed 2026-09-11**, see §10.1 |
| 0c | Per-cell throughput calibration, budget re-estimate | answered by gate 2 data | Budget re-estimated in §10.1 |
| 1 | Pilot labelling, 800 prompts x 8 cells x k=3 | planned ~8-12 h; **measured rate implies ~75 h** | Superseded by §13 |
| 1' | Smoke stage, 6 prompts x 0.6B/1.7B | 5 min (actual) | **Passed 2026-09-11**, §13.3 |
| 1'' | Pilot labelling, 300 gradable prompts x 8 cells x k=1 | ~9.3 h estimated from gate 2 rates | Started 2026-09-11 |
| 2 | Router training and evaluation on pilot | ~2-4 h | First real numbers |
| 3 | Main labelling, 900 further gradable prompts | ~28 h | After the pilot is clean |
| 3a | Noise subset, 75 prompts x 2 extra samples | ~4.7 h | — |
| 3b | Cap sensitivity, think cap 8192 on cap-hit prompts | ~5 h | — |
| 3c | Open-ended family, 400 prompts, plus judge | ~14 h + ~4.5 h | Needs download approval |
| 4 | Full router sweep, ablations, seeds | ~6-10 h | — |
| 5 | Public-benchmark arm (no local inference) | CPU only | Needs a non-pickle source |

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

1. **Open-ended dataset** — source, licence, redistributability. Blocks §4.
2. **Judge model identity** — pending VRAM headroom measurement. Blocks §5.4.
3. **`p_accept` / `p_retry`** — cite, measure, or sweep and label as assumed. Blocks the headline tax number.
4. **Title-block details** — author name, institution, department, city/country, mentor's co-authorship
   and ORCID. Needed for the LaTeX template, not for any experiment.

Items 1-3 are resolved provisionally in §13. Item 4 and a code/data availability statement remain with
Indrajeet and the mentor.

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

### 13.2 Open-ended family and judge (pending downloads)

Proposed source: `databricks/databricks-dolly-15k` (CC BY-SA 3.0, 12.5 MB), stratified by category,
human-written responses as references. Proposed judge: `llama3.1:8b-instruct-q4_K_M` (4.58 GB,
out-of-family). The judge runs after labelling, never loaded alongside a routed model. Neither is
downloaded until Indrajeet approves it. Until then the corpus has 1,200 gradable prompts, and every result
states that the open-ended family is not yet included.

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
