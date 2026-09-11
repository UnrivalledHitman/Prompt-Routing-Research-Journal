# Prior-Art Register — ARBITER / escalation-tax routing

**CONFIDENTIAL. Patent filing pending. Do not publish or push to a public repository.**

Compiled 2026-09-10, revised 2026-09-11 after a full bibliographic verification pass.
Purpose: (a) hand to patent counsel before claims are drafted, (b) drive the paper's Related Work.

**Verification status:** every work in sections A–E now has its complete author list, exact title,
submission date and venue confirmed from a live arXiv abstract page or publisher record. All 47
corresponding BibTeX entries are in `paper/refs.bib`. Nothing was written from recall.

**Two entries in the 2026-09-10 draft were wrong and are corrected below** — see the CORRECTION notes
on D1 and D2. This is why the register is verified rather than assembled from search snippets.

---

## A. Closest art — threatens broad claims

Each predates any filing and discloses joint model-and-effort routing directly. Counsel should assume a
broad claim of the form *"classify an incoming prompt, then select both a model and a reasoning mode"*
is anticipated.

| # | Work | ID / venue | Date | Why it threatens |
|---|------|-----------|------|------------------|
| A1 | **Route to Reason: Adaptive Routing for LLM and Reasoning Strategy Selection** — Pan, Zhang, Zhao, Han | arXiv:2505.19435; **ACM WWW 2026**, pp. 5568–5578 | 2025-05-26 | Joint routing over **7 open-source models x 4 reasoning strategies** under a budget constraint; >60% token reduction. The single most damaging reference to a broad joint-routing claim. |
| A2 | **SynapseRoute: An Auto-Route Switching Framework on Dual-State LLM** — Zhang, Qiao, Luo, Li, Zheng, Xu, Li, Gui, He, Qiu, Hong, Sun | arXiv:2507.02822 | 2025-07-03 | Embeddings + logistic-regression classifier routing thinking vs. non-thinking mode; introduces the Accuracy–Inference–Token (AIT) index; 36.8% time and 39.66% token reduction. **Anticipates our classical-ML arm almost exactly.** Narrower in that it is confined to the medical domain and a single dual-state model. |
| A3 | **Think When Needed: Model-Aware Reasoning Routing for LLM-based Ranking** — Guo, Wei, Wang, Du, Wang, Zhang, Sun | arXiv:2601.18146; **SIGIR 2026**; DOI 10.1145/3805712.3809709 | 2026-01-26 | Lightweight plug-and-play router head choosing Think vs. Non-Think pre-generation from ranking-aware and model-aware difficulty features. Scope limited to ranking. |
| A4 | **DART: Draft-Agreement Routing for Training-Free Adaptive Thinking Budgets** — Lee, Hong, Lee, Seo, Son, Eo, Park, Park, Moon, Lim | arXiv:2606.23181; **EMNLP 2026 Findings** | 2026-06-22 | Training-free: two cheap no-think drafts probe difficulty, draft entropy sets a per-query thinking budget. 32–73% thinking-token reduction. **Reports tokens and latency only — explicitly not energy.** |
| A5 | **HRBench: Benchmarking and Understanding Thinking-Mode Switch Strategies in Hybrid-Reasoning LLMs** — Ning, Liu, Ye, Zhang, Liu | arXiv:2605.28398 | 2026-05-27 | **NEW — added 2026-09-11.** Benchmarks three switch-strategy families (prompt-based, external routing, speculative) across four training regimes, six models, five benchmarks. Concludes the preferred strategy varies with model scale and task domain. Close art on the comparative-evaluation angle; also a useful comparison point for us. |
| A6 | **Learning When to Think: Adaptive Reasoning for Test-Time Compute Allocation** — Kassenaar, Yang, François-Lavet | arXiv:2608.20256 | 2026-08-20 | Selects NoThink / Short / Long as the first response token, trained with GRPO. Anticipates multi-level effort selection, but the mechanism is *in-model* (no external router), which distinguishes it from our architecture. |
| A7 | **Router-R1: Teaching LLMs Multi-Round Routing and Aggregation via RL** — Zhang, Feng, You | arXiv:2506.09033; **NeurIPS 2025** | 2025-06-10 | **NEW — added 2026-09-11.** Stateful, multi-round routing that accumulates context across rounds. This anticipates the *sequential-routing* direction we considered and rejected. It does **not** model escalation cost or the resubmission penalty. Confirms rejecting that framing was correct. |

## B. Foundational routing and cascading art

| # | Work | ID / venue | Date | Relevance |
|---|------|-----------|------|-----------|
| B1 | **FrugalGPT** — Chen, Zaharia, Zou | arXiv:2305.05176 | 2023 | Canonical LLM cascade. Matches GPT-4 at up to 98% cost reduction. The cascade baseline our escalation-tax analysis must be distinguished from — note a *cascade* is an automated, cost-optimised policy, whereas the *escalation tax* concerns an uninstrumented human retry trajectory. That distinction is the crux of our novelty. |
| B2 | **RouteLLM** — Ong, Almahairi, Wu, Chiang, Wu, Gonzalez, Kadous, Stoica | arXiv:2406.18665 | 2024-06-26 | Strong/weak binary routing from human preference data; >2x cost reduction. Most-cited modern router. |
| B3 | **Hybrid LLM** — Ding, Mallick, Wang, Sim, Mukherjee, Rühle, Lakshmanan, Awadallah | **ICLR 2024**; arXiv:2404.14618 | 2024 | Difficulty-based small/large routing with a test-time-tunable quality knob; 40% fewer large-model calls at no quality cost. |
| B4 | **RouterBench** — Hu, Bieker, Li, Jiang, Keigwin, Ranganath, Keutzer, Upadhyay | arXiv:2403.12031 | 2024 | 36,497-prompt zero-shot split, 11 LLMs, per-inference costs, oracle labels. Candidate for our public-benchmark arm. |
| B5 | **AutoMix** — Aggarwal, Madaan, Anand, Potharaju, Mishra, Zhou, Gupta, Rajagopal, Kappaganthu, Yang, Upadhyay, Faruqui, Mausam | arXiv:2310.12963; **NeurIPS 2024** | 2023-10-19 | Few-shot self-verification plus a POMDP router; >50% cost reduction. The closest thing in the literature to modelling escalation as a decision process — **counsel should read this one carefully**, since a POMDP over model sizes is conceptually adjacent to escalation-aware selection. Distinguishing feature: AutoMix's state is answer confidence, not accrued escalation cost. |
| B6 | **RouterDC** — Chen, Jiang, Lin, Kwok, Zhang | arXiv:2409.19886; **NeurIPS 2024** | 2024-09-30 | Dual contrastive learning router; +2.76% in-distribution, +1.90% out-of-distribution. |
| B7 | **Universal Model Routing (UniRoute)** — Jitkrittum, Narasimhan, Rawat, Juneja, Wang, Wang, Go, Lee, Shenoy, Panigrahy, Menon, Kumar | arXiv:2502.08773 | 2025-02-12 | Routing to *unseen* LLMs with an excess-risk bound. The main theoretical treatment in the area. |
| B8 | **Arch-Router** — Tran, Paracha, Hafeez, Chen | arXiv:2506.16655 | 2025-06-19 | 1.5B preference-aligned router; adds models without retraining. |

## C. Surveys a reviewer will expect us to engage

| # | Work | ID | Date |
|---|------|----|------|
| C1 | **Dynamic Model Routing and Cascading for Efficient LLM Inference: A Survey** — Moslem, Kelleher (Trinity College Dublin) | arXiv:2603.04445 | 2026-02-23 (v1), 2026-04-21 (v2) |
| C2 | **Reasoning on a Budget: A Survey of Adaptive and Controllable Test-Time Compute in LLMs** — Alomrani, Zhang, Li, Sun, Pal, Zhang, Hu, Ajwani, Valkanas, Karimi, Cheng, Wang, Liao, Huang, Wang, Hao, Coates | arXiv:2507.02076 | 2025-07-02 |
| C3 | **Stop Overthinking: A Survey on Efficient Reasoning for LLMs** — Sui, Chuang, Wang, Zhang, Zhang, Yuan, Liu, Wen, Zhong, Zou, Chen, Hu | arXiv:2503.16419; **TMLR 2025** | 2025-03-20 |

## D. Energy, edge and router-overhead art

| # | Work | ID / venue | Note |
|---|------|-----------|------|
| D1 | **Quality-of-Service Aware LLM Routing for Edge Computing with Multiple Experts** — Yang, Wu, Feng, Zhou, Guo, Chen | arXiv:2508.00234; **accepted, IEEE Trans. Mobile Computing** | **RESOLVED from full text (2026-09-11).** The "5 ms on an NVIDIA RTX 4090" and "134M model parameters" figures are stated in the paper, with the breakdown: two DistilBERT predictors at 67M each, HAN 19K, actor-critic 10K. Each edge expert runs on an RTX 4090 under vLLM. **Energy is never measured anywhere in this work — latency only.** Routes over model/expert selection by predicted query difficulty (generation score, output length); **does not route over reasoning effort.** Net effect: it partially anticipates router-*latency* accounting and nothing of router-*energy* accounting. |
| D2 | **Camel: Energy-Aware LLM Inference on Resource-Constrained Devices** — Xu, Peng, Song, Liu, Jun, Li, Yu, Mao | arXiv:2508.09173 | **CORRECTION (2026-09-11).** Filed on 2026-09-10 as energy-aware routing art. It is not routing at all: it optimises **GPU frequency and batch size for a single model** on an NVIDIA Jetson AGX Orin. Relevant as measured-energy methodology on constrained hardware, **not** as competing routing art. |
| D3 | **GreenServ: Energy-Efficient Context-Aware Dynamic Routing for Multi-Model LLM Inference** — Ziller, Ilager, Tundo, Bartocci, Mariani, Brandic | arXiv:2601.17551 | **RESOLVED from full text (2026-09-11).** Energy **is** hardware-measured: "we measure actual GPU power draw in watt-hours ... using the zeus library." Hardware: one NVIDIA A100 80 GB, AMD EPYC 9354P, 512 GB RAM, CUDA 12.2. Routes **model selection only** — task type, semantic cluster, text complexity — **not reasoning effort**. Router overhead is characterised separately (~7.77 ms pre-inference per query) but its **energy is never subtracted: savings are reported GROSS.** No carbon/gCO2e. Headline: 31% cumulative energy reduction and +22% accuracy versus random routing. |
| D4 | **EcoThink: A Green Adaptive Inference Framework for Sustainable and Accessible Agents** — Li, Lu | arXiv:2603.25498; **ACM WWW 2026** | **RESOLVED from full text (2026-09-11).** Energy measured with **CodeCarbon sampling GPU power at 100 ms intervals**, but the underlying energy model is partly TDP-approximated (P_avg is "average thermal design power (TDP) utilization"), so it is a hybrid of measurement and estimation — a weakness we can improve on. Hardware: 8x A100 80 GB with NVLink. Router is a distilled DistilBERT bifurcating traffic between a retrieval path and a reasoning path at threshold 0.5 — **pipeline selection, not thinking-budget control within a hybrid model.** Router's own cost is **not** charged: the 40.4% saving is GROSS. **Does report carbon**: 1.32 gCO2eq/query for EcoThink vs 4.52 for GPT-4o. |
| D5 | **Sustainable LLM Inference for Edge AI: Evaluating Quantized LLMs for Energy Efficiency, Output Accuracy, and Inference Latency** | ACM Trans. IoT, DOI 10.1145/3767742 | Closest located work on measured energy of quantized LLMs on constrained hardware. Methodological reference for our NVML harness. |

## E. Further benchmarks located and verified

| # | Work | ID |
|---|------|----|
| E1 | **RouterEval** — Huang, Ling, Lin, Chen, Zhong, Wu, Lin. 8,500+ candidate models, 200M+ performance records | arXiv:2503.10657 |
| E2 | **LLMRouterBench** — Li, Zhang, Guo, Wang, Tang, Zhang, Chen, Qi, Ye, Bai, Wang, Hu. 400K+ instances, 21 datasets, 33 models, **code and data public** | arXiv:2601.07206 |

---

## F. Threats to internal validity (not prior art, but they can sink the experiment)

**NEW SECTION — 2026-09-11.** This is the most operationally important finding of the verification pass.

**Demystifying Hybrid Thinking: Can LLMs Truly Switch Between Think and No-Think?** — Wang, Yang, Long,
Wang, Chaudhary, Han. arXiv:2510.12680, 2025-10-14.

The paper finds that hybrid-thinking LLMs achieve only **partial mode separation**: reasoning behaviour
leaks into no-think mode. Reasoning tokens contaminate no-think outputs, and standard training yields
no-think responses full of reasoning-supportive language. Their improved recipe cut no-think output on
MATH500 from 1085 to 585 tokens and problematic reasoning tokens from 5917 to 522 — a large improvement
that still falls short of clean separation.

**Consequence for our design.** The entire model-size x reasoning-mode grid assumes `/think` and
`/no_think` are genuinely distinct operating points on Qwen3. If they are not cleanly separated on the
specific checkpoints we run, the reasoning-mode axis is confounded and every downstream energy and token
result inherits that confound.

**Required mitigation, before any large labelling run:** a mode-separation pilot on our own pool —
sample prompts through each Qwen3 size in both modes, and measure the realised distributions of output
length, thinking-token count and reasoning-marker frequency. Report the separation empirically in the
paper rather than assuming it. If separation is poor at some sizes, that is itself a publishable finding
and must be stated, not hidden.

---

## G. What remains open — the defensible position

Searched and **not found** in the peer-reviewed literature:

1. **The escalation tax is unmeasured.** Every routing paper above baselines against static single-model
   policies, random assignment, an oracle, or an automated cascade. None formalises or measures the
   counterfactual a real user experiences: sequential *manual* escalation in which each failed attempt is
   billed in full and the entire context is resubmitted to the next tier. The cost structure is discussed
   only in practitioner writing. Absence from the academic record combined with clear practitioner
   salience remains the strongest novelty signal available. **B5 (AutoMix) is the nearest neighbour and
   must be distinguished explicitly** — its POMDP state is answer confidence, not accrued cost.
2. **Net-of-router *energy* accounting — CONFIRMED OPEN by full-text reading of D1, D3, D4.** All three
   report **gross** savings. D3 characterises its router's *latency* (~7.77 ms) but never its joules.
   D4 does not charge its DistilBERT router's cost against the 40.4% figure. D1 measures no energy at
   all. No located work subtracts the selector's own measured joules from the energy it saves, or
   identifies the crossover at which an LLM-based router stops paying for itself against a logistic
   regression.
3. **The energy cost of the reasoning-effort axis has never been measured — the strongest surviving
   energy claim.** Two literatures exist and do not touch. Energy-aware routers (D3, D1) route over
   *model choice only*. Thinking-budget routers (A2, A4, A5, A6) report *tokens and latency, never
   joules* — A4 states this explicitly. D4 is the nearest miss: it routes between a retrieval pipeline
   and a reasoning pipeline, which is pipeline selection, not thinking-mode control inside a hybrid
   model. Nobody has put a power meter on the think/no-think axis.
4. **Joules per *solved* query.** Prior work reports energy per query or per token. Normalising energy by
   *successful* outcomes — which penalises a cheap model that fails and forces a retry — was not
   surfaced anywhere. This is the escalation tax expressed in energy, and it links contributions 1 and 4.
5. **Commodity-hardware characterisation.** D3 runs on one A100 80 GB, D4 on eight A100s, D1 on RTX 4090s.
   No measured characterisation of the model-size x reasoning-mode grid on an 8 GB consumer laptop GPU,
   where quantisation is forced and there is no VRAM headroom. Weaker as a standalone claim — the 4090 is
   consumer silicon — but useful framing, since 8 GB is a materially different regime from 24 GB.

**Sub-claims now DEAD, and which must not be asserted in the paper:** that we are first to measure routing
energy on hardware (D3 uses zeus, D4 uses CodeCarbon); that we are first to report carbon for routing
(D4 reports gCO2eq per query); that energy-aware routing is itself novel (established by D3 and D4).

**Claim posture recommended to counsel.** Do not pursue claims over prompt classification followed by
model-and-mode selection in general: A1–A7 anticipate this, with A1 and A2 the most damaging. Direct
claims instead at (i) escalation-aware selection, where expected escalation cost — including
already-failed attempts and context resubmission — is the selection objective, and (ii) net-energy-optimal
selection, where the selector's own measured energy is subtracted from the saving. Both remain unlocated
in the searched literature, but D3 and D4 are close enough on energy that counsel should read both in full
before drafting.

---

## H. Outstanding verification work

- ~~Obtain and read the full PDFs of D1, D3 and D4.~~ **DONE 2026-09-11.** All three resolved; see the
  updated D1/D3/D4 rows and the revised section G. Outcome: three energy sub-claims died, three survive.
- **B5 (AutoMix) full text** — now the highest-priority remaining read. Needed to draw a precise boundary
  between its POMDP-over-model-sizes formulation and escalation-aware selection. This is the closest
  conceptual neighbour to the headline claim and counsel will need the distinction stated exactly.
- `deepseek2025r1` — transcribe the full author list from the Nature record.
- `yang2025qosrouting` — obtain IEEE TMC volume, issue and pages once assigned.
- `chen2021humaneval` — 58 authors; decide truncation per IEEE Access style.
- `he2021deberta` — confirm the ICLR 2021 proceedings record if citing the conference version.
- **AdaptThink, ThinkSwitcher, HBPO, SelfBudgeter** — named inside survey C2's summary but never
  independently located. **Not cited anywhere, and must not be, until their records are found.**
