# Co-occurrence Hallucination Diagnostic

## Research Hypothesis

This project tests whether LVLMs are more likely to hallucinate an absent object when that object frequently co-occurs with the objects present in the image.

> **If a target object $A$ (absent from the image) has higher co-occurrence with the objects present in the image, hallucination occurs more easily.**

All experiments use LLaVA-1.5-7B and COCO 2017 (train2017 for co-occurrence statistics, val2017 for all model-facing evaluation), seed=42 throughout.

The project is organized as four phases, each building on the previous phase's result rather than restating it. Detailed numbers for any given stage live with that stage's own outputs/README (linked below); this file only tracks what exists, what it found at a headline level, and how it connects to the next phase.

```
Phase A  Behavioral effect (does it happen?)         Stage 1-8
Phase B  Representational locus (where does it live?) Stage 9-11 (+ Stage 5)
Phase C  Single-pair mechanistic case study            Stage 11 case study + cooccurrence_causal_coupling
Phase D  Debiasing feasibility (can it be removed?)     adversarial_functional_debiasing_pilot + adversarial_signal_debiasing_pilot
```

---

## Phase A — Behavioral Effect (Stage 1-8)

Population-level pipeline across many object-category pairs. No separate README per stage; this is the primary write-up for these results.

| Stage | What it tests | Method | Headline result |
|---|---|---|---|
| **1** | Existence of real object co-occurrence structure in COCO | PMI over train2017 (118,287 images), pairs with support < 10 excluded | `mouse–keyboard` PMI = 3.65 (lift 38.3×); structure confirmed |
| **2** | Isolation of the co-occurrence effect from confounds | High/Low co-occurrence groups (top/bottom 33% of mean PMI to present objects), Coarsened Exact Matching on marginal frequency / object area / CLIP similarity | 77,828 matched pairs, all covariates \|SMD\| < 0.04 after matching |
| **3** | Difficulty of *inducing* hallucination via attack | Targeted PGD attack on `Is there a {A}?`, minimum flip budget $\epsilon^*$ ($L_\infty$, 20 steps × 2 restarts, max $\epsilon=32/255$), 150 matched pairs (300 samples) | 3 mandatory sanity checks pass (attack ≫ random-noise flip rate) |
| **4** | Whether High co-occurrence needs a smaller $\epsilon^*$ than Low | Stratified Cox / Weibull AFT / paired bootstrap / McNemar on Stage 3's $\epsilon^*$, `pair_id` as stratum | Cox HR = 1.627 (p = 0.00325); ε\* ≈42% smaller under High; survives Holm correction |
| **5** | Linear decodability of the bias from the frozen visual encoder | Linear probe on CLIP visual features vs. a Y-only (present-objects) baseline, excess AUC | excess AUC = −0.023, 95% CI [−0.025, −0.020] — no additional information |
| **6** | Transfer of the yes/no-optimized attack to open-ended captioning | Reuse Stage 3's adversarial images, check target-mention rate in a free-form caption | Exploratory only — not yet controlled to this project's rigor bar; see `outputs/.../stage6_open_ended_transfer/`, not written up here |
| **7** | Reproduction of the Stage 3 effect with a *caption-targeted* attack | Structural clone of Stage 3, readout = target-category mention in a short-answer response instead of forced yes/no | Pipeline built and runs end-to-end; went through several margin-design pivots and observed run-to-run noise at small N — not yet a controlled result |
| **8** | Whether High vs. Low holds for the caption-attack $\epsilon^*$ | Same survival analysis as Stage 4, applied to Stage 7's $\epsilon^*$ | Same caveat as Stage 7 — the run completed but was judged not properly controlled; not reported as a finding here (see `outputs/.../stage8_survival_analysis_caption/` for the raw numbers) |

**Phase A summary:** the co-occurrence → hallucination effect is real and well-controlled for forced yes/no VQA (Stages 3-4), and does not reduce to visual-encoder-only information (Stage 5). Stages 6-8 explore whether the same effect holds for open-ended/caption-style readouts; those runs completed but are not reported as findings here — they went through enough mid-flight design pivots and small-N noise that they don't meet this project's bar for a controlled result (raw outputs are kept for reference, a rerun would be needed before writing up a conclusion).

---

## Phase B — Representational Locus (Stage 9-11, general population)

Moves from *behavior under attack* to *clean-image (ε=0) internal evidence*, locating where the bias actually lives rather than just whether it changes an attack budget.

| Stage | What it tests (paper label) | Method | Headline result |
|---|---|---|---|
| **9** (Exp1) | Whether High co-occurrence already raises clean target-positive evidence $s_T$ = logit(Yes) − logit(No), before any attack | Same 150 matched High/Low pairs as Stage 3, $s_T$ at $\epsilon=0$ | mean diff = +0.729, 95% CI [0.399, 1.061], Cohen's $d_z$ = 0.353, p < 0.001 |
| **10** (Exp2) | Whether, holding the image fixed, a *stronger* co-occurrence relationship to that specific image raises $s_T$ (within-image specificity, not just a between-image confound) | 50 images × 74 target categories (3,039 pairs), two-way fixed-effects model $s_T \sim \text{coocScore} + \text{imageFE} + \text{targetFE}$ | $\beta = 0.406$, 95% CI [0.290, 0.522], p = 6.6e-12, permutation null ≈ 0 |
| **11** (Exp3) | Localization of the layer at which this effect emerges | Logit lens (final RMSNorm + lm_head) applied to Stage 10's 3,039 pairs at every one of the 32 LLaVA decoder layers, refitting Stage 10's exact FE model per layer | Signal is ≈0 for layers 1-6, significantly negative for layers 7-12, sign-flips positive at layer 13, and plateaus at partial $r \approx 0.21$–$0.26$ from layer 16 through the final layer (32, which exactly reproduces Stage 10's $\beta$) |

**Phase B summary:** the effect is (a) already present with no attack at all, (b) specific to the image-target relationship and not a generic frequency confound, and (c) linearly decodable from the LLM's own decoder stream from the mid layers onward — in contrast to Stage 5's negative result for the frozen visual encoder. This localizes the effect to LLM decoder layers ≥13 and leads into Phase C's test of whether that representation is *causally used*.

---

## Phase C — Single-Pair Mechanistic Case Study

Everything above pools many object-category pairs. Phase C runs the same chain of tests end-to-end for **one concrete, visually inspectable pair** (`baseball bat` context → `sports ball` target), so every step can be sanity-checked by eye, then pushes past correlation into intervention.

### Stage 11 case study — `baseball bat → sports ball`

Own full write-up: [`outputs/CooccurrenceHallucinationDiagnostic/stage11_case_bat_ball/README.md`](../../../outputs/CooccurrenceHallucinationDiagnostic/stage11_case_bat_ball/README.md) (Exp0-Exp7). Summary:

| Exp | What it tests | Result |
|---|---|---|
| 0 | Pair strength check | PMI = 2.342, lift = 10.40, ranked 59th/5,338 pairs — **GO** |
| 1 | Whether bat context increases attack vulnerability | Median $\epsilon^*$ 0.0001148 (G10, bat+ball−) vs 0.0005203 (G00 control), Cox HR = 1.760, p = 0.024 |
| 2 | Whether clean $s_\text{ball}$ is already higher with bat present | Cohen's $d$ = 0.745 |
| 3 | Consistency between clean evidence and $\epsilon^*$ | Consistent (definitionally linked, reported as a consistency check, not independent evidence) |
| 4/4B | Effect of removing the bat region (counterfactual) on ball evidence, vs. a sham edit | Bat removal reduces unsupported ball evidence more than sham; visually audited |
| 5 | Layer localization of the bat→ball signal | Logit lens peaks at decoder layer 19 (Δ_bat_to_ball = 1.493 vs. Δ_sham = −0.030) |
| 6 | Causal effect of editing layer 19 | Effect present, but **not selective** — reduces genuine ball evidence and bat recognition by an equal or larger amount |
| 7 | Feasibility of a selective, fully decoupled edit (P1-P4) | **Not established** — gated on Exp6's non-selectivity; P4 (association-knowledge preservation) not even measured |

**Summary:** the single-direction, single-layer edit that most cleanly reduces the spurious effect also destroys genuine object recognition at the same layer. Localization (Phase B, Exp5 here) is confirmed; a selective causal fix at this resolution is not.

### `src/cooccurrence_causal_coupling/` — generalizing the causal test to population scale

Own full write-up: [`src/cooccurrence_causal_coupling/README.md`](src/cooccurrence_causal_coupling/README.md). Re-runs the case study's causal test (Exp6 above) across the full 50-image / 3,039-pair population from Stage 10-11, using a fixed-effects-controlled direction estimate (not a naive diff-in-means) and a pre-registered λ-grid over 4 layers (3, 13, 16, 24), with train/val/test image-disjoint splits.

- The FE-controlled direction has a real, dose-dependent causal effect on the downstream $\beta$: at layer 24, λ=1.0, $\beta$ drops 99% (0.306 → 0.004), monotonic in λ, and clearly beats random/shuffled-direction controls.
- It is **still not selective**: the same intervention destroys genuine target-recognition evidence by an even larger margin at the same layer (−3.06 logits), and low-co-occurrence absent targets *rise* rather than staying flat.
- **Conclusion:** causal load-bearing is confirmed at population scale, generalizing the single-pair case study's negative selectivity finding rather than overturning it.

---

## Phase D — Debiasing Feasibility

Given that representation-editing is not selective (Phase C), Phase D asks whether **training** (rather than a runtime edit) can reduce the spurious pathway on the same fixed `baseball bat → sports ball` pair, while preserving genuine recognition.

### `src/adversarial_functional_debiasing_pilot/`

Own full write-up: [`src/adversarial_functional_debiasing_pilot/README.md`](src/adversarial_functional_debiasing_pilot/README.md). LoRA fine-tuning (`q_proj`/`v_proj` only) on two variants — Clean Debias (train on clean negatives) vs. Adv Debias (train on PGD-attacked negatives at ε=16/255) — evaluated on held-out clean images, image-ID-disjoint train/test.

- **GO**: bat→ball coupling $B$ drops from 1.937 (Original) → 0.320 (Clean Debias) → −0.182 (Adv Debias), with genuine ball/bat recognition staying within a pre-registered 10-pp non-selectivity threshold.
- Caveat: the Adv-vs-Clean *ordering* (the pattern that would specifically motivate adversarial exposure) is not statistically significant at n=20 test images (paired 95% CI for Adv−Clean spans −0.631 to +0.439).

### `src/adversarial_signal_debiasing_pilot/`

Own full write-up: [`src/adversarial_signal_debiasing_pilot/README.md`](src/adversarial_signal_debiasing_pilot/README.md). Follow-up: decomposes the adversarially-induced layer-19 representation shift (PCA / PLS) to test whether a more *selective* spurious component exists, then trains a fourth variant, Adv+Decomp Debias, that suppresses only that component.

- **NO-GO**: the best decomposed candidate (PLS2) beats a plain mean-direction baseline but does not clearly beat the best of 20 random directions on an internal selectivity test.
- Corroborated by the downstream training result: Adv+Decomp Debias is not better than plain Adv Debias on coupling, and is worse than every other variant on genuine Ball+ retention.
- Reinforces Phase C's finding from a different angle: the non-selectivity of the layer-19 shift is not an artifact of using a single diff-in-means direction — it survives both PCA (unsupervised) and PLS (supervised) decomposition.

---

## Where Things Stand

| What was tested | Phase | Result |
|---|---|---|
| Whether higher co-occurrence makes hallucination easier to induce (forced yes/no) | A | Confirmed |
| Whether the same holds for open-ended captioning | A | Not yet established — pipeline exists (Stage 6-8), no controlled result written up |
| Whether the bias is linearly readable from the frozen visual encoder | B | Not supported |
| Whether the bias is linearly readable from the LLM decoder stream | B | Confirmed, from layer ≈13 onward |
| Whether that representation is causally used by the final decision | C | Confirmed (single pair and, generalized, full population) |
| Whether a selective (non-destructive) causal edit is currently achievable | C | Not achieved — best edits found are not selective |
| Whether training instead of editing can reduce the effect while preserving recognition | D | Promising (GO) for adversarial LoRA fine-tuning; decomposition-guided selectivity (NO-GO) adds nothing on top |

Everything here is single-model (LLaVA-1.5-7B), single-dataset (COCO 2017); see each stage's own README for its specific limitations.

---

## File Structure

```text
configs/                                   Stage 1-11 (general) YAML configs
src/cooc_diagnostic/                       shared library for Stages 1-11 (general)
  coco_index.py                            COCO image-level present-category indexing
  cooccurrence_stats.py                    PMI/lift/raw conditional computation
  covariates.py                            category-level average object area
  clip_similarity.py                       CLIP image-text similarity and image embeddings
  strata_sampling.py                       candidate generation + high/low strata construction
  matching.py                              Coarsened Exact Matching + balance table
  stratified_subsample.py                  stratified subsampling preserving CEM cell ratios
  llava_runtime.py                         LLaVA preprocessing/prompting/Yes-No decision
  pgd_attack.py / random_attack.py         L∞ PGD attack and random-noise control
  epsilon_star.py                          ε* exponential search + binary search
  sanity_checks.py                         aggregation of three required sanity checks
  survival_analysis.py                     KM, stratified Cox, Weibull AFT, McNemar, Holm
  linear_probe.py                          Stage 5 linear probe (excess AUC)
  mention_detection.py / caption_attack.py Stage 6-8 open-ended/caption-mention readout + attack
  masking.py                               counterfactual region removal (Stage 11 case study)
scripts/run_stage{1..11}_*.py              execution scripts, one (or more, for Stage 11) per stage
tests/                                      CPU-only unit tests (62 tests, all passing)
outputs/CooccurrenceHallucinationDiagnostic/stage{1..11}_*/   stage-wise outputs (repo-root outputs/, not under this directory)

src/cooccurrence_causal_coupling/          Phase C: population-scale causal intervention (own README)
src/adversarial_functional_debiasing_pilot/  Phase D: LoRA debiasing pilot, Clean vs. Adv (own README)
src/adversarial_signal_debiasing_pilot/      Phase D: signal decomposition + selective debiasing pilot (own README)
```

## Reproduction

```bash
cd src/experiments/CooccurrenceHallucinationDiagnostic
export PYTHONPATH=src
PY=/opt/anaconda3/envs/py3_11/bin/python

# Unit tests (GPU not required; all 62 tests should pass)
$PY -m unittest discover -s tests -v

# Phase A / B, Stage 1-11 (general population) -- run in order, each depends on prior stage's output
$PY scripts/run_stage1_cooccurrence.py            --config configs/stage1_cooccurrence.yaml
$PY scripts/run_stage2_sampling_matching.py       --config configs/stage2_sampling_matching.yaml        # GPU: CLIP
$PY scripts/run_stage3_attack.py                  --config configs/stage3_attack.yaml --pilot           # GPU: LLaVA; run pilot first
$PY scripts/run_stage3_attack.py                  --config configs/stage3_attack.yaml
$PY scripts/run_stage4_survival_analysis.py       --config configs/stage4_survival_analysis.yaml
$PY scripts/run_stage5_linear_probe.py            --config configs/stage5_linear_probe.yaml             # GPU: CLIP features
$PY scripts/run_stage6_open_ended_transfer.py     --config configs/stage6_open_ended_transfer.yaml       # GPU: LLaVA
$PY scripts/run_stage7_caption_attack.py          --config configs/stage7_caption_attack.yaml            # GPU: LLaVA
$PY scripts/run_stage8_survival_analysis_caption.py --config configs/stage8_survival_analysis_caption.yaml
$PY scripts/run_stage9_clean_target_evidence.py   --config configs/stage9_clean_target_evidence.yaml     # GPU: LLaVA
$PY scripts/run_stage9_analysis.py                --config configs/stage9_analysis.yaml
$PY scripts/run_stage10_within_image_evidence.py  --config configs/stage10_within_image_evidence.yaml    # GPU: LLaVA
$PY scripts/run_stage10_analysis.py               --config configs/stage10_analysis.yaml
$PY scripts/run_stage11_layer_localization.py     --config configs/stage11_layer_localization.yaml       # GPU: LLaVA
$PY scripts/run_stage11_analysis.py               --config configs/stage11_analysis.yaml

# Phase C, Stage 11 case study (baseball bat -> sports ball), Exp0-Exp7
$PY scripts/run_stage11_exp0_pair_statistics.py      --config configs/stage11_case_bat_ball.yaml
$PY scripts/run_stage11_exp1_build_sample.py         --config configs/stage11_case_bat_ball.yaml
$PY scripts/run_stage11_exp1_attack.py               --config configs/stage11_case_bat_ball.yaml         # GPU
$PY scripts/run_stage11_exp2_clean_evidence.py       --config configs/stage11_case_bat_ball.yaml         # GPU
$PY scripts/run_stage11_exp3_boundary_consistency.py --config configs/stage11_case_bat_ball.yaml
$PY scripts/run_stage11_exp4_counterfactual.py       --config configs/stage11_case_bat_ball.yaml         # GPU
$PY scripts/run_stage11_exp4b_visual_audit.py        --config configs/stage11_case_bat_ball.yaml
$PY scripts/run_stage11_exp5_localization.py         --config configs/stage11_case_bat_ball.yaml         # GPU
$PY scripts/run_stage11_exp6_causal_intervention.py  --config configs/stage11_case_bat_ball.yaml --layer 19 --device cuda:0  # GPU
$PY scripts/run_stage11_exp7_decoupling.py           --config configs/stage11_case_bat_ball.yaml
# full narrative: outputs/CooccurrenceHallucinationDiagnostic/stage11_case_bat_ball/README.md

# Phase C, population-scale causal coupling -- see src/cooccurrence_causal_coupling/README.md for details
cd src/cooccurrence_causal_coupling/scripts
$PY 01_collect_hidden_states.py   --config ../configs/01_collect_hidden_states.yaml   # GPU
$PY 02_estimate_directions.py     --config ../configs/02_estimate_directions.yaml
$PY 03_screen_layers.py           --config ../configs/03_screen_layers.yaml           # GPU
$PY 04_full_intervention_scan.py  --config ../configs/04_full_intervention_scan.yaml   # GPU
$PY 05_analyze_intervention.py    --config ../configs/05_analyze_intervention.yaml

# Phase D, debiasing pilots -- see each pilot's own README.md for full commands
cd ../../adversarial_functional_debiasing_pilot
# build_split.py -> generate_adversarial_forget_set.py -> train_lora_debias.py (x2 variants) -> evaluate_model.py -> analysis/
cd ../adversarial_signal_debiasing_pilot
# prepare_data.py -> extract_layer19_shifts.py -> decompose_signal.py -> evaluate_components.py -> train_adv_decomp_debias.py -> evaluate_all_models.py
```
