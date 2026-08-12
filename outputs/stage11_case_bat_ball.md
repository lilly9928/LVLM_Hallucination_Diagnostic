# Stage 11 — Case Study: "baseball bat" → "sports ball"

## Core Question

Stages 1–5 established a **population-level** behavioral effect across many object pairs: objects
that co-occur more with an image's present objects are easier to hallucinate. Stage 11 asks the same
question end-to-end for **one concrete, visually inspectable pair**, and pushes further — from
behavior, to a counterfactual input-level test, to internal localization, to a causal intervention:

> Does the `baseball bat` ↔ `sports ball` co-occurrence create a spurious functional pathway — bat
> context → unsupported ball evidence → hallucination vulnerability — and can that specific pathway
> be selectively edited away inside the model without breaking genuine ball evidence, bat recognition,
> or general output quality?

Model: `llava-hf/llava-1.5-7b-hf` (same runtime as Stages 3/9/10). Data: COCO 2017. Seed: 42
throughout. All code reuses Stage 1–5's modules wherever possible (PMI, matching, LLaVA runtime, PGD/
epsilon\*, probe fitting) — see `outputs/CooccurrenceHallucinationDiagnostic/stage11_case_bat_ball/repository_audit.md`
for the full reuse map.

This is a diagnostic case study, not a method paper: **every stage was allowed to fail**, and one of
them (Exp6) did — reported below exactly as it came out.

---

## Exp0 — Is this actually a strong pair?

Train2017 (118,287 images): N(bat) = 2506, N(ball) = 4262, N(bat & ball) = 939.

$$\text{PMI(bat, ball)} = 2.342 \qquad \text{lift} = 10.40 \qquad P(\text{ball}\mid\text{bat}) = 0.375 \text{ vs. } P(\text{ball}) = 0.036$$

Ranked **59th of 5,338** ordered COCO category pairs by PMI — a top-~1% association. Val2017 group
sizes: G00 (bat−,ball−) = 4766, **G10 (bat+,ball−) = 65**, G01 (bat−,ball+) = 137, G11 (bat+,ball+) = 32.

**Decision: GO.** (All 65 G10 images are used in every later experiment — no further subsampling of
the treatment arm anywhere in Stage 11.)

---

## Exp1 — Phenomenon: does bat context increase attack vulnerability?

65 G10 images vs. 65 G00 controls, matched (coarsened exact matching, reusing Stage 2's exact
mechanism) on scene complexity and CLIP similarity to "sports ball" / "baseball bat". Same PGD/
$\epsilon^*$ attack as Stage 3, unchanged (20 steps × 2 restarts, $\epsilon_{max}=32/255$).

| Analysis | Result |
|---|---|
| Stratified Cox (primary) | **HR = 1.760**, 95% CI [1.077, 2.875], p = 0.024 |
| Weibull AFT time ratio | **0.416**, 95% CI [0.254, 0.683], p = 0.0005 |
| Median $\epsilon^*$ | G10 = 0.000115 vs. G00 = 0.000520 |
| McNemar (raw clean Yes-rate) | p = 0.69 (**not significant**) |

Bat context needs roughly **42% of the perturbation budget** to hallucinate a ball, compared to
matched controls — but this shows up in continuous evidence/vulnerability, not in the raw binary
"already hallucinating at $\epsilon=0$" rate at this sample size. **Decision: GO.**

---

## Exp2 — Clean functional evidence, before any attack

$s_{ball} = \text{logit(Yes)} - \text{logit(No)}$ for *"Is there a sports ball in the image?"*, no attack:

| Group | n | mean $s_{ball}$ | Yes rate |
|---|---|---|---|
| G00 (bat−,ball−) | 65 | −1.260 | 0.215 |
| **G10 (bat+,ball−)** | 65 | **−0.320** | 0.262 |
| G01 (bat−,ball+) | 65 | 2.580 | 0.785 |
| G11 (bat+,ball+) | 32 | 1.665 | 0.813 |

G10 vs. G00: mean diff = **+0.940**, Cohen's d = 0.745, bootstrap 95% CI [0.50, 1.36], Mann-Whitney
p = 1.75e-6. **Decision: GO.**

---

## Exp3 — Boundary consistency (a check, not new evidence)

Joining Exp1 and Exp2 on image_id: Spearman ρ = **−0.895** between $s_{ball}$ and $\epsilon^*$. This is
expected and partly mechanical ($\epsilon^*$ is derived from the same Yes/No margin as $s_{ball}$), so
it is reported as a consistency check, not independent causal evidence.

---

## Exp4 — Is it the bat itself, or just image style? (counterfactual)

For all 65 G10 images: gray-fill the bat region (Stage-independent `mask_dog_regions` technique,
generalized to any category) vs. a **mirrored + translated sham** mask of equal area.

$$\Delta_{bat \to ball} = s_{ball}(\text{original}) - s_{ball}(\text{bat removed})$$

| Quantity | Value |
|---|---|
| $\Delta_{bat \to ball}$ mean | **0.238**, 95% CI [0.114, 0.380], Wilcoxon p = 0.00046 |
| $\Delta_{sham}$ mean | −0.015 (≈ 0, as expected for a null edit) |
| $\Delta_{bat} - \Delta_{sham}$ | **0.254**, 95% CI [0.108, 0.407], p = 0.00085 |

Both hypotheses hold: removing the bat reduces unsupported ball evidence, and it does so **more** than
an equal-area sham edit. Visual audit (mandatory before interpreting this) found 4/65 masks (6%) with
no visible effect — small/edge bat regions lost to LLaVA's own resize+crop preprocessing — and 7/65
sham placements (11%) with unresolved overlap, both disclosed, neither excluded from the statistics.

### Checkpoint 1: **GO**

All four criteria met (elevated evidence, bat-specific reduction, exceeds sham, artifacts disclosed
and non-disqualifying) → proceed to internal localization.

---

## Exp5 — Where inside the model does this appear?

Readout: **logit lens** (final RMSNorm + `lm_head`, restricted to yes/no token ids) at all 33 LLM
readout points; a **logistic probe** (Stage 5's exact `fit_probe`, trained on an independent 500-image
train2017 sample) for the vision-tower and projector stages.

The signal is small in the vision tower / projector (0.32 / 0.25, vs. sham 0.11 / 0.02), stays small
and noisy through LLM layers 1–13, then rises sharply and **peaks at layer 19 of 32**
($\Delta_{bat}=1.493$ vs. $\Delta_{sham}=-0.030$ — the cleanest separation in the whole profile), then
declines toward the output. A **candidate localization**, not a causal claim.

---

## Exp6 — Does editing that layer selectively fix it?

Motivated directly by Exp5's peak: subtract the mean G10 (original − bat-removed) hidden-state shift
at layer 19's output, at every token position, during scoring — the least destructive single-direction
edit available.

| Test | mean reduction in $s$ | Yes-rate before → after |
|---|---|---|
| **Main: unsupported ball (G10)** | **0.257** (p=1.7e-12) | 0.262 → 0.154 |
| Control 1: genuine ball (G01) | 0.275 (p=2.6e-8) | 0.750 → 0.750 |
| Control 1: genuine ball (G11) | 0.256 (p=6.8e-7) | 0.813 → 0.688 |
| Control 2: bat recognition (G10) | 0.320 (p=1.9e-12) | 0.969 → 0.969 |
| Control 3: general captions | 8/10 byte-identical, rest minor rewording | no collapse |

**All four reductions are essentially the same size.** This is the textbook signature of a
**non-selective** edit — it suppresses generic yes-leaning evidence at that layer, not a bat-ball-
specific pathway. General output quality is unaffected (Control 3), but selectivity fails.

### Checkpoint 2: **STOP (non-selective)**

---

## Exp7 — Is selective decoupling feasible?

| Requirement | Result | Met? |
|---|---|---|
| P1: spurious effect decreases | reduction = 0.257, p=1.7e-12 | **YES** |
| P2: genuine ball evidence remains | reduced by ~same amount as main effect | **NO** |
| P3: bat/context info remains | reduced by *more* than the main effect | **NO** |
| P4: association knowledge remains | not measured — no adequate metric found in scope | n/a |

**1 of 3 measurable requirements met. Functional decoupling is not established** with this
intervention, at this layer, for this pair.

---

## Overall Result

| Question | Answer |
|---|---|
| Strong train-set association? | **Yes** (PMI=2.34, rank 59/5338) |
| Higher clean unsupported-ball evidence? | **Yes** (d=0.75, p=1.75e-6) |
| Higher attack vulnerability? | **Yes** for continuous evidence (Cox p=0.024); **not** for raw clean-Yes rate (p=0.69) |
| Is it the bat specifically (vs. sham)? | **Yes** ($\Delta_{bat}>\Delta_{sham}$, p=0.00085) |
| Localizable inside the model? | **Candidate localization** at LLM layer 19 |
| Is that layer causally, selectively responsible? | **No** — non-selective (Exp6) |
| Selective decoupling feasible? | **No**, 1/3 measurable requirements met |

$$
\text{Bat}
\;\to\;
\underbrace{\text{unsupported Ball evidence}}_{\text{Exp2, Exp4 — SUPPORTED}}
\;\to\;
\underbrace{\text{candidate readout at LLM layer 19}}_{\text{Exp5 — localization only}}
\;\to\;
\underbrace{\text{hallucination vulnerability}}_{\text{Exp1 — SUPPORTED}}
$$

The arrow "layer 19 causally, selectively mediates this" is **not** drawn as fact — Exp6 tested it and
it failed the selectivity controls. That is a real, useful negative result, not an unfinished step.

---

## What We Cannot Claim

- Single pair, single model — no cross-pair or cross-model generalization implied.
- Exp6's null result is about *this* difference-of-means intervention at *this* layer, not proof that
  no selective intervention could ever work for this pathway.
- ~6% of Exp4's bat-removal masks had no visible effect (preprocessing-scale artifact); ~11% of sham
  placements had unresolved overlap with other objects.
- The Exp1 matched sample is well-balanced on scene complexity but *not* on CLIP similarity to
  "baseball bat" (expected, since G10 images genuinely contain a bat) or fully on "sports ball"
  similarity.
- Association-knowledge preservation (P4) was **not measured** — see
  `outputs/.../stage11_case_bat_ball/association_measure_definition.md`.
- Exp3's correlation is a consistency check, not independent evidence.

---

## File Structure

```text
configs/stage11_case_bat_ball.yaml                   config for all Exp0-7 scripts
scripts/run_stage11_exp0_pair_statistics.py          Exp0
scripts/run_stage11_exp1_build_sample.py             Exp1a: CEM-matched G10/G00 sample
scripts/run_stage11_exp1_attack.py                   Exp1b: epsilon* attack + survival stats
scripts/run_stage11_exp2_clean_evidence.py           Exp2 + Exp2B ranked gallery
scripts/run_stage11_exp3_boundary_consistency.py     Exp3
scripts/run_stage11_exp4_counterfactual.py           Exp4: bat removal vs. sham
scripts/run_stage11_exp4b_visual_audit.py            Exp4B: mandatory visual audit figure
scripts/run_stage11_exp5_localization.py             Exp5: logit-lens + probe layerwise readout
scripts/make_exp5_figure.py                          Exp5 figure
scripts/run_stage11_exp6_causal_intervention.py      Exp6: layer-19 direction ablation + controls
scripts/run_stage11_exp7_decoupling.py               Exp7: decoupling feasibility summary
src/cooc_diagnostic/masking.py                        new: gray-fill removal + mirrored/translated sham

outputs/CooccurrenceHallucinationDiagnostic/stage11_case_bat_ball/
  repository_audit.md                                 Step 0: what was reused vs. built new
  exp0_pair_statistics.json, exp0_group_membership.csv
  exp1_sample_selection.csv, exp1_sample_balance.json, exp1_epsilon_star.csv, exp1_statistics.json
  exp2_clean_evidence.csv, exp2_statistics.json, exp2_g10_ranked.csv, exp2_g10_selections.json
  exp3_boundary_consistency.csv, exp3_statistics.json
  exp4_counterfactual.csv, exp4_statistics.json, exp4b_selected_examples.json, counterfactual_images/
  exp5_layerwise_evidence.csv, exp5_statistics.json
  exp6_internal_intervention.csv, exp6_statistics.json
  exp7_decoupling_summary.csv, exp7_statistics.json
  checkpoint1.md, checkpoint2.md, association_measure_definition.md, figures/
  README.md                                            full narrative writeup (this file's source)

notebook/baseball_bat_sports_ball_case_analysis.ipynb   executed, 0 errors -- all inline plots + galleries
```

## Reproduction

```bash
cd src/experiments/CooccurrenceHallucinationDiagnostic
PY=/opt/anaconda3/envs/py3_11/bin/python

$PY scripts/run_stage11_exp0_pair_statistics.py --config configs/stage11_case_bat_ball.yaml
$PY scripts/run_stage11_exp1_build_sample.py    --config configs/stage11_case_bat_ball.yaml
$PY scripts/run_stage11_exp1_attack.py          --config configs/stage11_case_bat_ball.yaml --device cuda:2
$PY scripts/run_stage11_exp2_clean_evidence.py  --config configs/stage11_case_bat_ball.yaml --device cuda:0
$PY scripts/run_stage11_exp3_boundary_consistency.py --config configs/stage11_case_bat_ball.yaml
$PY scripts/run_stage11_exp4_counterfactual.py  --config configs/stage11_case_bat_ball.yaml --device cuda:3
$PY scripts/run_stage11_exp4b_visual_audit.py   --config configs/stage11_case_bat_ball.yaml
$PY scripts/run_stage11_exp5_localization.py    --config configs/stage11_case_bat_ball.yaml --device cuda:2
$PY scripts/make_exp5_figure.py                 --config configs/stage11_case_bat_ball.yaml
$PY scripts/run_stage11_exp6_causal_intervention.py --config configs/stage11_case_bat_ball.yaml --layer 19 --device cuda:2
$PY scripts/run_stage11_exp7_decoupling.py      --config configs/stage11_case_bat_ball.yaml

# Notebook (loads saved outputs only, no inference re-run):
cd ../../../notebook
jupyter nbconvert --to notebook --execute --inplace baseball_bat_sports_ball_case_analysis.ipynb \
    --ExecutePreprocessor.kernel_name=py3_11
```

Full narrative, all numbers, and every limitation are in
`outputs/CooccurrenceHallucinationDiagnostic/stage11_case_bat_ball/README.md`.
