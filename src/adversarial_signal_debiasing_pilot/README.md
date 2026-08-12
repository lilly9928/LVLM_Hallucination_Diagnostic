# Adversarial Signal Decomposition and Functional Debiasing Pilot

## Research Question

Fixed case: context C = "baseball bat", target T = "sports ball", model =
`llava-hf/llava-1.5-7b-hf`.

- **Question A (signal decomposition feasibility):** does the representation
  shift induced by a targeted adversarial attack contain a separable
  component that selectively supports the spurious baseball-bat -> sports-ball
  hallucination effect?
- **Question B (method feasibility):** if such a component exists, can we
  train the model to suppress it and obtain better *selective* functional
  debiasing than simple clean/adversarial fine-tuning?

## Motivation

A prior single-direction Layer-19 intervention (diff-in-means, "original
minus bat-removed") reduced unsupported Ball evidence, but reduced genuine
Ball evidence and Bat recognition by a similar amount -- i.e. it behaved like
a generic "reduce whatever Yes-leaning evidence is present at this layer"
edit, not a co-occurrence-specific one (see "Prior Non-selective Intervention
Result" below). This motivates asking whether the adversarially induced shift
is a *mixture* of components, only one (or none) of which is actually
spurious-selective.

## Prior Evidence

1. Higher bat-ball co-occurrence -> smaller epsilon* (easier to flip to "Yes, there's a ball").
2. Higher co-occurrence -> higher clean target-positive evidence s_T.
3. Within-image fixed-effects show a positive co-occurrence-specific effect
   even controlling for image and target effects.
4. Baseball-bat -> sports-ball case study: strong pair-level co-occurrence,
   higher Ball evidence under Bat+/Ball-, and a bat-removal counterfactual
   reduces unsupported Ball evidence more than a sham edit.

## Prior Non-selective Intervention Result

`CooccurrenceHallucinationDiagnostic/scripts/run_stage11_exp6_causal_intervention.py`:
a forward hook on `language_model.layers[18]` (= hidden_states[19])
unconditionally subtracted a fixed mean direction v_19 (diff-in-means over 65
G10 images, original vs. bat-removed) from every position's hidden state.

| Test | Mean reduction in s |
|---|---:|
| Main (G10, unsupported ball) | 0.257 |
| Control 1a (genuine ball, G01) | 0.275 |
| Control 1b (genuine ball, G11) | 0.256 |
| Control 2 (bat recognition, G10) | 0.320 (larger) |

**Classified non-selective** -- the reduction on genuine ball evidence and
bat recognition was as large as (or larger than) the reduction on the
spurious target, so the edit does not distinguish spurious from genuine
evidence.

## Hypothesis

> The adversarially induced representation shift is a mixture of multiple
> latent components, and a more selective spurious component may be
> recoverable by decomposition (PCA/SVD, optionally PLS), rather than using
> the single mean-direction shift the prior intervention used.

## What We Reused vs. What Is New

See `audit/repository_audit.md` for the full Step-0 audit. In short: the
dataset split (seed=42, image-ID-disjoint), the 45-image adversarial forget
set (PGD, epsilon=16/255, from the prior `adversarial_functional_debiasing_pilot`),
and two of the four model variants (**Original**, **Clean Debias**, **Adv
Debias**) are reused verbatim -- they were already produced for this exact
fixed case and re-running them would not change any number. This pilot's new
work is the Layer-19 signal decomposition, the component functional
selectivity test, and training/evaluating the fourth variant, **Adv+Decomp
Debias**.

## Dataset

Reused verbatim (seed=42, split independently within each of the 4 disjoint
COCO groups G00/G10/G01/G11, then combined -- no image ever appears in both
train and test, leakage explicitly checked at build time):

| Role | TRAIN | TEST |
|---|---:|---:|
| G10 (Bat+/Ball-, forget/spurious) | 45 | 20 |
| GC (Bat+, context retention) | 45 (same images as G10) | 20 (same images as G10 test) |
| GT (Ball+, target retention) | 45 (22 G11 + 23 G01) | 25 (10 G11 + 15 G01) |
| G00 (Bat-/Ball-, coupling denominator) | -- | 25 |

Additionally, for Part VII's component-selection discipline, TRAIN images are
further split 70/30 into an internal DEV/VAL split (seed=42, independent per
role, GC mirrors G10's split since they are the same images):
G10_forget dev=31/val=14, GC_context_retain dev=31/val=14,
GT_target_retain dev=31/val=14.

## Adversarial Exposure

Reused verbatim: existing targeted PGD attack (`pgd_attack_with_restarts`),
epsilon=16/255 (fixed before inspecting any debiasing result), 20 steps, 2
restarts, on TRAIN G10 only, question "Is there a sports ball in the image?".

| N | Attack success rate | Mean clean s_ball | Mean adv s_ball | Mean delta_s_ball |
|---:|---:|---:|---:|---:|
| 45 | 1.000 | -0.4240 | 1.0531 | 1.4771 |

![Fig 1](../../../../../outputs/adversarial_signal_debiasing_pilot/figures/fig1_adversarial_exposure.png)

## Representation Shift

`delta_h = h_adv - h_clean` at Layer 19 (`hidden_states[19]`, last
teacher-forced decision-token position), for all 45 TRAIN G10 images.

- delta_h shape: (45, 4096)
- mean ||delta_h||: 11.3276

## Signal Decomposition

PCA fit on the DEV portion only (31 images), K=10 components:

| Component | Explained variance | Cumulative |
|---|---:|---:|
| PC1 | 56.56% | 56.56% |
| PC2 | 10.16% | 66.73% |
| PC3 | 7.58% | 74.31% |
| PC4 | 6.13% | 80.44% |
| PC5 | 3.75% | 84.19% |
| PC6 | 2.68% | 86.86% |
| PC7 | 2.26% | 89.13% |
| PC8 | 1.79% | 90.92% |
| PC9 | 1.47% | 92.39% |
| PC10 | 1.42% | 93.81% |

PC1 alone captures the large majority (56.6%) of variance in the adversarial
shift. As required, PC1 is **not** assumed spurious on that basis alone --
see Component Selectivity below, where PC1 turns out to behave almost
identically to the naive mean direction (the same non-selective pattern as
the prior intervention).

PLS was additionally fit (Part VIII), triggered because the PCA-only pass
showed poor selectivity relative to the random-direction baseline (see
below) -- 3 components, X=delta_h, y=delta_s_ball, DEV images only.

## Component Selectivity

Projection intervention `h' = h - lambda*proj_u(h)` at Layer 19, lambda in
{0.5, 1.0}, evaluated on the internal VAL split (14 images per group, carved
from TRAIN, never touched by final test). Ranking metric:
`Selectivity_k = |Delta_spurious| - |Delta_target| - |Delta_context|`, minimum
over the two lambda values (a candidate must be selective at *both*
magnitudes, not just one).

| Candidate | Selectivity (min over lambda) | Mean Δ_spurious | Mean Δ_target | Mean Δ_context |
|---|---:|---:|---:|---:|
| Mean direction | -3.3114 | +0.969 | -1.101 | -2.347 |
| PC1 (56.6% var.) | -3.3817 | +1.046 | -1.109 | -2.499 |
| PLS1 | -3.5848 | +0.836 | -1.143 | -2.424 |
| PC2 | -0.5257 | +0.050 | +0.147 | +0.291 |
| PC5 | -0.6250 | +0.001 | +0.211 | +0.247 |
| PC4 | -0.3806 | -0.006 | +0.160 | +0.129 |
| PC9 | -0.3705 | +0.089 | +0.139 | +0.225 |
| PLS3 | -0.3661 | +0.018 | +0.068 | +0.218 |
| PC7 | -0.2723 | -0.008 | -0.103 | -0.110 |
| PC10 | -0.1886 | +0.064 | +0.066 | +0.137 |
| PC3 | -0.1016 | -0.039 | -0.054 | -0.063 |
| PC8 | -0.0647 | -0.008 | +0.026 | +0.031 |
| **PC6** | **-0.0480** | +0.004 | +0.008 | -0.035 |
| **PLS2 (best decomposed)** | **-0.0156** | +0.035 | +0.003 | -0.038 |
| Best random direction (of 20) | **+0.0033** | +0.007 | +0.002 | +0.002 |

Full table (all 20 random directions, all 2 lambdas separately):
`decomposition/component_selectivity.csv`, `component_intervention/*.csv`.

**PC1 replicates the prior non-selective intervention almost exactly**
(selectivity -3.38 vs. the mean direction's -3.31; both show a large,
indiscriminate shift across spurious/target/context) -- despite explaining
56.6% of the variance, PC1 is not a spurious-selective direction, confirming
the brief's warning not to assume this from variance alone. PLS1 (the
target-supervised component regressed directly on delta_s_ball) shows the
*same* failure mode, for the same reason: both are dominated by the single
large shared shift in the adversarial perturbation.

PC2-PC10 and PLS2-PLS3 have much smaller-magnitude effects, but **none beats
the best of 20 random directions** (+0.0033). The single best decomposed
candidate, **PLS2**, clearly beats the mean direction but does not beat the
random-direction baseline -- it fails Part VII's selection criterion #5
("better selectivity than random directions"). PLS was run (Part VIII)
because the PCA-only pass already showed this same weak-selectivity pattern.

![Fig 3](../../../../../outputs/adversarial_signal_debiasing_pilot/figures/fig3_component_selectivity.png)
![Fig 4](../../../../../outputs/adversarial_signal_debiasing_pilot/figures/fig4_mean_vs_best_component.png)

**Conclusion for Question A: not supported.** Simple linear decomposition
(PCA or PLS) of the Layer-19 adversarial shift does not isolate a
functionally selective spurious component beyond the noise floor set by
random directions, in this pilot's data (n=31 dev / n=14 val TRAIN G10
images).

## Debiasing Prototype

Despite the negative selectivity result, Part IX/X's method prototype was
still trained on the best available component (PLS2), per the pilot's
protocol -- Question B is "if such a component exists, can we train..."; a
component that fails the selectivity bar is still worth testing to see
whether training transfers differently than the raw intervention did.

**First attempt (`loss_mode: spur_only`, the brief's preferred first
prototype) FAILED**: `L_total = L_spur + lambda_T*L_target_retain +
lambda_C*L_context_retain`, with no direct "answer No" supervision on the
forget examples. The model collapsed to **always answering "Yes"** on every
question (100% Yes-rate on G10, G00, GT, and GC test images -- see
`evaluation/adv_decomp_results_spur_only_FAILED.csv`, checkpoint preserved at
`checkpoints/adv_decomp_debias_spur_only_FAILED/`). This is exactly the
failure mode the brief's fallback rule anticipates: the retain-target loss's
repeated "Yes" supervision generalized to *every* sports-ball question,
unconstrained by any competing "No" signal on the forget examples.

**Second attempt (`loss_mode: spur_plus_adv_no`, the documented fallback)**:
`L_total = L_adv_no + lambda_spur*L_spur + lambda_T*L_target_retain +
lambda_C*L_context_retain`, adding plain "No"-supervision CE on the
adversarial forget images. This trained without collapse (final losses:
loss_spur=0.594, loss_adv_no=0.677, loss_target_retain=0.710,
loss_context_retain=0.048) and is the version reported below as
**Adv+Decomp Debias**.

## Training Variants

| | Model A: Clean Debias | Model B: Adv Debias | Model C: Adv+Decomp Debias |
|---|---|---|---|
| Status | Reused verbatim | Reused verbatim | **Newly trained** |
| Forget data | Clean G10 (45) | Adversarial G10 (45) | Adversarial G10 pairs (45) |
| Loss | CE("No") + retain | CE("No") + retain | L_adv_no + L_spur(PLS2) + retain |
| LoRA | r=16, alpha=32, dropout=0.05, q_proj/v_proj | (identical) | (identical) |
| Trainable params | 8,388,608 / 7,071,815,680 (0.1186%) | (identical) | (identical) |
| Epochs / optimizer steps | 3 / 33 | 3 / 33 | 3 / 33 |

## Clean Test Results

All four models evaluated on the same 90 unseen (image_id, role) CLEAN TEST
rows (G10=20, G00=25, GT=25, GC=20; Original/Clean/Adv reused verbatim, only
Adv+Decomp newly run).

| Method | Coupling B ↓ | G10 s_ball ↓ | Ball+ Acc ↑ | Bat+ Acc ↑ |
|---|---:|---:|---:|---:|
| Original | 1.9369 [1.363, 2.439] | -0.0875 | 0.880 | 0.950 |
| Clean Debias | 0.3205 [-0.692, 1.315] | -0.4258 | 0.920 | 1.000 |
| Adv Debias | -0.1823 [-0.649, 0.240] | -0.4867 | 0.840 | 1.000 |
| Adv + Decomp Debias | 0.2834 [0.040, 0.506] | -0.4484 | **0.720** | 1.000 |

(Bootstrap 95% CIs in brackets for Coupling B.)

## Functional Coupling

`B = E[s_ball | Bat+, Ball-] - E[s_ball | Bat-, Ball-]`.

Observed order: **B_Adv (-0.182) < B_AdvDecomp (0.283) < B_Clean (0.320) <
B_Original (1.937)**. Adv+Decomp Debias *does* reduce coupling relative to
Original (paired bootstrap Adv+Decomp minus Original on G10 s_ball:
-0.361, 95% CI [-0.656, -0.073], excludes 0), but it does **not** beat plain
Adv Debias (paired diff Adv+Decomp minus Adv: +0.038, 95% CI [-0.170,
0.195], includes 0 -- no significant difference), and sits essentially
where Clean Debias already was.

![Fig 5](../../../../../outputs/adversarial_signal_debiasing_pilot/figures/fig5_coupling_by_method.png)

## Preservation Results

| Method | Ball+ Acc (GT) | Bat+ Acc (GC) | Ball+ drop vs. Original | Bat+ drop vs. Original |
|---|---:|---:|---:|---:|
| Original | 0.880 | 0.950 | -- | -- |
| Clean Debias | 0.920 | 1.000 | -0.040 (improved) | -0.050 (improved) |
| Adv Debias | 0.840 | 1.000 | +0.040 | -0.050 (improved) |
| Adv + Decomp Debias | **0.720** | 1.000 | **+0.160** | -0.050 (improved) |

Adv+Decomp Debias's genuine-Ball retention drop (16 percentage points)
**exceeds the pre-specified 10pp retention threshold** -- worse retention
than either reused variant, despite the extra L_spur/retain machinery. Bat
retention is preserved across all variants (context retain loss appears
robust regardless of forget-loss formulation).

![Fig 6](../../../../../outputs/adversarial_signal_debiasing_pilot/figures/fig6_method_selectivity.png)

## Go / No-Go Decision

**NO-GO A: no separable component.** The best decomposed candidate (PLS2)
does not clearly beat both the mean-direction and random-direction baselines
on the internal VAL selectivity test (it beats the mean direction by a wide
margin, but does not beat the best of 20 random directions). This
classification is made from the component-selection stage alone, before
looking at Model C's clean-test numbers -- which turn out to corroborate it:
Adv+Decomp Debias is not better than plain Adv Debias on coupling B, and is
worse than every other variant on genuine Ball+ retention.

## What We Can Claim

- The prior single-direction (diff-in-means) Layer-19 intervention's
  non-selectivity replicates under PCA's PC1 and PLS's PLS1: both recover
  essentially the same large, non-selective shift, from two different
  decomposition objectives (unsupervised variance-maximizing and
  supervised target-regressing). This is a real, converging negative result,
  not an artifact of one particular decomposition choice.
- Simple linear decomposition (K=10 PCA, 3 PLS) of a 45-image (31 dev / 14
  val) adversarial shift at one layer does not surface a component that is
  both non-trivial in effect size and more selective than random noise, in
  this pilot's data.
- A minimal L_spur-based training objective is trainable (with adequate
  direct task supervision) but did not produce a clean-test debiasing result
  better than the much simpler Adv Debias baseline, and produced worse
  genuine-target retention than any other variant.
- The `spur_only` loss formulation, absent direct "No" supervision, reliably
  collapses this model to a degenerate always-"Yes" policy under this
  forget/retain mixture -- worth knowing before anyone else tries the
  "preferred first prototype" as specified.

## What We Cannot Claim

- That no separable spurious component exists at Layer 19 in general --
  only that PCA/PLS with these sample sizes (31 dev images) and this
  candidate set (K=10 PCA, 3 PLS) did not find one clearly better than
  chance.
- Any claim about other layers, other object pairs, or other models -- this
  is a single-pair, single-layer, single-model pilot by design.
- That the L_spur training objective is a dead end -- only that this
  particular formulation, with this particular (weakly-selective) component,
  did not help. A genuinely selective component might transfer differently.
- Any causal interpretation of PLS2 beyond "best of a weak field" -- it was
  selected because it ranked highest among decomposed candidates on VAL, not
  because it passed the pilot's own selectivity bar.

## Limitations

- Single pair (baseball bat -> sports ball), single model (LLaVA-1.5-7B),
  single layer (19) -- a feasibility pilot, not a general result.
- Component selection used a small internal VAL split (n=14 per group);
  selectivity estimates are noisy (see the notebook's Limitations section
  for a specific measurement-artifact caveat in the `Selectivity_k` metric:
  it can reward directions that change nothing over directions with partial,
  real but imperfect selectivity).
- Only one adversarial epsilon, one PGD configuration (reused, not
  re-tuned) -- the decomposed shift is specific to this attack's geometry.
- Two training-loss variants were tried for Model C (documented above); this
  is the anticipated fallback in the brief, not post-hoc hyperparameter
  search on a metric.
- No causal claim beyond correlation-plus-intervention at one layer, one
  token position, one direction-length (unit-norm projection) parameterization.

## Next Step

Given the **NO-GO A** result, the most defensible next step is *not* to
scale up this same recipe (more PCA components, more layers) without first
addressing why PC1/PLS1 both collapse onto the same non-selective direction:
a natural hypothesis is that the *adversarial* perturbation's dominant shift
is largely attack-geometry-driven (a generic "increase target logit" push)
rather than concept-specific, so decomposing delta_h from *this* attack may
never surface a concept-selective axis. A more promising direction implied
by this pilot: decompose the shift between genuine co-occurring evidence
(e.g. G11 vs. G10, as Exp6's bat-removal counterfactual already probes)
rather than the adversarially *induced* shift, since that shift is
concept-driven by construction. That redesign is out of scope for this
pilot and would need its own audit before starting.

## Reproduction Commands

All commands assume `cwd =
src/experiments/CooccurrenceHallucinationDiagnostic/src/adversarial_signal_debiasing_pilot`
and the `py3_11` conda environment (`/opt/anaconda3/envs/py3_11/bin/python`).

```bash
# Step 0 (already done) -- see audit/repository_audit.md

# Part I-II: reuse dataset + adversarial exposure from the prior pilot
python scripts/prepare_data.py --config configs/data.yaml

# Part III: Layer-19 representation shift
python scripts/extract_layer19_shifts.py \
    --data-config configs/data.yaml --decomp-config configs/decomposition.yaml --device cuda:0

# Part IV (+ VIII, PLS run because PCA showed weak selectivity):
python scripts/decompose_signal.py \
    --data-config configs/data.yaml --decomp-config configs/decomposition.yaml --with-pls

# Part V-VII: component functional selectivity test + selection
python scripts/evaluate_components.py \
    --data-config configs/data.yaml --decomp-config configs/decomposition.yaml --device cuda:0

# Part IX-X: train Model C (Adv+Decomp Debias)
# NOTE: configs/training_adv_decomp.yaml already reflects the fallback loss_mode
# (spur_only failed -- see checkpoints/adv_decomp_debias_spur_only_FAILED/)
python scripts/train_adv_decomp_debias.py \
    --data-config configs/data.yaml --decomp-config configs/decomposition.yaml \
    --train-config configs/training_adv_decomp.yaml --device cuda:0

# Part XI: evaluate all 4 models on the same CLEAN TEST split
python scripts/evaluate_all_models.py --data-config configs/data.yaml --device cuda:0

# Statistics + figures
python analysis/statistics.py --data-config configs/data.yaml
python analysis/visualization.py --data-config configs/data.yaml

# Or run the above as one pipeline (excluding the audit and the failed attempt):
python scripts/run_full_pilot.py --device cuda:0 --with-pls

# Notebook (loads saved outputs only, does not rerun anything):
jupyter nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.kernel_name=py3_11 \
    notebooks/adversarial_signal_debiasing_pilot.ipynb
```
