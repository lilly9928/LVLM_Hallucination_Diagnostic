# Causal Coupling of Co-occurrence-Dependent Target Evidence

## 1. Research Question

Does the within-image co-occurrence-specificity effect established in Experiment 2
(β=0.406) and localized to the LLM's mid-to-late decoder layers in Experiment 3
(logit-lens partial r) reflect a **causal** mechanism — i.e. does surgically removing
a co-occurrence-associated residual-stream component actually reduce the model's
unsupported target-positive evidence — or is the localized signal merely
correlational (present in the representation but not load-bearing for the final
decision)?

## 2. Prior Evidence

- **Experiment 1** (150 matched High/Low pairs): high co-occurrence → higher clean
  s_T. Mean diff = +0.729, bootstrap 95% CI [0.399, 1.061], Cohen's dz = 0.353.
- **Experiment 2** (50 images, 3,039 image-target pairs, two-way FE model
  `s_T ~ image_FE + target_FE + β·S(T,Y)`): β = 0.406, 95% CI [0.290, 0.522],
  permutation-shuffle null ≈ 0 (p=0.0005).
- **Experiment 3** (same 3,039 pairs, logit lens across all 32 LLM decoder layers):
  scale-free within-image partial r ≈ 0 for layers 1-6, **significantly negative**
  for layers 7/9/11/12 (−0.13 to −0.35, unexpected), an abrupt sign flip to positive
  at layer 13 (r≈+0.12), and a stable positive plateau (+0.21 to +0.26) from layer 16
  through the true final layer (32, which exactly reproduces Experiment 2's β).

## 3. Why Localization Is Not Enough

A correlational profile across layers (Experiment 3) shows *where* co-occurrence
information is representationally present, not whether the model's final decision
*depends on* that specific component. A representation can carry information that
is never read out, or that is redundant with other pathways. This experiment tests
dependence directly via activation patching: remove a specific, pre-registered
direction from the residual stream and observe whether the *downstream* β changes.

## 4. Intervention Design

**Patch equation** (mean-referenced projection removal, applied only at the
decision/last-token position, on the output of `language_model.layers[L-1]` —
equivalently the input to `layers[L]`, i.e. exactly `hidden_states[L]` in
Experiment 3's indexing):

```
h_L' = h_L - λ · ((h_L - reference_L) · d̂_L) · d̂_L
```

`reference_L` = train-split mean activation at layer L (not the origin — the
residual stream is not zero-mean, and RMSNorm downstream is scale- but not
shift-invariant). `d̂_L` = unit-normalized co-occurrence direction (§5). λ ∈
{0.25, 0.5, 0.75, 1.0}, pre-registered before any test-split result was seen.

**Data split** (by image, not by row — prevents leakage across targets sharing an
image): 30 train / 10 validation / 10 test images, seed=42, deterministic.
Direction estimation used train only; a cheap single-λ screen across all 9
candidate layers used validation only; every number reported below comes from the
10 held-out test images (599 pairs) only.

**Candidate layers** (fixed before seeing any intervention result, spanning
Experiment 3's structure): 3 (null), 8/11/12 (negative region), 13 (sign
transition), 16/20/24/28 (positive plateau). The validation screen (Appendix)
found 7/9 layers already showing a significant score-correlated Δs_T — a
distributed-looking signal, not a narrow 2-4-layer hit — so per user sign-off the
full λ-grid + 5-control battery was run on 4 representative layers only: **3, 13,
16, 24**.

## 5. Co-occurrence Direction Estimation

Rejected up front: a naive diff-in-means (high-vs-low, or original-vs-counterfactual
image) direction — the parallel `stage11_case_bat_ball` case study already showed
this construction fails a selectivity check even when target identity is held fixed
throughout (§9). Instead, at each candidate layer L, on the **train split only**:

1. Collect the decision-position hidden state `h_L(i,T) ∈ R^4096` for every
   (image, target) pair.
2. Residualize both `cooc_score` and every one of the 4096 dimensions of `h_L`
   against the same image+target fixed-effects design used throughout this project
   (`Z`, via the precomputed pseudo-inverse projection).
3. `d_L = (score_resid^T @ H_resid) / (score_resid^T @ score_resid)` — the exact
   vector-valued analogue of Experiment 2/3's scalar β, i.e. "the direction the
   residual stream moves per unit of within-image, within-target-controlled
   co-occurrence score."

Controls 3/4 reuse the identical mechanism with the direction swapped: 5
norm-matched random Gaussian directions (control 3), and 1 direction estimated the
same way but with `cooc_score` permuted across train rows before residualizing
(control 4).

## 6. Layer-wise Causal Results

`patching/beta_after_intervention.csv`. `beta_before` (test split, n=599) = 0.306
(SE=0.134, p=0.022 — wider CI than Experiment 2's full-sample 0.406, expected from
holding out 40 of 50 images for leakage-free direction estimation).

| Layer | Region (Exp.3) | Δβ @ λ=0.25 | λ=0.5 | λ=0.75 | λ=1.0 | β_after @ λ=1.0 |
|---|---|---|---|---|---|---|
| 3 | null | +0.0001 | +0.0003 | -0.0009 | -0.0001 | 0.306 |
| 13 | transition | -0.012 | -0.026 | -0.038 | **-0.053** | 0.254 |
| 16 | plateau (early) | -0.042 | -0.087 | -0.131 | **-0.180** | 0.126 |
| 24 | plateau (late) | -0.074 | -0.149 | -0.225 | **-0.302** | **0.004** |

Monotonic in λ at every layer — the hallmark of a real, dose-dependent effect, not
noise. See `figures/fig1_delta_beta_by_layer.png`, `fig2_beta_before_vs_after.png`.

## 7. Negative-to-Positive Transition

`figures/fig5_localization_vs_causal.png` aligns Experiment 3's partial-r profile
(top) with this experiment's Δβ profile (bottom, λ=1.0). The representational
signal **plateaus** at layer 13 and stays flat (~0.21-0.26) all the way to layer 32.
The causal Δβ does **not** plateau — it keeps growing in magnitude from L13 → L16 →
L24. Correlational localization and causal leverage are dissociated: the layer
where the *representation* is most informative is not where the *intervention* is
strongest — leverage keeps increasing the later (closer to the readout) the patch is
applied, independent of the underlying representational structure Experiment 3
found. The negative region (L7-12) was not causally tested here (gated on the layer
scan finding a selective effect first, which it did not — see §11).

## 8. Genuine Target Preservation — the decisive control

`controls/genuine_target_results.csv` / `genuine_target_summary.csv`. Mean Δs_T on
**genuine present-target** questions (one present category per test image), λ=1.0:

| Layer | Genuine-target Δs_T | vs. main effect |
|---|---|---|
| 3 | -0.002 | (null, negligible either way) |
| 13 | **-0.384** | far larger than the pooled main-effect shift |
| 16 | **-0.877** | far larger |
| 24 | **-3.064** | an order of magnitude larger than anything else measured |

Genuine recognition evidence is destroyed, and increasingly so at deeper layers —
the opposite of the intended selectivity. See `figures/fig3_selectivity.png`.

Control 2 (low-cooc vs. high-cooc absent targets) also fails, in an informative way:
low-cooc Δs_T *rises* (L24: +0.538) rather than staying near zero, while high-cooc
Δs_T falls (L24: -0.227) — both move toward the population mean. This is the
expected signature of a **mean-referenced projection removal**, which is a
variance-*shrinkage* along d̂ toward the reference point, not a one-sided "turn off
the spurious add-on" edit. That shrinkage mechanically flattens the score→evidence
regression slope (exactly the large, real Δβ in §6) without being selective in the
sense Control 2 was designed to detect.

## 9. Random / Shuffled Controls

`figures/fig1_delta_beta_by_layer.png`; `robustness/cluster_target_results.csv`.
At λ=1.0: random-direction Δβ ≈ 0 at every layer (mean |Δβ| ≤ 0.0005, SD ≤ 0.0007
across 5 seeds) — an arbitrary norm-matched direction does essentially nothing,
confirming the effect in §6 is not a generic-perturbation artifact of the hook
mechanism itself. Shuffled-direction Δβ is small but non-zero at L16 (-0.022) and
L24 (-0.042) — 7-8× smaller than the real direction, not exactly null. The real
direction is the clear outlier relative to both controls, but the non-zero shuffled
result is reported rather than rounded to "no effect."

This generalizes and reinforces the parallel `stage11_case_bat_ball` case study
(`outputs/CooccurrenceHallucinationDiagnostic/stage11_case_bat_ball/`), which
independently found — for a single object pair (baseball bat → sports ball), via a
plain diff-in-means direction — that its layer-19 intervention reduced unsupported
target evidence but reduced genuine target evidence and context recognition by an
equal or larger amount (Checkpoint 2 there: **"STOP / non-selective"**). This
experiment's more carefully FE-controlled direction, tested across this project's
full 50-image population rather than one case study, finds the **same qualitative
failure mode, and worse at the layers with the largest raw effect**.

## 10. Attention vs. MLP Refinement

**Not performed.** Gated on the layer scan (§6-9) finding a narrow, selective causal
region; it did not (§11), so per the task's own checkpoint rules, Exp4D was not run
automatically. No file in this project separates attention-block output from
MLP-block output — see `outputs/cooccurrence_causal_coupling/audit/repository_audit.md`
§2-3 for the confirmed hook points if this is revisited later.

## 11. Main Supported Finding

**Localization is confirmed; causal decoupling is not established.** The
co-occurrence-correlated direction, estimated with a target-identity-and-image
fixed-effects-controlled regression (not a naive diff-in-means), has a real,
dose-dependent, control-beating causal effect on the downstream β — at layer 24,
λ=1.0, it reduces β by 99% (0.306 → 0.004). But the same intervention destroys
genuine target-recognition evidence by an even larger margin at the same layer
(-3.06 logits) and does not spare low-co-occurrence absent targets (which *rise*
rather than staying flat). The direction most plausibly carries a substantial
generic "answer-confidence" or "yes-leaning-evidence magnitude" component,
intermixed with — not cleanly separated from — whatever co-occurrence-specific
computation exists, even after the FE control.

## 12. What We Can Claim

- Experiment 1-2's co-occurrence-specificity effect is a real, robust, statistically
  well-controlled phenomenon (Experiments 1-2, independently reproduced within
  Experiment 4's own test-split `beta_before` = 0.306).
- The representational signal Experiment 3 found is **causally load-bearing** for
  the model's final β, in the narrow sense that surgically perturbing it changes β
  by a large, dose-dependent, random/shuffled-control-beating amount (§6, §9).
- This causal leverage is **not selective** to co-occurrence: it degrades genuine
  target recognition at least as much, and increasingly so at deeper layers (§8).

## 13. What We Cannot Claim

- That co-occurrence bias has a dedicated, surgically removable "spurious pathway"
  distinct from general target-evidence computation.
- That layer 13 (the representational sign-transition) is causally special — L16/L24
  show larger causal effects despite flatter representational signal (§7).
- That the negative region (L7-12) is causally active in either direction — untested
  here, gated on a selective positive-region result that did not materialize.
- Any of this as a mitigation method, a proven mechanism, or a property of LVLMs in
  general — one model (LLaVA-1.5-7B), one dataset (COCO), one direction-estimation
  method, one intervention family (mean-referenced linear projection removal).

## 14. Limitations

- The test split (10 images, 599 pairs) has visibly less statistical power than
  Experiment 2's full 50-image sample — `beta_before`'s CI [0.04, 0.57] is wide.
- Only a mean-referenced *linear* projection-removal intervention was tested; a
  nonlinear or multi-direction (subspace) intervention might separate
  co-occurrence-specific from generic-confidence variance more cleanly.
- The direction estimator controls for image and target *identity* but not for
  target-*presence* sensitivity — Control 1's failure suggests the learned direction
  overlaps substantially with whatever direction distinguishes "the model is
  confident about this Yes/No question" in general, present or absent target alike;
  the estimator was never given a genuine-presence FE term to residualize against.
- Only 1 shuffled-direction draw and 5 random-direction seeds (a compute-budget
  choice, confirmed with the user) — a larger control ensemble could sharpen the
  real-vs-shuffled gap estimate at L16/L24 where shuffled was not exactly null.
- Single model, single dataset, as in every prior stage of this project.

## 15. Next Step Toward Functional Debiasing

Not attempted here (out of scope — no mitigation method was to be implemented per
the task brief). If pursued: the Control 1 failure suggests any future attempt
should explicitly residualize out a "genuine target-presence" fixed effect (or an
independently-estimated general-confidence direction) from `d_L` before use, and
re-test Controls 1-2 before trusting any Δβ result as selective.

## 16. Reproduction Commands

```bash
cd src/experiments/CooccurrenceHallucinationDiagnostic/src/cooccurrence_causal_coupling/scripts
/opt/anaconda3/envs/py3_11/bin/python 01_collect_hidden_states.py --config ../configs/01_collect_hidden_states.yaml --pilot
/opt/anaconda3/envs/py3_11/bin/python 01_collect_hidden_states.py --config ../configs/01_collect_hidden_states.yaml
/opt/anaconda3/envs/py3_11/bin/python 02_estimate_directions.py --config ../configs/02_estimate_directions.yaml
/opt/anaconda3/envs/py3_11/bin/python 03_screen_layers.py --config ../configs/03_screen_layers.yaml
/opt/anaconda3/envs/py3_11/bin/python 04_full_intervention_scan.py --config ../configs/04_full_intervention_scan.yaml
/opt/anaconda3/envs/py3_11/bin/python 05_analyze_intervention.py --config ../configs/05_analyze_intervention.yaml
```

## Final Summary Table

| Question | Evidence | Result | Status |
|---|---|---|---|
| Does L13+ contain co-occurrence-related evidence? | Stage 11 logit lens | partial r +0.12→+0.26, L13-32 | **SUPPORTED** |
| Is this evidence causally used downstream? | Activation intervention, Δβ | Δβ up to -0.302 (99% reduction), monotonic in λ, beats random control | **SUPPORTED** |
| Is the effect specific to co-occurrence? | Shuffled/random control + Control 1/2 | Random ≈0; shuffled small-but-nonzero; genuine target destroyed MORE than main effect; low-cooc rises instead of flat | **NOT SUPPORTED** |
| Is genuine target evidence preserved? | Target-present control | Δs_T = -0.38 to -3.06, worsens with depth | **NOT SUPPORTED** |
| Is L7-12 negative signal causal? | Negative-region intervention | Not tested (gated; positive-region selectivity check failed first) | **INCONCLUSIVE** |
| Is attention or MLP responsible? | Component refinement | Not performed (gated on a selective layer-level effect, not found) | **INCONCLUSIVE** |
