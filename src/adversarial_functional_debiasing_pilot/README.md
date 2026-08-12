# Adversarial Functional Debiasing Pilot

## Research Question

Fixed pair: context C = "baseball bat", target T = "sports ball" (LLaVA-1.5-7B).

If we train on adversarial examples that strongly expose the bat→ball hallucination
shortcut, does the spurious dependency decrease on unseen **clean** test images more
than a comparable clean-only debiasing baseline, while genuine sports-ball and
baseball-bat recognition remain preserved?

This is a feasibility pilot only — not a final method.

## Motivation

Stage 11 of this repository already established that G10 (bat+, ball−) images need
only ~22% of the perturbation budget that matched G00 (bat−, ball−) controls need to
flip "Is there a sports ball?" to Yes (median ε\*=0.0001148 vs 0.0005203, Cox
HR=1.760, p=0.024), and that clean (ε=0) `s_ball` evidence is already meaningfully
higher for G10 than G00 (Cohen's d=0.745). This pilot asks whether *exposing* that
shortcut adversarially during training — rather than only supervising on clean
negatives — gives a stronger debiasing signal.

## Experimental Setup

- Model: `llava-hf/llava-1.5-7b-hf`, LoRA on `language_model` `q_proj`/`v_proj`
  only (64 modules across 32 decoder layers), r=16, alpha=32, dropout=0.05,
  lr=1e-4, 3 epochs, batch size 1, grad-accum 4 (≈33 optimizer steps).
  8,388,608 trainable params / 7,071,815,680 total (0.1186%).
- Reused unchanged from this repo: `llava_runtime.py` (loading, prompt
  construction, `yes_no_logits`/`yes_no_margin`), `pgd_attack.py`
  (`pgd_attack_with_restarts`), `coco_index.py`, and Stage 11's existing
  `exp0_group_membership.csv` (G00/G10/G01/G11 partition of val2017).
- New code (this directory): data split, adversarial forget-set generation,
  minimal LoRA SFT loop, evaluation, statistics, figures.

## Data Split

Built from Stage 11's existing group membership (G00/G10/G01/G11 are mutually
exclusive/exhaustive over val2017, so splitting each group independently and
combining guarantees zero image-ID overlap between train and test). Seed=42.

| Role | Train (unique images) | Test (unique images) |
|---|---:|---:|
| G10 (forget: bat+, ball−) | 45 | 20 |
| GT (target retention: ball+) | 45 | 25 |
| GC (context retention: bat+) | 45 | 20 |
| G00 (bat−, ball−, test-only) | — | 25 |

GT train/test = G11 (bat+ball+, 22 train / 10 test of 32 total) + a 38-image
subsample of G01 (bat−ball+, 23 train / 15 test). GC = the G10 split itself
(same images serve as both the forget-eval set and the context-retention-eval
set under a *different question* — same train/test fold, never crossed).
Verified programmatically: `train_ids ∩ test_ids = ∅`.

G10's own pool (65 total val2017 images) was the limiting resource; the chosen
45/20 split lands inside the requested 40–50 / 20–30 ranges with no leftover
images unused.

## Adversarial Exposure

Fixed epsilon = 16/255 = 0.0627, chosen **before** inspecting any debiasing
result, reusing Stage 3/11's own already-documented `sanity_check_epsilon`
("generous budget for sanity checks", ~550× the measured median ε\*=0.0001148
for this exact G10 pool). PGD: 20 steps, step-size multiplier 2.5, 2 restarts —
identical hyperparameters to Stage 3/11, not retuned.

| n | attack success rate | mean clean s_ball | mean adv s_ball | mean Δ |
|---:|---:|---:|---:|---:|
| 45 | 1.000 | −0.4240 | 1.0531 | 1.4771 |

All 45 TRAIN G10 images flipped the differentiable margin proxy at this fixed
epsilon. See `fig1_adversarial_exposure.png`.

## Training Variants

Two LoRA runs, identical code/hyperparameters/seed/retain-data, differing
**only** in which image backs the forget examples:

- **Clean Debias**: forget images = clean G10 train images.
- **Adv Debias**: forget images = the adversarial images above.

Retain data (identical for both, always clean): GT train → "Is there a sports
ball?" / "Yes"; GC train → "Is there a baseball bat?" / "Yes". Mixture ratio
1 forget : 1 retain-target : 1 retain-context per optimizer step, fixed before
seeing any test result. Ordinary supervised cross-entropy (labels masked to
−100 over the prompt span; the answer span is the model's own detected Yes/No
decision point, teacher-forced) — no custom unlearning objective.

## Clean Test Evaluation

All three models (Original, Clean Debias, Adv Debias) evaluated **only** on the
unseen clean test split above — never on adversarial or training images.

## Main Results

| Method | Coupling B ↓ | G10 s_ball ↓ | Ball+ Acc ↑ | Bat+ Acc ↑ |
| ------------ | -----------: | -----------: | ----------: | ---------: |
| Original     |  1.9369 |  −0.0875 | 0.880 | 0.950 |
| Clean Debias |  0.3205 |  −0.4258 | 0.920 | 1.000 |
| Adv Debias   | **−0.1823** |  −0.4867 | 0.840 | 1.000 |

Absolute change from Original:

| Method | ΔB | ΔG10 s_ball | ΔBall+ Acc | ΔBat+ Acc |
|---|---:|---:|---:|---:|
| Clean Debias | −1.6164 | −0.3383 | +0.040 | +0.050 |
| Adv Debias | −2.1192 | −0.3992 | −0.040 | +0.050 |

Adv vs Clean: ΔB = −0.5028 (Adv lower); paired-bootstrap difference on G10
`s_ball` itself (same 20 test images, both models): observed −0.061, 95% CI
[−0.631, 0.439] — **includes zero**. The point-estimate ordering favors Adv,
but this pilot's sample size cannot statistically distinguish Adv from Clean
on G10 alone (see Limitations).

Bootstrap 95% CIs (unpaired, resampling within each group):

| Method | Coupling B CI |
|---|---|
| Original | [1.363, 2.439] |
| Clean Debias | [−0.692, 1.315] |
| Adv Debias | [−0.649, 0.240] |

Paired bootstrap on G10 `s_ball` (same 20 images across models):

| Comparison | Observed | 95% CI |
|---|---:|---|
| Adv − Original | −0.399 | [−0.677, −0.116] (excludes 0) |
| Clean − Original | −0.338 | [−0.751, 0.121] (includes 0) |
| Adv − Clean | −0.061 | [−0.631, 0.439] (includes 0) |

Only Adv Debias's reduction vs. Original on G10 itself is statistically
distinguishable from zero at this sample size.

## Selectivity

| Method | R_S = B_Orig − B_method | D_T = Acc_Ball,Orig − Acc_Ball,method | D_C = Acc_Bat,Orig − Acc_Bat,method |
|---|---:|---:|---:|
| Clean Debias | 1.6164 | −0.040 (improved) | −0.050 (improved) |
| Adv Debias | **2.1192** | +0.040 | −0.050 (improved) |

Adv Debias's Ball+ accuracy drop (4 pp: 22/25 correct under Original vs. 21/25
under Adv Debias, i.e. 1 additional misclassified image) is well under the
10-pp descriptive threshold used for the non-selectivity check; Bat+ accuracy
did not degrade for either variant.

## Go / No-Go Decision

**GO**: B_Adv (−0.182) < B_Clean (0.320) < B_Original (1.937). Retention did
**not** stay flat: Adv Debias's Ball+ accuracy actually *decreased* 4pp versus
Original (0.880 → 0.840, i.e. 1 additional misclassified image out of 25), and
Clean Debias's *increased* 4pp (0.880 → 0.920). Bat+ accuracy did not decrease
for either variant (0.950 → 1.000 for both). "Retention preserved" below means
these changes stay within the pre-registered 10-pp non-selectivity threshold —
not that accuracy was literally unchanged.

## Interpretation

On this single fixed pair, adversarial exposure produced the strongest and only
statistically-confirmed-vs-Original reduction in the bat→ball functional
coupling on unseen clean images, without collapsing genuine ball or bat
recognition. However, the Adv-vs-Clean point-estimate ordering (the pattern
that would motivate adversarial exposure specifically, rather than any
debiasing at all) is **not** statistically significant at n=20 G10 test images
— the paired 95% CI for Adv−Clean on G10 `s_ball` spans −0.631 to +0.439. The
directional GO pattern replicated the desired ordering, but this pilot's own
numbers do not yet establish that adversarial exposure is *necessary* over
clean negatives; they are consistent with it being *helpful* only in degree.

## Limitations

- n=20 G10 test images is small; CIs on the primary metric are wide.
- Single trigger/target pair, single model, single epsilon, single LoRA config,
  single seed — no variance-across-seed estimate.
- G10's own pool (65 val2017 images total) is the same pool from which the
  retain-context (GC) role and Stage 11's own analyses are drawn; images
  double as forget-eval and context-retention-eval targets, which is by
  design (same fold, different question) but means the pilot's context-
  retention signal is not independent of the forget-training distribution.
- "Attack success" here means the differentiable PGD margin proxy reached
  >0, not the authoritative greedy-decoded flip (training images were not
  re-verified with `generate_greedy_answer`, per the fixed-epsilon design that
  intentionally skips per-image epsilon\* search — see task spec's "Critical
  Interpretation": adversarial images are training-exposure data only, never
  evaluation evidence).
- The 10-pp retention-drop threshold used for the selectivity check is a
  descriptive convenience picked before running (not derived from any prior
  distribution) and should not be read as a validated significance criterion.

## Next Step

If pursued further: repeat with multiple seeds to get a real Adv-vs-Clean
variance estimate; test a second trigger/target pair to check the coupling-
reduction pattern isn't specific to bat/ball; consider whether a larger G10
test pool (relaxing to include some train2017 bat+/ball− images) would
tighten the primary CI enough to resolve the Adv-vs-Clean question this pilot
left open.

## Reproduction Commands

```bash
cd /data3/KJE/code/UQ/src/experiments/CooccurrenceHallucinationDiagnostic/src/adversarial_functional_debiasing_pilot
PY=/opt/anaconda3/envs/py3_11/bin/python

$PY scripts/build_split.py --config configs/pilot.yaml
$PY scripts/generate_adversarial_forget_set.py --config configs/pilot.yaml --device cuda:0

$PY training/train_lora_debias.py --config configs/pilot.yaml --variant clean --device cuda:0
$PY training/train_lora_debias.py --config configs/pilot.yaml --variant adv   --device cuda:2

$PY scripts/evaluate_model.py --config configs/pilot.yaml --model original     --device cuda:0
$PY scripts/evaluate_model.py --config configs/pilot.yaml --model clean_debias --device cuda:2
$PY scripts/evaluate_model.py --config configs/pilot.yaml --model adv_debias   --device cuda:3

$PY analysis/compute_statistics.py --config configs/pilot.yaml
$PY analysis/make_figures.py --config configs/pilot.yaml
$PY analysis/qualitative_examples.py --config configs/pilot.yaml
```

Outputs land under `/data3/KJE/code/UQ/outputs/adversarial_functional_debiasing_pilot/`.
