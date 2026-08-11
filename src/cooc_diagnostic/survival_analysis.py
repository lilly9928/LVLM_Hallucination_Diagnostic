"""Stage 4: survival analysis on epsilon* with right-censoring and matched-pair
structure.

- "already_yes" (baseline hallucination at epsilon=0) is a real event at
  duration=0, not a special case to exclude -- Kaplan-Meier and Cox handle
  duration=0 events natively. It is ALSO reported as its own descriptive rate,
  per the experiment brief ("이미 환각하는 케이스, 별도 집계").
- "censored" (never flipped by epsilon_max) is duration=epsilon_max,
  event_observed=0 -- included in every fit, never dropped.
- The matched-pair design from Stage 2 is reflected by stratifying the Cox
  model on pair_id (equivalent in spirit to a 1:1-matched conditional
  survival analysis): each pair is its own stratum, so only within-pair
  comparisons contribute to the partial likelihood, exactly mirroring what the
  matching was built to control for.
"""

from __future__ import annotations

import math

import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter, WeibullAFTFitter
from lifelines.statistics import logrank_test
from scipy import stats


def build_survival_frame(rows: list[dict], epsilon_max: float) -> pd.DataFrame:
    records = []
    for r in rows:
        status = r["status"]
        if status == "censored":
            duration = epsilon_max
            event_observed = 0
        else:
            duration = float(r["epsilon_star"])
            event_observed = 1
        records.append(
            {
                "pair_id": r["pair_id"],
                "arm": r["arm"],
                "arm_binary": 1 if r["arm"] == "treatment" else 0,
                "duration": duration,
                "event_observed": event_observed,
                "already_yes": status == "already_yes",
                "status": status,
            }
        )
    return pd.DataFrame.from_records(records)


def fit_km_by_arm(df: pd.DataFrame) -> dict[str, KaplanMeierFitter]:
    fits = {}
    for arm, group in df.groupby("arm"):
        kmf = KaplanMeierFitter(label=arm)
        kmf.fit(group["duration"], event_observed=group["event_observed"])
        fits[arm] = kmf
    return fits


def pooled_logrank_test(df: pd.DataFrame) -> dict:
    """Naive (non-stratified) log-rank test -- a sensitivity check only; it
    ignores the matched-pair structure and treats all 300 observations as
    independent, which they are not (some pairs share an image). The
    stratified Cox test below is the primary inferential result.
    """
    treatment = df[df["arm"] == "treatment"]
    control = df[df["arm"] == "control"]
    result = logrank_test(
        treatment["duration"], control["duration"],
        event_observed_A=treatment["event_observed"], event_observed_B=control["event_observed"],
    )
    return {"test_statistic": float(result.test_statistic), "p_value": float(result.p_value)}


def stratified_cox_test(df: pd.DataFrame) -> dict:
    """Primary inferential test: Cox proportional hazards with pair_id as
    strata, so the partial likelihood only compares treatment vs. control
    WITHIN each matched pair -- the direct survival-analysis analogue of the
    matched design Stage 2 built. HR > 1 means the treatment (high
    co-occurrence) arm has higher instantaneous hazard of flipping at any
    given epsilon, i.e. flips at smaller budgets, consistent with the hypothesis.
    """
    cph = CoxPHFitter()
    cph.fit(df[["duration", "event_observed", "arm_binary", "pair_id"]], duration_col="duration", event_col="event_observed", strata=["pair_id"])
    summary = cph.summary.loc["arm_binary"]
    return {
        "hazard_ratio": float(summary["exp(coef)"]),
        "ci_lower": float(summary["exp(coef) lower 95%"]),
        "ci_upper": float(summary["exp(coef) upper 95%"]),
        "p_value": float(summary["p"]),
        "log_likelihood": float(cph.log_likelihood_),
    }


def weibull_aft_time_ratio(df: pd.DataFrame, epsilon_floor: float) -> dict:
    """Secondary effect-size estimate directly on the epsilon scale: the
    Weibull AFT model's time ratio for arm_binary is the multiplicative factor
    by which the treatment arm's typical required epsilon is scaled relative
    to control (e.g. 0.7 = treatment needs 30% smaller epsilon on average).
    Not stratified by pair (AFT strata would require per-stratum shape/scale,
    infeasible with n=2/stratum) -- report alongside, not in place of, the
    stratified Cox result.

    AFT is a log-time model and cannot take duration=0 (already_yes cases);
    those durations are shifted up by `epsilon_floor` -- Stage 3's own search
    floor (eps0), i.e. the finest budget the pipeline actually resolved, rather
    than an arbitrary small constant. Only this AFT fit is affected; Cox/KM
    handle duration=0 natively and are computed on the unshifted durations.
    """
    shifted = df[["duration", "event_observed", "arm_binary"]].copy()
    shifted["duration"] = shifted["duration"] + epsilon_floor
    aft = WeibullAFTFitter()
    aft.fit(shifted, duration_col="duration", event_col="event_observed")
    summary = aft.summary.loc[("lambda_", "arm_binary")]
    return {
        "time_ratio": float(math.exp(summary["coef"])),
        "ci_lower": float(math.exp(summary["coef lower 95%"])),
        "ci_upper": float(math.exp(summary["coef upper 95%"])),
        "p_value": float(summary["p"]),
    }


def paired_bootstrap_median_diff(df: pd.DataFrame, n_boot: int, rng) -> dict:
    """Bootstrap CI for the median(treatment epsilon*) - median(control epsilon*)
    difference, resampling PAIRS with replacement (respects the matched design).
    Restricted to pairs where BOTH arms have an observed event (flipped or
    already_yes) -- a censored side only gives a lower bound on its true
    epsilon*, which cannot be differenced against an exact value without
    biasing the estimate, so such pairs are excluded from this specific
    descriptive statistic (not from the primary Cox test, which handles
    censoring correctly via the partial likelihood).
    """
    wide = df.pivot(index="pair_id", columns="arm", values=["duration", "event_observed"])
    both_observed = wide[("event_observed", "treatment")].eq(1) & wide[("event_observed", "control")].eq(1)
    usable = wide[both_observed]
    n_excluded = len(wide) - len(usable)

    treat_vals = usable[("duration", "treatment")].to_numpy()
    control_vals = usable[("duration", "control")].to_numpy()
    observed_diff = float(pd.Series(treat_vals).median() - pd.Series(control_vals).median())

    n_pairs = len(usable)
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n_pairs, size=n_pairs)
        diffs.append(float(pd.Series(treat_vals[idx]).median() - pd.Series(control_vals[idx]).median()))
    diffs.sort()
    lo = diffs[int(0.025 * n_boot)]
    hi = diffs[int(0.975 * n_boot) - 1]

    return {
        "n_pairs_used": n_pairs,
        "n_pairs_excluded_censored": n_excluded,
        "observed_median_diff": observed_diff,
        "ci_lower": lo,
        "ci_upper": hi,
    }


def paired_mcnemar_test(df: pd.DataFrame, column: str) -> dict:
    """Paired exact test on a boolean per-arm indicator (e.g. already_yes, or
    Stage 6's mentioned_in_caption): the natural test for a paired binary
    outcome is McNemar's, implemented here as the exact binomial test on
    discordant pairs (equivalent to exact McNemar).

    Pairs missing one side (e.g. Stage 6 excluded that side for failing to
    reproduce its closed-question flip) have no valid within-pair comparison
    and are dropped -- not silently, `n_pairs_incomplete_dropped` is reported.
    """
    wide = df.pivot(index="pair_id", columns="arm", values=column)
    n_before = len(wide)
    wide = wide.dropna(subset=["treatment", "control"])
    n_incomplete_dropped = n_before - len(wide)
    wide = wide.astype(bool)

    both_yes = int((wide["treatment"] & wide["control"]).sum())
    both_no = int((~wide["treatment"] & ~wide["control"]).sum())
    treatment_only = int((wide["treatment"] & ~wide["control"]).sum())
    control_only = int((~wide["treatment"] & wide["control"]).sum())

    n_discordant = treatment_only + control_only
    if n_discordant == 0:
        p_value = 1.0
    else:
        p_value = float(stats.binomtest(treatment_only, n_discordant, 0.5).pvalue)

    return {
        f"both_{column}": both_yes,
        "both_not": both_no,
        f"treatment_only_{column}": treatment_only,
        f"control_only_{column}": control_only,
        "p_value": p_value,
        "n_pairs_incomplete_dropped": n_incomplete_dropped,
    }


def mcnemar_already_yes_test(df: pd.DataFrame) -> dict:
    return paired_mcnemar_test(df, "already_yes")


def holm_correction(pvalues: dict[str, float]) -> dict[str, float]:
    """Holm-Bonferroni step-down correction (more powerful than plain
    Bonferroni while still controlling the family-wise error rate) across the
    named p-values (here: the primary epsilon* test and the secondary
    already_yes-rate test -- two distinct hypotheses about the same data)."""
    names = list(pvalues.keys())
    ordered = sorted(names, key=lambda n: pvalues[n])
    m = len(ordered)
    adjusted = {}
    running_max = 0.0
    for rank, name in enumerate(ordered):
        adj = min((m - rank) * pvalues[name], 1.0)
        running_max = max(running_max, adj)
        adjusted[name] = running_max
    return adjusted
