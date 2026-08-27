"""Off-policy evaluation, judged by COVERAGE rather than by a point estimate.

An estimator that lands near the truth on the seed its author looked at has
demonstrated nothing. The question is whether the interval it ships means what it
says, so the headline here is a measured coverage rate over 240 independent seeds
against a truth obtained by actually running the candidate policy. If a nominal 95%
interval covers at 88%, that number gets published as the finding.

Everything else in this file exists to keep that measurement honest: the anchor that
must hold exactly, the positivity refusal that must fire before any number is emitted,
and the divergence axis that says WHERE the estimator is allowed to be trusted.
"""

import numpy as np
import pytest
from prompt_experiments.stats.proportions import wilson_interval

from route_regret.bench import run
from route_regret.fixture import REGISTRY, WorkloadSpec, build_workload
from route_regret.metrics import paired_bootstrap
from route_regret.offpolicy import (ConstantReward, EpsilonExploring, collect,
                                    counterfactual_cost, fit_reward_model, ips_cost,
                                    measure_coverage, positivity, quality_direct,
                                    quality_dr, quality_ips, quality_snips, tv_divergence)
from route_regret.policies import AlwaysCheapest, AlwaysTop, ContentBlind, ThresholdLadder
from route_regret.report import DEFAULT_LADDER

N = 3000
EPSILON = 0.5          # pre-registered before any coverage number was looked at
COVERAGE_SEEDS = 240
COVERAGE_N = 800
CANDIDATE_OFFSET = 0.12


def _ladder(offset: float = 0.0) -> ThresholdLadder:
    return ThresholdLadder(DEFAULT_LADDER, offset=offset)


def _explorer(epsilon: float = EPSILON, offset: float = 0.0) -> EpsilonExploring:
    return EpsilonExploring(_ladder(offset), epsilon)


def _logged(policy, *, n=N, verify_rate=0.0, seed_tag="ope", signal=0.70):
    spec = WorkloadSpec(mix="balanced", n=n, signal=signal, seed_tag=seed_tag)
    cases = build_workload(spec)
    led = run(policy, cases, spec, verify_rate=verify_rate)
    return cases, spec, collect(led, cases, spec, verify_rate=verify_rate), led


# --------------------------------------------------------------------------------
# The instrument itself: if the logged propensities are not the sampling
# distribution, every weighted estimate below is arithmetic on a fiction.
# --------------------------------------------------------------------------------

def test_the_logged_propensity_is_the_distribution_the_logger_actually_sampled():
    """IPS divides by a number the log claims is P(action | context). If that claim is
    off, every downstream interval is wrong in a way no amount of data fixes.

    Checking the MARGINAL is not that test, and shipping only the marginal was the hole
    here. A propensity that is right on average and wrong per context passes it, while
    IPS divides by the conditional: with the logger's declared propensity decorrelated
    from its own draw (the base policy evaluated at a shifted seed_tag), 822 of 3000
    logged propensities stopped matching the sampling law and all five marginals still
    sat inside their Wilson intervals -- and the 95% coverage headline still read
    calibrated at 93.8%. The second block is the one that fires. Every (context, action)
    the logger scores at probability p is a Bernoulli(p) trial, so grouping the logged
    draws by the declared p and reading the realised frequency back is a statement about
    the conditional law rather than about its average.
    """
    logger = _explorer()
    cases, spec, log, _ = _logged(logger)

    for card in REGISTRY:
        taken = sum(1 for r in log.records if r.model == card.name)
        claimed = np.mean([logger.propensity(c, spec, card) for c in cases])
        iv = wilson_interval(taken, len(log.records))
        assert iv.low <= claimed <= iv.high, (
            f"{card.name}: logger claims p={claimed:.4f} but took it "
            f"{taken}/{len(log.records)} = {iv}")

    # Rounded because the declared values are arithmetic on floats (eps/|A| + 1 - eps is
    # 0.6000000000000001); the grouping is over the logger's distinct probability levels,
    # not over a discretisation of a continuous score.
    trials: dict[float, list[int]] = {}
    for rec in log.records:
        for card in REGISTRY:
            p = round(logger.propensity(rec.case, spec, card), 9)
            trials.setdefault(p, []).append(1 if rec.model == card.name else 0)

    # A logger that declared one flat probability everywhere would make the block below
    # a restatement of the marginal check, so the precondition is asserted rather than
    # assumed: the exploring logger must put mass at more than one level.
    assert len(trials) > 1, f"only one declared probability level: {sorted(trials)}"
    for p, drawn in sorted(trials.items()):
        iv = wilson_interval(sum(drawn), len(drawn))
        assert iv.low <= p <= iv.high, (
            f"the logger declares p={p:.4f} on {len(drawn)} (context, action) pairs but "
            f"was drawn on {sum(drawn)} of them = {iv}; the propensity is calibrated on "
            f"average and wrong per context, which is the case IPS cannot survive")


# --------------------------------------------------------------------------------
# Cost is a closed-form plug-in. Quality is not.
# --------------------------------------------------------------------------------

def test_the_closed_form_cost_estimator_is_exact_and_needs_no_reweighting():
    """Under a DETERMINISTIC candidate the counterfactual bill is not an estimate at
    all: the model it would have picked and the tokens it would have spent are both in
    the log. Reaching for IPS here buys an interval around a quantity that has no
    sampling error, which is how a pipeline reports uncertainty it does not have."""
    candidate = _ladder(0.20)
    cases, spec, log, _ = _logged(_explorer(), verify_rate=0.05)

    est = counterfactual_cost(log, candidate)
    truth = run(candidate, cases, spec, verify_rate=0.05).total_cost()

    assert est.identifiable
    assert est.total == pytest.approx(truth, abs=1e-9), (
        f"closed form ${est.total:.6f} vs actually running it ${truth:.6f}")
    # And the IPS estimate of the SAME quantity is not exact, which is the whole point.
    ips = ips_cost(log, candidate)
    assert abs(ips.total - truth) > 1e-6, (
        "IPS landing exactly on the truth would mean the weighting is inert")


def test_closed_form_cost_has_a_materially_tighter_interval_than_ips():
    """Same estimand, two estimators. The IPS interval is wider because it throws away
    every logged case where the candidate disagreed with the logger and then inflates
    the survivors. 'Materially' is stated as a 2x bar, and the measured ratio is
    reported beside the design effect that predicts it."""
    candidate = _ladder(0.20)
    _, _, log, _ = _logged(_explorer(), verify_rate=0.05)

    closed = counterfactual_cost(log, candidate)
    ips = ips_cost(log, candidate)
    ratio = ips.width / closed.width

    assert ratio > 2.0, (
        f"IPS interval ${ips.width:.6f} vs closed form ${closed.width:.6f} "
        f"= {ratio:.2f}x, below the stated 2x materiality bar")
    predicted = np.sqrt(len(log.records) / ips.n_effective)
    assert 0.5 < ratio / predicted < 2.0, (
        f"width ratio {ratio:.2f}x should sit near the design-effect prediction "
        f"{predicted:.2f}x; a large gap means the inflation is coming from somewhere "
        f"other than the reweighting")


def test_self_normalising_is_a_choice_that_was_measured_not_assumed():
    """Why the shipped quality estimator divides by the realised weight mass.

    Plain IPS divides by the row count instead, which assumes the weights sum to n. They
    do in expectation and they do not in any finite sample, so the estimate inherits
    that miss as bias. Both estimators are run over the same seeds and both failure
    modes are measured; the self-normalised one also carries an interval, and the plain
    one deliberately does not, because its support is unbounded above and a Wilson
    interval on it would be a category error dressed as a number.
    """
    candidate = _ladder(0.20)
    logger = _explorer()
    sn_err, ips_err = [], []
    for k in range(30):
        cases, spec, log, _ = _logged(logger, n=1500, seed_tag=f"sn{k}")
        truth = 1.0 - run(candidate, cases, spec).violation_rate()
        sn_err.append(quality_snips(log, candidate, logging_policy=logger).value - truth)
        ips_err.append(quality_ips(log, candidate, logging_policy=logger).value - truth)
    sn_err, ips_err = np.array(sn_err), np.array(ips_err)
    sn_rmse = float(np.sqrt((sn_err ** 2).mean()))
    ips_rmse = float(np.sqrt((ips_err ** 2).mean()))

    assert ips_rmse > 1.5 * sn_rmse, (
        f"self-normalising should materially beat plain IPS: RMSE {sn_rmse:.5f} vs "
        f"{ips_rmse:.5f} = {ips_rmse / sn_rmse:.2f}x, under the stated 1.5x bar")

    # What the gap is NOT. This file previously asserted `abs(ips_err.mean()) >
    # abs(sn_err.mean())` and called it "the un-normalised estimator carries the larger
    # bias". Both means are indistinguishable from zero -- plain IPS is unbiased for the
    # candidate's success rate on the logged cases whenever the propensity is right, and
    # over 300 seeds its mean error is +0.0035 against a Monte-Carlo standard error of
    # 0.0037. The old assertion held because the larger-variance estimator has the larger
    # noise in its sample mean, so it was the RMSE line above wearing a second hat.
    # Stated properly it needs an interval, and the interval comes from the repo's own
    # deterministic bootstrap rather than from a t-statistic written here.
    for name, err in (("snips", sn_err), ("ips", ips_err)):
        point, lo, hi = paired_bootstrap(list(err), [0.0] * len(err))
        assert lo <= 0.0 <= hi, (
            f"{name} mean error {point:+.5f} with 95% bootstrap interval [{lo:+.5f}, "
            f"{hi:+.5f}] excludes zero: this estimator has a bias these seeds can see, "
            f"and the variance story above is not the whole story")

    plain = quality_ips(log, candidate, logging_policy=logger)
    assert plain.identifiable and plain.low is None and plain.high is None, (
        "plain IPS must ship without an interval; giving it a plausible-looking one is "
        "how the wrong estimator gets adopted")


# --------------------------------------------------------------------------------
# The anchor that must hold exactly.
# --------------------------------------------------------------------------------

def test_the_estimator_recovers_the_truth_exactly_when_the_candidate_is_the_logger():
    """The sanity anchor. Under a deterministic logger every propensity is 1, every
    weight is 1, and the reweighted estimate must collapse to the plain sample mean.
    Any drift here is a bug in the weighting, not sampling noise -- so the tolerance
    is float exactness, not a confidence interval."""
    logger = _ladder()
    cases, spec, log, led = _logged(logger)

    est = quality_snips(log, logger, logging_policy=logger)
    truth = 1.0 - led.violation_rate()

    assert est.identifiable, est.reason
    assert est.value == pytest.approx(truth, abs=1e-12), (
        f"estimate {est.value!r} vs truth {truth!r}")
    assert est.n_effective == pytest.approx(len(log.records), abs=1e-9), (
        "with unit weights the effective sample size IS the sample size")

    cost = counterfactual_cost(log, logger)
    assert cost.total == pytest.approx(led.total_cost(), abs=1e-9)

    # The same anchor for the IPS control, on a log that actually carries verify and
    # judge spend -- and it has to be a verifying log, because at verify_rate=0 the
    # overhead column is zero and an estimator that silently dropped it would look right.
    # Without this the only assertion on ips_cost is the one-sided "it is not exact",
    # which gets EASIER the more wrong the control is: pricing only the routed call
    # (12.5% off the true bill instead of 5.6%) passed the whole file.
    _, _, vlog, vled = _logged(logger, verify_rate=0.05, seed_tag="anchor-v")
    overhead = sum(r.cost_usd for r in vled.rows if r.call_kind != "route")
    assert overhead > 0.05 * vled.total_cost(), (
        f"precondition: the anchor log must carry material verify+judge spend, got "
        f"${overhead:.4f} of ${vled.total_cost():.4f}")
    assert ips_cost(vlog, logger).total == pytest.approx(vled.total_cost(), abs=1e-9), (
        "with unit weights the importance-weighted bill IS the logged bill, overhead "
        "included; a control that estimates a different quantity is not a control")

    # The shipped default confidence, pinned here because the coverage sweep stopped
    # pinning it: that sweep now passes `confidence=nominal` explicitly (it previously
    # did not, which was the bug), so nothing else would notice the default drifting off
    # 95% -- and every other assertion on an interval width in this file is scale-free.
    assert est.width == quality_snips(log, logger, logging_policy=logger,
                                      confidence=0.95).width, (
        "quality_snips must ship a 95% interval by default")
    assert est.width < quality_snips(log, logger, logging_policy=logger,
                                     confidence=0.99).width


# --------------------------------------------------------------------------------
# Positivity is a precondition, not a footnote.
# --------------------------------------------------------------------------------

def test_a_deterministic_logger_makes_most_counterfactuals_unidentifiable():
    """The reason production routing logs are usually worthless for this. A logger that
    always picks one model per context observes nothing about the others, and no
    reweighting invents it. The count is measured over a slate of candidates rather
    than asserted, and the logger's own policy must come back identifiable or the check
    is refusing everything and proving nothing."""
    logger = _ladder()
    cases, spec, log, _ = _logged(logger, n=1500)

    marginal = {m.name: 1.0 / len(REGISTRY) for m in REGISTRY}
    slate = {
        "logger_itself": logger,
        "ladder+0.05": _ladder(0.05),
        "ladder+0.20": _ladder(0.20),
        "ladder-0.20": _ladder(-0.20),
        "always_top": AlwaysTop(),
        "always_cheapest": AlwaysCheapest(),
        "content_blind": ContentBlind(marginal),
    }
    verdicts = {name: quality_snips(log, pol, logging_policy=logger)
                for name, pol in slate.items()}

    refused = [n for n, e in verdicts.items() if not e.identifiable]
    assert len(refused) > len(slate) / 2, (
        f"a deterministic logger should refuse most counterfactuals, refused "
        f"{len(refused)}/{len(slate)}: {refused}")
    assert verdicts["logger_itself"].identifiable, (
        "the logging policy's own value is always identifiable; a check that refuses "
        "it too is refusing everything and is not a positivity check")
    assert all(e.value is None for n, e in verdicts.items() if not e.identifiable), (
        "an unidentifiable estimand must come back as no number at all, not as a "
        "number with a caveat attached")
    # And it must not RENDER as one either. A refusal that prints in the same column
    # layout as an estimate gets read as an estimate by everyone who skims the table.
    refused_line = verdicts[refused[0]].line()
    assert "not identifiable" in refused_line
    assert not any(ch.isdigit() for ch in refused_line.split("not identifiable")[0]), (
        f"a refusal must not render a figure where an estimate would sit: {refused_line!r}")


def test_positivity_names_the_cell_it_could_not_estimate():
    """'Not identifiable' is only actionable if it says where the hole is. The report
    has to name a (context cell, action) pair the logger never produced -- that pair is
    the exploration the log would have needed."""
    logger = _ladder()
    _, _, log, _ = _logged(logger, n=1500)

    report = positivity(log, AlwaysCheapest(), logging_policy=logger)
    assert not report.identifiable
    assert report.supported_fraction < 1.0
    assert report.unsupported, "refusing without naming the missing cell is not a diagnostic"
    missing_actions = {action for _, action in report.unsupported}
    assert missing_actions == {AlwaysCheapest().model.name}, (
        f"the hole is the cheapest model in the hard cells, got {missing_actions}")

    # Exploration is exactly what closes it: the same candidate under an exploring
    # logger is identifiable, which is the actionable form of the refusal.
    explorer = _explorer()
    _, _, elog, _ = _logged(explorer, n=1500)
    assert positivity(elog, AlwaysCheapest(), logging_policy=explorer).identifiable


def test_support_can_be_established_from_the_log_alone_when_the_logger_is_unknown():
    """The case that actually arrives: a log from a system nobody instrumented.

    Without the logging policy as an object there are no propensities to read, so
    support has to come from what was OBSERVED -- did this cell ever see this action.
    That check is strictly weaker than the exact one and must be reported as such, but
    it still has to fire on a hole this obvious, or an uninstrumented log would sail
    through the gate that exists to stop it.
    """
    _, _, det_log, _ = _logged(_ladder(), n=1500)
    _, _, exp_log, _ = _logged(_explorer(), n=1500, seed_tag="ope-x")

    blind = positivity(det_log, AlwaysCheapest())
    assert not blind.identifiable and blind.unsupported
    assert "observed actions" not in blind.reason  # refusals name the hole, not the basis
    assert positivity(exp_log, AlwaysCheapest()).identifiable, (
        "an exploring logger leaves the cheapest model observed in every cell, so the "
        "empirical check should clear it too")
    # The exact check is the one that gets used when the logger is known, and on this
    # log the two agree -- stated so a future divergence between them is visible.
    assert (positivity(det_log, AlwaysCheapest(), logging_policy=_ladder()).identifiable
            == blind.identifiable)

    # And the two branches must report their evidence differently, because they have
    # different evidence. The exact branch has a smallest propensity and it is arithmetic,
    # not a fitted number: the candidate's only action is the cheapest model, which the
    # explorer reaches at eps/|A| wherever its base would have chosen something else. The
    # empirical branch never reads a propensity at all -- and printed "min propensity
    # 1.0000" anyway, the initialiser leaking into a sentence, while the struct beside it
    # reported 0.0 for the same report.
    exact = positivity(exp_log, AlwaysCheapest(), logging_policy=_explorer())
    assert exact.identifiable
    assert exact.min_propensity == pytest.approx(EPSILON / len(REGISTRY), abs=1e-12), (
        f"the explorer's floor is eps/|A|={EPSILON / len(REGISTRY):.4f}, report says "
        f"{exact.min_propensity:.4f}")
    assert f"min propensity {exact.min_propensity:.4f}" in exact.reason
    empirical = positivity(exp_log, AlwaysCheapest())
    assert empirical.min_propensity == 0.0 and "min propensity" not in empirical.reason, (
        f"the empirical branch must not render a propensity it never read: "
        f"{empirical.min_propensity!r}, {empirical.reason!r}")


def test_positivity_is_checked_over_the_candidates_distribution_not_its_sample():
    """A rare arm is where positivity fails quietly, and where the obvious check misses it.

    A randomised candidate that plays one model 99.98% of the time still has an estimand
    that INTEGRATES over the rare arm, so the log needs data there. Checking support at
    the action the candidate happened to draw on each logged case is not the same
    question, and on a sample this size it never draws the rare arm at all -- so the
    sampled check calls the estimand identifiable while the log contains nothing about
    the arm the answer partly depends on. The precondition is asserted rather than
    assumed, so a fixture change that starts drawing the rare arm makes this test say so
    instead of passing for the wrong reason.
    """
    rare = "gpt-4o-mini"
    logger = AlwaysTop()
    cases, spec, log, _ = _logged(logger, n=1500)
    candidate = ContentBlind({logger.model.name: 0.9998, rare: 0.0002})

    drawn = {candidate.choose(c, spec).name for c in cases}
    assert rare not in drawn, (
        "precondition: this candidate must never SAMPLE the rare arm on these cases, "
        "or the sampled check would catch the hole and the distributional check would "
        "be untested")
    assert candidate.propensity(cases[0], spec,
                                next(m for m in REGISTRY if m.name == rare)) > 0

    report = positivity(log, candidate, logging_policy=logger)
    assert not report.identifiable, (
        "an arm the log never covers makes the estimand unidentifiable even when the "
        "candidate never happened to draw it")
    assert rare in {action for _, action in report.unsupported}


# --------------------------------------------------------------------------------
# THE HEADLINE: measured coverage.
# --------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def coverage():
    return measure_coverage(lambda: _explorer(), lambda: _ladder(CANDIDATE_OFFSET),
                            seeds=COVERAGE_SEEDS, n=COVERAGE_N, tag="cov")


def test_the_nominal_95_percent_interval_covers_at_95_percent(coverage):
    """The number this module is judged on.

    240 independent workloads, a fresh log each time, a nominal 95% interval each time,
    and the truth obtained by actually running the candidate. The band is not chosen:
    it is the Wilson interval on the measured coverage itself, so the test goes red
    exactly when the miscalibration is larger than the Monte-Carlo noise that measured
    it. Over-covering fails too -- an interval that is always right because it is too
    wide is not a 95% interval either.
    """
    assert coverage.n_used >= 200, f"only {coverage.n_used} usable seeds"
    band = coverage.coverage_interval()
    assert band.low <= coverage.nominal <= band.high, (
        f"nominal {coverage.nominal:.0%} intervals covered {coverage.covered_fraction:.1%} "
        f"[{band.low:.1%}, {band.high:.1%}] over {coverage.n_used} seeds -- "
        f"{'under' if coverage.covered_fraction < coverage.nominal else 'over'}-covering. "
        f"{coverage.narrate()}")


def test_the_truth_the_intervals_are_scored_against_is_sharper_than_they_are(coverage):
    """A coverage sweep measures an interval against a target; a noisy target measures
    the target. `truth_n` is the only guard on that, and it was an unpinned default --
    dropping it from 200,000 to 800, which makes the truth exactly as noisy as one
    seed's own sample, moved the headline from 96.2% to 95.8% and the file stayed green.
    The bar is pre-registered at a tenth: the truth's own Wilson width has to be an order
    of magnitude below the widths it is judging, or the sweep is scoring noise against
    noise and the coverage figure means less than it reads.
    """
    iv = wilson_interval(int(round(coverage.truth * coverage.truth_n)), coverage.truth_n)
    truth_width = iv.high - iv.low
    assert truth_width < coverage.mean_width / 10, (
        f"truth from {coverage.truth_n} cases carries a Wilson width of {truth_width:.4f} "
        f"against a mean interval width of {coverage.mean_width:.4f} -- the target is "
        f"{truth_width / coverage.mean_width:.1%} of the thing it is measuring")


def test_the_nominal_confidence_reaches_the_estimator_and_not_only_the_yardstick():
    """The knob has to move the interval it names.

    It did not: `measure_coverage(nominal=...)` built the naive control at `nominal`
    while the interval under test stayed at `quality_snips`' 0.95 default, so mean width
    came back byte-identical at 0.80, 0.95 and 0.99 and every non-default call compared a
    95% interval against a bar it was never asked to meet. The default hid it, because
    0.95 is what the parameter would have set anyway. Small sweep on purpose: this is a
    wiring assertion, and the calibration claim is measured by the fixture above.
    """
    widths = {}
    for nominal in (0.80, 0.95, 0.99):
        cov = measure_coverage(lambda: _explorer(), lambda: _ladder(CANDIDATE_OFFSET),
                               seeds=24, n=400, tag="nom", nominal=nominal,
                               truth_n=20_000)
        widths[nominal] = cov.mean_width
    assert list(widths.values()) == sorted(widths.values()) and len(set(widths.values())) == 3, (
        f"a higher nominal confidence must buy a wider interval: {widths}")


def test_ignoring_the_design_effect_undercovers(coverage):
    """The correction has to earn its place. The same point estimate with a Wilson
    interval at the RAW sample size -- the obvious thing to write -- ignores that
    reweighting threw most of the sample away, and reports a tighter interval than the
    evidence supports. If this passed with the two coverages equal, the effective
    sample size would be decoration."""
    assert coverage.naive_covered_fraction < coverage.covered_fraction, (
        f"Wilson at the raw n covered {coverage.naive_covered_fraction:.1%} vs "
        f"{coverage.covered_fraction:.1%} at the effective n -- the design effect is "
        f"doing no work, so either the weights are uniform or the correction is inert")
    assert coverage.mean_n_effective < 0.9 * coverage.n, (
        f"effective n {coverage.mean_n_effective:.0f} of {coverage.n} is barely a "
        f"discount; with weights this flat the comparison above is not a test")


def test_the_narration_is_read_off_the_trials_rather_than_typed(coverage):
    """This repo has been bitten three times by a print() that stated what its author
    expected instead of what the run produced.

    Pinning three of the eight clauses was not enough to stop the fourth: with only the
    coverage figure and the effective sample size asserted, `implied_width_scale`,
    `mean_width`, `mean_absolute_error` and `sample_covered_fraction` could each be
    replaced by a hard-coded literal and the whole file stayed green -- which is the
    defect named in `narrate`'s own docstring, sitting inside the test written to prevent
    it. Every number the sentence renders is now recomputed from the trials here and
    matched against the string, so a typed constant has nowhere to hide.
    """
    text = coverage.narrate()
    trials = coverage.trials

    # The comparison is on the RAW value, not on the rendered one. Matching only the
    # formatted string is what a constant hides behind: mean |error| replaced by a typed
    # 0.0257 and the same-seed coverage by 0.983 both render identically to the measured
    # 0.025741 and 0.983333, and both survived the first version of this loop. The format
    # is then checked separately, so the number has to be both right and visible.
    recomputed = {
        "coverage": (coverage.covered_fraction,
                     sum(t.covered for t in trials) / len(trials), ".1%"),
        "naive coverage": (coverage.naive_covered_fraction,
                           sum(t.naive_covered for t in trials) / len(trials), ".1%"),
        "same-seed coverage": (coverage.sample_covered_fraction,
                               sum(t.covered_sample for t in trials) / len(trials), ".1%"),
        "mean width": (coverage.mean_width,
                       float(np.mean([t.high - t.low for t in trials])), ".1%"),
        "mean |error|": (coverage.mean_absolute_error,
                         float(np.mean([abs(t.point - t.truth) for t in trials])), ".1%"),
        "effective n": (coverage.mean_n_effective,
                        float(np.mean([t.n_effective for t in trials])), ".0f"),
        "width scale": (coverage.implied_width_scale,
                        float(np.percentile(
                            [abs(t.point - t.truth) / ((t.high - t.low) / 2)
                             for t in trials], 100 * coverage.nominal)), ".2f"),
        "truth": (coverage.truth, trials[0].truth, ".4f"),
    }
    for label, (reported, derived, fmt) in recomputed.items():
        assert reported == pytest.approx(derived, rel=1e-12, abs=1e-12), (
            f"{label}: the report says {reported!r} but the trials say {derived!r}")
        assert format(reported, fmt) in text, (
            f"{label}: {format(reported, fmt)!r} is measured but never reaches the "
            f"narration, so a literal in its place would read the same: {text!r}")

    assert ("MIS-CALIBRATED" in text) is not coverage.calibrated, (
        "the verdict word must flip on the measurement, not on a comment being updated")
    assert f"{coverage.refused} refused" in text and str(coverage.truth_n) in text


# --------------------------------------------------------------------------------
# Doubly robust: the reward model is allowed to be wrong.
# --------------------------------------------------------------------------------

def test_dr_survives_a_biased_reward_model_and_the_direct_method_does_not():
    """The property DR is bought for, demonstrated by breaking the thing it depends on.

    A reward model fit on logged features is always somewhat wrong; the question is
    whether being wrong propagates. Handed a deliberately biased model, the direct
    method inherits the bias in full while DR degrades toward IPS, because the
    correction term is computed against the SAME wrong model and cancels it.
    """
    candidate = _ladder(0.20)
    cases, spec, log, _ = _logged(_explorer(), n=4000)
    truth = 1.0 - run(candidate, cases, spec).violation_rate()

    honest = fit_reward_model(log)
    biased = ConstantReward(0.30)   # every action, every context: flatly wrong

    err_direct_honest = abs(quality_direct(log, candidate, honest).value - truth)
    err_direct_biased = abs(quality_direct(log, candidate, biased).value - truth)
    err_dr_honest = abs(quality_dr(log, candidate, honest, logging_policy=_explorer()).value - truth)
    err_dr_biased = abs(quality_dr(log, candidate, biased, logging_policy=_explorer()).value - truth)

    assert err_direct_biased > 0.2, (
        f"the biased reward model must actually be biased, direct method was off by "
        f"{err_direct_biased:.3f} -- a harmless bias proves nothing")
    assert err_dr_biased < err_direct_biased / 4, (
        f"DR should repair the bias: DR off by {err_dr_biased:.3f} vs direct method "
        f"{err_direct_biased:.3f}")
    assert err_dr_honest < 0.05, (
        f"with a reward model fit on the log DR should land close, off by {err_dr_honest:.3f} "
        f"(direct method alone: {err_direct_honest:.3f})")

    # And the fitted reward model has to be a model of P(success | features, ACTION).
    # Nothing above forces that: `err_direct_honest` was computed and then used only
    # inside an f-string, so a reward model with its action one-hot collapsed to a
    # constant -- a model that literally cannot tell opus-5 from gpt-4o-mini -- passed
    # every assertion here, and DR's error even improved (0.0152 -> 0.0004) because the
    # correction term did all the work. The ordering below is the cheapest statement of
    # action-sensitivity that is also true of the ground truth, so it cannot be satisfied
    # by a model that has merely learned the marginal.
    top_hat = quality_direct(log, AlwaysTop(), honest).value
    cheap_hat = quality_direct(log, AlwaysCheapest(), honest).value
    top_true = 1.0 - run(AlwaysTop(), cases, spec).violation_rate()
    cheap_true = 1.0 - run(AlwaysCheapest(), cases, spec).violation_rate()
    assert top_true - cheap_true > 0.2, (
        f"precondition: the two policies must actually deliver different quality, "
        f"{top_true:.3f} vs {cheap_true:.3f}")
    assert top_hat - cheap_hat > 0.2, (
        f"the reward model rates always-top at {top_hat:.3f} and always-cheapest at "
        f"{cheap_hat:.3f} (truth {top_true:.3f} / {cheap_true:.3f}): it is not reading "
        f"the action, so DR's first term is a constant and 'the reward model's opinion' "
        f"is an opinion about nothing")
    assert abs(err_direct_honest) < 0.05, (
        f"and the direct method with that model has to be roughly right on the candidate "
        f"too, off by {err_direct_honest:.3f}; without this the honest reward model is "
        f"never scored and only the deliberately-broken one is")


# --------------------------------------------------------------------------------
# Error grows with divergence -- and divergence is measured.
# --------------------------------------------------------------------------------

def test_estimator_error_grows_with_measured_policy_divergence():
    """The honest statement of where off-policy evaluation may be believed.

    Divergence is the mean total-variation distance between the logging and candidate
    action distributions, computed from the two policies' own propensities -- not the
    knob that generated it. Error is averaged over seeds, because a single seed's error
    is noise and correlating against noise is how a monotone claim gets manufactured.
    """
    # Offsets stop at 0.30. Past it the ladder sends essentially every case to the top
    # model, so total variation pins at its ceiling and further offsets are duplicate
    # x-values -- points that cannot inform a trend but do dilute a correlation. The
    # strictly-increasing assertion below is what forces that to stay true.
    offsets = (0.0, 0.06, 0.12, 0.20, 0.30)
    seeds = 20
    logger = _explorer()

    probe_spec = WorkloadSpec(mix="balanced", n=600, seed_tag="div-probe")
    probe = build_workload(probe_spec)

    divergence, rmse, mae, widths = [], [], [], []
    for off in offsets:
        candidate = _ladder(off)
        divergence.append(tv_divergence(logger, candidate, probe, probe_spec))
        errs, w = [], []
        for k in range(seeds):
            cases, spec, log, _ = _logged(logger, n=1200, seed_tag=f"div{k}")
            est = quality_snips(log, candidate, logging_policy=logger)
            assert est.identifiable, est.reason
            errs.append(est.value - (1.0 - run(candidate, cases, spec).violation_rate()))
            w.append(est.width)
        errs = np.array(errs)
        # Root-mean-square, because variance is the quantity divergence is predicted to
        # drive -- the design effect is a statement about second moments. Mean absolute
        # error is carried alongside so the choice is visible rather than convenient.
        rmse.append(float(np.sqrt((errs ** 2).mean())))
        mae.append(float(np.abs(errs).mean()))
        widths.append(float(np.mean(w)))

    assert divergence == sorted(divergence) and len(set(divergence)) == len(offsets), (
        f"the divergence axis must be strictly increasing or the correlation below is "
        f"reading duplicate x-values: {[round(d, 3) for d in divergence]}")
    r = float(np.corrcoef(divergence, rmse)[0, 1])
    assert r > 0.5, (
        f"estimator error should grow as the policies diverge; corr={r:+.3f} over "
        f"divergence {[round(d, 3) for d in divergence]} and RMSE "
        f"{[round(e, 4) for e in rmse]} (MAE {[round(e, 4) for e in mae]})")
    assert rmse[-1] > 2 * rmse[0], (
        f"error at divergence {divergence[-1]:.3f} is {rmse[-1]:.4f} against "
        f"{rmse[0]:.4f} at {divergence[0]:.3f} -- less than the doubling that would "
        f"make this an axis worth publishing")
    # The mechanism, not just the symptom: the interval widens because the reweighting
    # is discarding more of the log, and it must widen MONOTONICALLY or the estimator
    # is not tracking its own loss of information.
    assert widths == sorted(widths), (
        f"the reported interval must widen with divergence too: {[round(x, 4) for x in widths]}")


def test_the_divergence_axis_has_a_fixed_origin_and_a_fixed_scale():
    """At divergence 0 the candidate IS the logger and the estimator has nothing to do.

    An origin alone does not pin the axis: every other assertion on `tv_divergence` is
    about monotonicity or correlation, and both are scale-free, so dropping the 1/2 from
    the definition -- reporting L1 distance and calling it total variation, which doubles
    every published divergence figure -- changed nothing anywhere in this file. Two exact
    identities fix the scale without a chosen constant. Disjoint deterministic policies
    are at TV 1 by definition, and an epsilon-greedy logger sits exactly
    epsilon * (1 - 1/|A|) from its own base, which is arithmetic on the two propensity
    formulae rather than a number read off a run.
    """
    logger = _explorer()
    probe_spec = WorkloadSpec(mix="balanced", n=400, seed_tag="div0")
    probe = build_workload(probe_spec)
    assert tv_divergence(logger, logger, probe, probe_spec) == pytest.approx(0.0, abs=1e-12)
    assert tv_divergence(logger, _ladder(), probe, probe_spec) > 0.0, (
        "an exploring logger must diverge from its own deterministic base, or the "
        "exploration is not happening")

    assert tv_divergence(AlwaysTop(), AlwaysCheapest(), probe, probe_spec) == pytest.approx(1.0, abs=1e-12), (
        "two deterministic policies that never agree are one unit of total variation "
        "apart; anything else means this is not a total-variation distance")
    assert tv_divergence(logger, _ladder(), probe, probe_spec) == pytest.approx(
        EPSILON * (1.0 - 1.0 / len(REGISTRY)), abs=1e-12), (
        "epsilon-greedy against its own base is exactly eps*(1-1/|A|) away")
