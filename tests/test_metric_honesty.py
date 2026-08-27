"""The tests that decide whether this repo is worth anything.

The project it replaces had a headline metric that was ANTI-correlated with the router
being right. No test in that spec could have caught it, because every test measured the
number the design was built to produce. These measure the metric itself.
"""

import numpy as np
import pytest

from route_regret.bench import references, run
from route_regret.fixture import REGISTRY, WorkloadSpec, build_workload
from route_regret.metrics import score, self_agreement_ceiling
from route_regret.models import ModelCard, succeeds
from route_regret.policies import AlwaysCheapest, AlwaysTop, Oracle, ThresholdLadder
from route_regret.report import DEFAULT_LADDER, bench_mix, tune_to_tau

N = 4000


def _signal_sweep(signals=(1.0, 0.8, 0.6, 0.4, 0.2, 0.0), n=N, tuned=True):
    """Degrade the router's information and watch what each metric does.

    `tuned` is the whole experiment. Untuned is the spec's regime: a fixed operating
    point, report the saving. Tuned holds every policy at the same delivered quality
    and compares only cost.
    """
    naive, fasc_d, fasc_raw, accuracy = [], [], [], []
    oracle = Oracle()
    for sig in signals:
        spec = WorkloadSpec(mix="balanced", n=n, signal=sig)
        cases = build_workload(spec)
        ref = references(cases, spec)
        if tuned:
            policy, led = tune_to_tau(
                lambda off: ThresholdLadder(DEFAULT_LADDER, offset=off),
                cases, spec, ref, 0.03, 0.05)
        else:
            policy = ThresholdLadder(DEFAULT_LADDER)
            led = run(policy, cases, spec, verify_rate=0.05)
        s = score(led, ref, delta=0.03)
        accuracy.append(np.mean([policy.choose(c, spec).name == oracle.choose(c, spec).name
                                 for c in cases]))
        naive.append(s.naive_savings)
        fasc_raw.append(s.fasc)
        fasc_d.append(s.fasc_at_delta if s.fasc_at_delta is not None else 0.0)
    return (np.array(accuracy), np.array(naive), np.array(fasc_d), np.array(fasc_raw))


def test_at_a_fixed_operating_point_every_cost_metric_rewards_the_worse_router():
    """The defect this repo exists because of, and it is worse than a bad metric.

    Harder prompts are longer; longer prompts cost more. A router that routes BADLY
    dumps the expensive long prompts on the cheap model and books a LARGER saving. The
    spec reports cost at a FIXED operating point, so its money shot pays you to route
    badly -- and normalising against a clairvoyant oracle does not help, because it
    rescales a numerator whose sign is already wrong.
    """
    accuracy, naive, _, fasc_raw = _signal_sweep(tuned=False)
    r_naive = np.corrcoef(accuracy, naive)[0, 1]
    r_fasc = np.corrcoef(accuracy, fasc_raw)[0, 1]
    assert r_naive < -0.5, f"naive savings should reward a worse router; corr={r_naive:+.3f}"
    assert r_fasc < -0.5, (
        f"oracle-normalised savings should be anti-correlated TOO (corr={r_fasc:+.3f}). "
        "If it is not, the claim that the constraint rather than the normalisation does "
        "the work is wrong, and the README must be corrected.")


def test_pinning_the_operating_point_is_what_actually_fixes_it():
    """The finding that matters, and it is not 'use a better metric'.

    Hold every policy at the same delivered quality and compare only cost, and the
    perverse incentive disappears -- a worse router can no longer buy a cheaper bill,
    because it has to spend its way back to the quality bar. Even the naive savings
    figure behaves under this protocol. The metric was never the fix; the PROTOCOL is.
    """
    accuracy, naive, fasc_d, _ = _signal_sweep(tuned=True)
    r_naive = np.corrcoef(accuracy, naive)[0, 1]
    r_fasc = np.corrcoef(accuracy, fasc_d)[0, 1]
    assert r_fasc > 0.8, f"FASC@delta should track router quality; corr={r_fasc:+.3f}"
    assert r_naive > 0.5, (
        f"under a pinned operating point even naive savings should track the router "
        f"(corr={r_naive:+.3f}) -- that is the point of the protocol")


def test_oracle_normalisation_alone_does_not_fix_it():
    """The correction that matters, and the one easy to get wrong.

    Normalising against a clairvoyant oracle makes the number comparable across traffic
    mixes -- but it rescales a numerator whose SIGN is already wrong, so unconstrained
    FASC is anti-correlated exactly like the naive metric. The quality constraint does
    all of the work. This test exists because the author proposed the normalisation as
    the fix and was wrong.
    """
    signals = (1.0, 0.7, 0.4, 0.1)
    acc, fasc_unconstrained = [], []
    for sig in signals:
        spec = WorkloadSpec(mix="balanced", n=N, signal=sig)
        cases = build_workload(spec)
        ref = references(cases, spec)
        # UNTUNED: the same operating point regardless of signal, so only the router's
        # information changes. Tuning to tau is what makes the metric behave.
        policy = ThresholdLadder(DEFAULT_LADDER)
        led = run(policy, cases, spec, verify_rate=0.05)
        oracle = Oracle()
        acc.append(np.mean([policy.choose(c, spec).name == oracle.choose(c, spec).name
                            for c in cases]))
        fasc_unconstrained.append(score(led, ref).fasc)
    r = np.corrcoef(acc, fasc_unconstrained)[0, 1]
    assert r < -0.5, (
        f"unconstrained FASC should be anti-correlated like the naive metric "
        f"(corr={r:+.3f}). If it is not, the claim that the CONSTRAINT rather than the "
        "normalisation does the work is wrong and the README must be corrected."
    )


def test_the_degenerate_router_wins_the_old_metric_and_loses_this_one():
    """always-cheapest is the policy the retired metric rewards most."""
    ref, scores = bench_mix("balanced", n=N)
    by = {s.policy: s for s in scores}
    cheapest, ladder = by["always_cheapest"], by["threshold_ladder"]

    assert cheapest.naive_savings > ladder.naive_savings, (
        "the degenerate policy should still win the OLD metric -- that is the point")
    assert cheapest.fasc_at_delta is None, "and it must be refused by the new one"
    assert cheapest.violation > ladder.violation


def test_the_router_earns_its_margin_over_a_blind_control_at_matched_quality():
    """The headline claim. Both policies are tuned to the same quality constraint and
    the blind one is given the router's own model marginal, so the margin is
    attributable to reading the request rather than to spending more."""
    ref, scores = bench_mix("balanced", n=N)
    by = {s.policy: s for s in scores}
    router, blind = by["threshold_ladder"], by["content_blind"]

    assert router.fasc_at_delta is not None, "the router must be admissible"
    assert blind.fasc_at_delta is not None, (
        "the blind control must also be able to reach the constraint, or 'the router is "
        "admissible and the control is not' is true by construction")
    margin = router.fasc_at_delta - blind.fasc_at_delta
    assert margin > 0.05, f"router earned only {margin:+.1%} over a blind control"


def test_reference_lines_sit_where_they_must():
    ref, scores = bench_mix("balanced", n=N)
    by = {s.policy: s for s in scores}
    assert by["oracle"].fasc_at_delta == pytest.approx(1.0, abs=1e-9)
    assert by["always_top"].fasc_at_delta == pytest.approx(0.0, abs=1e-9)
    assert by["oracle"].cost < by["always_top"].cost


def test_the_self_agreement_ceiling_is_published_not_assumed():
    """Two samples of the SAME model at accuracy p agree at p^2+(1-p)^2/(L-1). A
    pipeline that reports 100% parity as achievable is reporting its own noise floor
    as a routing failure."""
    assert self_agreement_ceiling(0.95) == pytest.approx(0.905, abs=1e-9)
    assert self_agreement_ceiling(1.00) == pytest.approx(1.0)
    # Monotone in accuracy: a better model has a higher ceiling.
    vals = [self_agreement_ceiling(p) for p in (0.85, 0.90, 0.95, 0.99)]
    assert vals == sorted(vals)
    with pytest.raises(ValueError):
        self_agreement_ceiling(0.95, n_labels=1)
