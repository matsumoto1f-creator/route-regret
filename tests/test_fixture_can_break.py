"""Break the fixture on purpose and assert the bench NOTICES.

A green suite proves nothing until each of these has been watched to fail. Every one of
them corrupts the fixture in a specific way and asserts that some headline number moves
the way it must. If a test here passes on a broken fixture, the bench is measuring the
fixture rather than the router.
"""

import numpy as np
import pytest

from route_regret.bench import references, run
from route_regret.fixture import REGISTRY, WorkloadSpec, build_workload, features
from route_regret.metrics import score
from route_regret.models import Case, ModelCard, succeeds
from route_regret.policies import AlwaysCheapest, AlwaysTop, Oracle, ThresholdLadder
from route_regret.report import DEFAULT_LADDER, tune_to_tau

N = 3000


def _perfect_cheap_registry():
    """A registry where the cheapest model never fails."""
    out = []
    for m in REGISTRY:
        if m.name == "gpt-4o-mini":
            # Just BELOW the frontier model, not above it. Setting it above would make
            # the cheapest model the reference model, which correctly collapses the
            # whole routing problem -- a real behaviour, but not the one under test.
            out.append(m.model_copy(update={"capability": {"default": 0.93}}))
        else:
            out.append(m)
    return out


def _useless_cheap_registry():
    out = []
    for m in REGISTRY:
        if m.name != "opus-5":
            out.append(m.model_copy(update={"capability": {"default": -5.0}}))
        else:
            out.append(m)
    return out


def test_when_the_cheap_model_never_fails_the_oracle_collapses_onto_it():
    """If the cheapest model is perfect, the clairvoyant oracle is just 'always
    cheapest' and the achievable-savings denominator is maximal. A bench that reports
    the same numbers here as on the real fixture is not reading the fixture."""
    spec = WorkloadSpec(mix="balanced", n=N)
    cases = build_workload(spec)
    reg = _perfect_cheap_registry()

    ref_real = references(cases, spec)
    ref_broken = references(cases, spec, registry=reg)
    assert ref_broken.oracle_cost < ref_real.oracle_cost, (
        "a near-perfect cheap model must make the clairvoyant oracle cheaper")
    assert ref_broken.achievable > ref_real.achievable, (
        "and must widen the gap a router could capture")

    oracle_led = run(Oracle(reg), cases, spec, registry=reg)
    def cheap_share(registry):
        led = run(Oracle(registry), cases, spec, registry=registry)
        picks = [r.model for r in led.rows if r.call_kind == "route"]
        return picks.count("gpt-4o-mini") / len(picks)

    # Comparative rather than an absolute threshold: an absolute bar would be a number
    # chosen to pass, which is the failure mode this file exists to guard against. A 3x
    # bar was impossible here (the share is already 34%), which is itself the argument
    # for stating the claim as a ratio and checking the ratio is reachable.
    assert cheap_share(reg) > 2 * cheap_share(REGISTRY), (
        f"making the cheap model near-perfect should move the oracle onto it: "
        f"{cheap_share(reg):.1%} vs {cheap_share(REGISTRY):.1%} on the real registry")


def test_when_only_the_top_model_works_there_are_no_savings_to_capture():
    """The mirror image: if nothing but the frontier model succeeds, the oracle IS
    always-frontier, achievable savings is zero, and FASC is undefined rather than
    a large number. A bench that reports a healthy percentage here is fabricating."""
    spec = WorkloadSpec(mix="balanced", n=N)
    cases = build_workload(spec)
    reg = _useless_cheap_registry()
    ref = references(cases, spec, registry=reg)

    assert ref.achievable == pytest.approx(0.0, abs=1e-9), (
        f"achievable savings should vanish, got ${ref.achievable:.4f}")
    led = run(ThresholdLadder(DEFAULT_LADDER, registry=reg), cases, spec, registry=reg)
    s = score(led, ref)
    assert np.isnan(s.fasc), "FASC must be undefined when its denominator is zero"


def test_full_leakage_makes_a_length_only_router_near_perfect():
    """The token_skew lesson, stated as a measurement.

    At leakage=1 the latent difficulty IS the token count, so a router that reads
    nothing but length matches the clairvoyant oracle and the bench measures nothing.
    At leakage=0 length says nothing and the same router collapses. The bench has to be
    able to demonstrate both, which is why leakage is a declared, swept parameter rather
    than a constant nobody can see.
    """
    oracle = Oracle()
    acc = {}
    for leak in (0.0, 1.0):
        spec = WorkloadSpec(mix="balanced", n=2000, leakage=leak)
        cases = build_workload(spec)
        lengths = np.array([c.input_tokens for c in cases])
        lo, hi = lengths.min(), lengths.max()
        cuts = [(0.30, "gpt-4o-mini"), (0.55, "haiku-4.5"),
                (0.80, "sonnet-5"), (1.01, "opus-5")]

        def choose(case):
            z = (case.input_tokens - lo) / max(1, hi - lo)
            for bound, nm in cuts:
                if z <= bound:
                    return next(m for m in REGISTRY if m.name == nm)
            return REGISTRY[-1]

        acc[leak] = np.mean([choose(c).name == oracle.choose(c, spec).name for c in cases])

    assert acc[1.0] > acc[0.0] + 0.10, (
        f"a length-only router should be much better when length reveals difficulty: "
        f"{acc[1.0]:.1%} at leakage=1 vs {acc[0.0]:.1%} at leakage=0")


def test_the_length_difficulty_coupling_is_real_and_switchable():
    """The coupling is the mechanism behind the anti-correlation finding. If it can be
    turned off and the fixture does not notice, the finding was never about the workload."""
    for leak, lo_bound in ((0.95, 0.85), (0.0, -0.15)):
        spec = WorkloadSpec(mix="balanced", n=2000, leakage=leak)
        cases = build_workload(spec)
        d = np.array([c.difficulty for c in cases])
        toks = np.array([c.input_tokens for c in cases])
        r = np.corrcoef(d, toks)[0, 1]
        if leak > 0.5:
            assert r > lo_bound, f"leakage={leak} should couple length to difficulty, got {r:+.3f}"
        else:
            assert abs(r) < 0.2, f"leakage=0 should decouple them, got {r:+.3f}"


def test_a_case_family_where_the_cheap_model_wins_is_representable():
    """Without this the metric is silently capped at agreement-with-frontier, and a
    verifier that treats the expensive model as truth penalises the cheap model for
    being right."""
    mini = next(m for m in REGISTRY if m.name == "gpt-4o-mini")
    opus = next(m for m in REGISTRY if m.name == "opus-5")
    hard_extraction = Case(case_id="x1", difficulty=0.70, input_tokens=900,
                           output_tokens=300, family="structured_extraction")
    # On this family the gap between the smallest and largest model nearly closes.
    from route_regret.models import per_item_accuracy
    assert per_item_accuracy(mini, hard_extraction) > 0.5
    gap_default = (per_item_accuracy(opus, hard_extraction.model_copy(update={"family": "default"}))
                   - per_item_accuracy(mini, hard_extraction.model_copy(update={"family": "default"})))
    gap_extract = per_item_accuracy(opus, hard_extraction) - per_item_accuracy(mini, hard_extraction)
    assert gap_extract < gap_default, "the extraction family must narrow the capability gap"


def test_the_item_count_gap_is_non_monotone():
    """Per-item accuracy q gives exact-match success q**N. The cheap-vs-frontier gap
    rises then falls in N: at N=1 both usually succeed, at huge N both always fail, and
    the interesting region is in between. No surface feature captures this, which is the
    sharpest argument for labelling on observed adequacy rather than on complexity."""
    from route_regret.models import per_item_accuracy
    mini = next(m for m in REGISTRY if m.name == "gpt-4o-mini")
    opus = next(m for m in REGISTRY if m.name == "opus-5")
    gaps = []
    for n_items in (1, 5, 20, 80, 300, 2000):
        c = Case(case_id="n", difficulty=0.45, input_tokens=500, output_tokens=200,
                 family="exact_match_list", n_items=n_items)
        gaps.append(per_item_accuracy(opus, c) ** n_items - per_item_accuracy(mini, c) ** n_items)
    peak = gaps.index(max(gaps))
    assert 0 < peak < len(gaps) - 1, f"gap should peak in the interior, peaked at {peak}: {gaps}"
    assert gaps[-1] < max(gaps) / 2, "the gap must close again at large N"
