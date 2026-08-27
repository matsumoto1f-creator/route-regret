"""How mix-dependent each way of reporting is — measured, and ordered.

The claim in the README is a ladder, not an invariance. Nothing here is invariant to
the traffic mix. Four ways of reporting the same router are ordered by how much they
move when only the traffic changes, and the README prints all four so a reader can see
the size of the effect rather than being told it was handled.
"""

import pytest

from route_regret.bench import references, run
from route_regret.fixture import MIXES, WorkloadSpec, build_workload
from route_regret.metrics import by_stratum
from route_regret.policies import AlwaysTop, Oracle, ThresholdLadder
from route_regret.report import DEFAULT_LADDER, bench_mix, spread, tune_to_tau

N = 4000


def _stratum_spread(frozen: bool) -> float:
    per_mix = {}
    for mix in MIXES:
        spec = WorkloadSpec(mix=mix, n=N)
        cases = build_workload(spec)
        d_of = {c.case_id: c.difficulty for c in cases}
        ref = references(cases, spec)
        if frozen:
            led = run(ThresholdLadder(DEFAULT_LADDER), cases, spec, verify_rate=0.05)
        else:
            _, led = tune_to_tau(lambda o: ThresholdLadder(DEFAULT_LADDER, offset=o),
                                 cases, spec, ref, 0.03, 0.05)
        strata = by_stratum(led.rows, run(AlwaysTop(), cases, spec).rows,
                            run(Oracle(), cases, spec).rows, lambda r: d_of[r])
        per_mix[mix] = {s.decile: s.fasc for s in strata if s.fasc is not None}

    spreads = []
    for dec in range(10):
        vals = [per_mix[m].get(dec) for m in MIXES if per_mix[m].get(dec) is not None]
        if len(vals) > 1:
            spreads.append(max(vals) - min(vals))
    return sum(spreads) / len(spreads)


@pytest.fixture(scope="module")
def ladder():
    table = {m: {s.policy: s for s in bench_mix(m, n=N)[1]} for m in MIXES}
    return {
        "naive_scalar": spread([table[m]["threshold_ladder"].naive_savings for m in MIXES]),
        "fasc_scalar": spread([table[m]["threshold_ladder"].fasc_at_delta for m in MIXES]),
        "stratum_tuned": _stratum_spread(frozen=False),
        "stratum_frozen": _stratum_spread(frozen=True),
    }


def test_the_reporting_ladder_is_ordered_as_the_readme_claims(ladder):
    assert ladder["fasc_scalar"] < ladder["naive_scalar"], (
        f"FASC@delta should be less traffic-dependent than naive savings: "
        f"{ladder['fasc_scalar']:.1%} vs {ladder['naive_scalar']:.1%}")
    assert ladder["stratum_frozen"] < ladder["fasc_scalar"], (
        f"per-stratum with a frozen policy should be the tightest view: "
        f"{ladder['stratum_frozen']:.1%} vs {ladder['fasc_scalar']:.1%}")


def test_nothing_here_is_actually_mix_invariant(ladder):
    """The claim the README must NOT make. If every spread collapsed to zero, the
    fixture's mixes would no longer differ and the whole table would be decoration."""
    assert all(v > 0.01 for v in ladder.values()), (
        f"a spread of zero means the mixes stopped differing: {ladder}")


def test_an_untuned_policy_can_exceed_100_percent_in_a_stratum():
    """Not a bug — the signature of inadmissibility. A policy that routes cheap on hard
    cases 'captures' more than the clairvoyant oracle by delivering worse answers. It is
    the anti-correlation showing up per-stratum, and the reason a stratum table without
    a quality column is as misleading as the scalar it replaced."""
    spec = WorkloadSpec(mix="balanced", n=N)
    cases = build_workload(spec)
    d_of = {c.case_id: c.difficulty for c in cases}
    led = run(ThresholdLadder(DEFAULT_LADDER), cases, spec, verify_rate=0.05)
    strata = by_stratum(led.rows, run(AlwaysTop(), cases, spec).rows,
                        run(Oracle(), cases, spec).rows, lambda r: d_of[r])
    over = [s for s in strata if s.fasc is not None and s.fasc > 1.0]
    assert over, "expected the hard strata to exceed 100% for an untuned policy"
    assert all(s.decile >= 5 for s in over), (
        f"and only in the HARD strata, got deciles {[s.decile for s in over]}")
