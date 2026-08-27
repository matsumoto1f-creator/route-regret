"""What the two metrics do when only the TRAFFIC changes.

No scalar summary of a router is invariant to the traffic mix, and claiming otherwise
would be the same error one level up. What is claimed here is weaker and true: under a
pinned quality constraint, FASC@delta is LESS a property of the traffic than the
savings figure it replaces, and the ordering of policies is stable across every mix.
"""

import pytest

from route_regret.fixture import MIXES
from route_regret.report import bench_mix, spread

N = 3000


@pytest.fixture(scope="module")
def table():
    return {mix: {s.policy: s for s in bench_mix(mix, n=N)[1]} for mix in MIXES}


def test_the_policy_ordering_is_stable_on_every_mix(table):
    """If the bench reordered the policies when the traffic changed, it would be
    measuring the traffic."""
    for mix, scores in table.items():
        oracle = scores["oracle"].fasc_at_delta
        router = scores["threshold_ladder"].fasc_at_delta
        blind = scores["content_blind"].fasc_at_delta
        top = scores["always_top"].fasc_at_delta
        assert oracle is not None and top is not None
        assert oracle > top, f"{mix}: the oracle must beat always-frontier"
        if router is not None and blind is not None:
            assert oracle >= router >= blind >= top - 1e-9, (
                f"{mix}: ordering broke — oracle {oracle:.1%} router {router:.1%} "
                f"blind {blind:.1%} top {top:.1%}")


def test_the_constrained_metric_is_less_a_property_of_the_traffic(table):
    """The headline table. Both spreads are printed in the README; the claim is only
    that one is smaller, never that either is zero."""
    naive = [table[m]["threshold_ladder"].naive_savings for m in MIXES]
    fasc = [table[m]["threshold_ladder"].fasc_at_delta for m in MIXES
            if table[m]["threshold_ladder"].fasc_at_delta is not None]
    assert len(fasc) == len(MIXES), "the tuned router should be admissible on every mix"
    assert spread(fasc) < spread(naive), (
        f"FASC@delta spread {spread(fasc):.1%} should be tighter than naive savings "
        f"spread {spread(naive):.1%}")


def test_the_degenerate_router_is_refused_on_every_mix(table):
    for mix, scores in table.items():
        assert scores["always_cheapest"].fasc_at_delta is None, (
            f"{mix}: always-cheapest must never be admissible")
        assert scores["always_cheapest"].naive_savings > scores["threshold_ladder"].naive_savings, (
            f"{mix}: and must still win the retired metric, or the control is toothless")
