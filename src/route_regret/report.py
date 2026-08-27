"""The headline table, and the control that shows why the old headline was retired."""

from __future__ import annotations

from route_regret.bench import marginal_of, references, run
from route_regret.fixture import MIXES, REGISTRY, WorkloadSpec, build_workload
from route_regret.metrics import Reference, Score, score
from route_regret.policies import (AlwaysCheapest, AlwaysTop, ContentBlind, Oracle,
                                   ThresholdLadder)

DEFAULT_LADDER = [(0.30, "gpt-4o-mini"), (0.55, "haiku-4.5"), (0.80, "sonnet-5"),
                  (1.01, "opus-5")]


def standard_policies(blind_marginal: dict[str, float] | None = None):
    pol = [Oracle(), AlwaysTop(), AlwaysCheapest(), ThresholdLadder(DEFAULT_LADDER)]
    if blind_marginal:
        pol.append(ContentBlind(blind_marginal))
    return pol


def tune_to_tau(make_policy, cases, spec, ref, delta: float, verify_rate: float,
                lo: float = -0.6, hi: float = 0.6, steps: int = 18):
    """Find the operating point at which a policy just meets the quality constraint.

    Comparing policies at whatever operating point their author happened to pick is not
    a comparison -- a policy can always look cheaper by spending more quality. Bisection
    on the knob, keeping the cheapest admissible point.

    Returns (policy, ledger) for the tuned policy, or the most conservative point tried
    if the constraint is unreachable, so an impossible policy reports as inadmissible
    rather than silently returning its best-looking run.
    """
    tau = ref.top_violation + delta
    best = None
    for _ in range(steps):
        mid = (lo + hi) / 2
        policy = make_policy(mid)
        led = run(policy, cases, spec, verify_rate=verify_rate)
        if led.violation_rate() <= tau:
            best = (policy, led)      # admissible: try routing more aggressively
            hi = mid
        else:
            lo = mid                  # too many failures: shift toward the top model
    if best is None:
        policy = make_policy(hi)
        best = (policy, run(policy, cases, spec, verify_rate=verify_rate))
    return best


def bench_mix(mix: str, *, n: int = 8000, verify_rate: float = 0.05,
              delta: float = 0.03, spec: WorkloadSpec | None = None
              ) -> tuple[Reference, list[Score]]:
    spec = spec or WorkloadSpec(mix=mix, n=n)
    cases = build_workload(spec)
    ref = references(cases, spec)

    # The reference lines are not entrants: the oracle is clairvoyant, so verifying it
    # would charge a denominator for a measurement it does not need.
    scores = [score(run(Oracle(), cases, spec, verify_rate=0.0), ref, delta=delta),
              score(run(AlwaysTop(), cases, spec, verify_rate=0.0), ref, delta=delta),
              score(run(AlwaysCheapest(), cases, spec, verify_rate=verify_rate), ref, delta=delta)]

    ladder, ladder_led = tune_to_tau(
        lambda off: ThresholdLadder(DEFAULT_LADDER, offset=off),
        cases, spec, ref, delta, verify_rate)
    scores.append(score(ladder_led, ref, delta=delta))

    # The blind control matches the tuned router's OWN tier marginal, then is tuned
    # itself -- so any margin is attributable to reading the request, not to tier mix.
    marginal = marginal_of(ladder_led)
    blind, blind_led = tune_to_tau(
        lambda off: ContentBlind(_shift_marginal(marginal, off)),
        cases, spec, ref, delta, verify_rate, lo=-1.0, hi=1.0)
    scores.append(score(blind_led, ref, delta=delta))
    return ref, scores


def _shift_marginal(marginal: dict[str, float], offset: float) -> dict[str, float]:
    """Blend the router's own tier marginal toward all-frontier.

    `offset` maps to a blend weight t in [0,1]: at t=0 the control uses exactly the
    router's model mix but assigns it at random; at t=1 it IS always-frontier.

    Spanning all the way to always-frontier is not cosmetic. The control has to be able
    to REACH the quality constraint, or "the router is admissible and the control is
    not" is true by construction and the headline comparison is a test that cannot fail.
    Anchoring one end at always-frontier guarantees the constraint is reachable, so the
    margin the router reports is a margin it had to earn.
    """
    from route_regret.fixture import REGISTRY as R

    top = max(R, key=lambda m: m.capability["default"]).name
    # Higher offset = more conservative, matching the ladder's knob direction so one
    # bisection routine tunes both. offset in [-0.6, 0.6] -> t in [0, 1].
    t = min(1.0, max(0.0, (offset + 0.6) / 1.2))
    blended = {m.name: (1.0 - t) * marginal.get(m.name, 0.0) for m in R}
    blended[top] = blended.get(top, 0.0) + t
    total = sum(blended.values())
    return {k: v / total for k, v in blended.items() if v > 0}


def spread(values) -> float:
    """Max minus min, ignoring inadmissible entries."""
    vals = [v for v in values if v is not None]
    return (max(vals) - min(vals)) if vals else 0.0


def mix_invariance_table(*, n: int = 8000, verify_rate: float = 0.05, delta: float = 0.03):
    """The repo's first table: what each metric does when only the traffic changes.

    Neither metric is mix-invariant. The savings figure is printed beside FASC@delta as
    the control that shows what the retired headline would have reported.
    """
    return {mix: {s.policy: s for s in bench_mix(mix, n=n, verify_rate=verify_rate,
                                                 delta=delta)[1]}
            for mix in MIXES}
