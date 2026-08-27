"""FASC@delta, and why every simpler metric was rejected.

The obvious metric -- percent saved against always-frontier -- is ANTI-CORRELATED with
the router being right. The mechanism is the workload's own coupling: harder prompts are
longer, longer prompts cost more, so a router that routes badly sends the expensive long
prompts to the cheap model and books a bigger saving. Measured on this fixture,
corrupting a router from full signal to none RAISES its reported savings.

Normalising against a clairvoyant oracle does NOT fix this -- it has the same
correlation, because it rescales a numerator whose sign is already wrong. What fixes it
is refusing to score any policy that spent quality to get there. The constraint does the
work; the normalisation only makes the number comparable across mixes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from route_regret.ledger import Ledger


@dataclass(frozen=True)
class Reference:
    """The two lines every policy is placed between, recomputed per workload."""
    top_cost: float
    top_violation: float
    oracle_cost: float

    @property
    def achievable(self) -> float:
        return self.top_cost - self.oracle_cost


@dataclass(frozen=True)
class Score:
    policy: str
    cost: float
    violation: float
    admissible: bool
    tau: float
    fasc: float                 # unconstrained; reported for transparency, never alone
    fasc_at_delta: float | None  # None when the policy bought its savings with quality
    naive_savings: float         # the retired metric, kept as a visible control

    def line(self) -> str:
        fd = "not admissible" if self.fasc_at_delta is None else f"{self.fasc_at_delta:7.1%}"
        return (f"{self.policy:<20}{self.cost:>12.4f}{self.violation:>11.1%}"
                f"{self.naive_savings:>12.1%}{fd:>17}")


def score(ledger: Ledger, ref: Reference, delta: float = 0.03) -> Score:
    cost = ledger.total_cost()
    violation = ledger.violation_rate()
    tau = ref.top_violation + delta
    admissible = violation <= tau + 1e-12
    fasc = (ref.top_cost - cost) / ref.achievable if ref.achievable > 0 else float("nan")
    return Score(
        policy=ledger.rows[0].policy_version if ledger.rows else "",
        cost=cost, violation=violation, admissible=admissible, tau=tau,
        fasc=fasc, fasc_at_delta=fasc if admissible else None,
        naive_savings=(ref.top_cost - cost) / ref.top_cost if ref.top_cost > 0 else 0.0,
    )


def self_agreement_ceiling(accuracy: float, n_labels: int = 2) -> float:
    """Two independent samples of the SAME model agree at p^2 + (1-p)^2/(L-1).

    The instrument's own ceiling. At p=0.95 with binary labels that is 90.5%, so a
    pipeline reporting "9.5% routing failures" for a model compared against itself is
    reporting its own noise floor. Every parity figure must be read as a distance from
    this line, never from 100%.
    """
    if n_labels < 2:
        raise ValueError("need at least two labels to disagree")
    return accuracy ** 2 + (1 - accuracy) ** 2 / (n_labels - 1)


def paired_bootstrap(a: list[float], b: list[float], *, draws: int = 2000,
                     tag: str = "bootstrap") -> tuple[float, float, float]:
    """Paired difference a-b with a 95% interval, deterministic across runs.

    Paired because both policies see the SAME cases; an unpaired interval here would be
    wider than the evidence warrants and would hide real margins.
    """
    if len(a) != len(b):
        raise ValueError("paired bootstrap needs equal-length samples")
    n = len(a)
    diffs = [x - y for x, y in zip(a, b)]
    point = sum(diffs) / n
    means = []
    for k in range(draws):
        total = 0.0
        for i in range(n):
            # Deterministic resample index, so the published interval never moves.
            h = int(math.fmod(abs(math.sin((k + 1) * 12.9898 + (i + 1) * 78.233)) * 43758.5453, 1.0) * n)
            total += diffs[h]
        means.append(total / n)
    means.sort()
    return point, means[int(0.025 * draws)], means[int(0.975 * draws)]


@dataclass(frozen=True)
class Stratum:
    decile: int
    n: int
    achievable: float
    fasc: float | None       # None when the denominator is too small to divide by

    def line(self) -> str:
        v = "not_estimable" if self.fasc is None else f"{self.fasc:7.1%}"
        return f"  d in [{self.decile/10:.1f},{(self.decile+1)/10:.1f})  n={self.n:>6}  " \
               f"achievable=${self.achievable:8.4f}  FASC={v}"


# Below this share of total achievable savings a decile's denominator is too small for
# the ratio to mean anything. Reporting a percentage there produces enormous numbers
# that look like findings.
ESTIMABLE_FLOOR = 0.005


def by_stratum(policy_rows, top_rows, oracle_rows, difficulty_of) -> list[Stratum]:
    """FASC per difficulty decile.

    This is the part that IS invariant to the traffic mix: with the policy frozen and
    the fixture deterministic, any quantity conditioned on latent difficulty does not
    move when the mix only reweights the strata. A mix-level scalar cannot be made
    invariant; a per-stratum one is invariant by construction. Strata whose achievable
    savings are negligible are reported as not_estimable rather than as a number.
    """
    buckets: dict[int, dict[str, float]] = {}
    for rows, key in ((policy_rows, "policy"), (top_rows, "top"), (oracle_rows, "oracle")):
        for r in rows:
            if r.call_kind == "judge":
                continue
            d = difficulty_of(r.request_id)
            b = buckets.setdefault(min(9, int(d * 10)), {"policy": 0.0, "top": 0.0,
                                                          "oracle": 0.0, "n": 0})
            b[key] += r.cost_usd
            if key == "policy" and r.call_kind == "route":
                b["n"] += 1

    total_achievable = sum(b["top"] - b["oracle"] for b in buckets.values())
    out = []
    for decile in sorted(buckets):
        b = buckets[decile]
        achievable = b["top"] - b["oracle"]
        share = achievable / total_achievable if total_achievable > 0 else 0.0
        fasc = ((b["top"] - b["policy"]) / achievable
                if achievable > 0 and share >= ESTIMABLE_FLOOR else None)
        out.append(Stratum(decile=decile, n=int(b["n"]), achievable=achievable, fasc=fasc))
    return out
