"""Every call the system makes, priced separately from the tokens it used.

The spec this repo replaces logged one `cost` column per request. That schema cannot
represent the system's own spend: the verification call, the judge call and the routed
call are three different amounts of money against one request, and a single column
silently reports only the first. A savings figure computed from that schema cannot be
wrong, because there is nothing in it to be wrong about.

Tokens are stored beside cost so any row can be re-priced. A vendor price change alone
moves a headline by several points, and a scalar cost column cannot separate that from
the routing having got better.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# route    the answer the user actually waited for
# verify   the shadow call to the reference model, on sampled traffic
# judge    the comparison between them
# escalate retained ONLY so the schema can represent a system that re-runs. This repo
#          never emits it: the verify call already produced the reference answer, so
#          re-running is paying twice for a result you are holding.
CALL_KINDS = ("route", "verify", "judge", "escalate")


@dataclass(frozen=True)
class Row:
    request_id: str
    call_kind: str
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    ok: bool
    # Stamped so a number can be traced to the configuration that produced it.
    price_table_version: str = "aug-2026"
    policy_version: str = ""
    # P(this policy chose this model for this case). Needed for off-policy estimation;
    # without it a logged run can price a counterfactual policy but cannot estimate its
    # quality.
    propensity: float = 1.0

    def __post_init__(self) -> None:
        if self.call_kind not in CALL_KINDS:
            raise ValueError(f"unknown call_kind {self.call_kind!r}")


@dataclass
class Ledger:
    rows: list[Row] = field(default_factory=list)

    def add(self, row: Row) -> None:
        self.rows.append(row)

    def total_cost(self) -> float:
        """Every row. This is the only headline denominator that is not a fiction —
        a system's cost includes what it spent measuring itself."""
        return sum(r.cost_usd for r in self.rows)

    def cost_by_kind(self) -> dict[str, float]:
        out = {k: 0.0 for k in CALL_KINDS}
        for r in self.rows:
            out[r.call_kind] += r.cost_usd
        return out

    def delivered(self) -> list[Row]:
        """One row per request: what the user actually received.

        A verified request whose routed answer failed is delivered from the reference
        model's answer, which the verify call already produced.
        """
        best: dict[str, Row] = {}
        for r in self.rows:
            if r.call_kind == "judge":
                continue
            prior = best.get(r.request_id)
            if prior is None or (not prior.ok and r.ok):
                best[r.request_id] = r
        return list(best.values())

    def violation_rate(self) -> float:
        d = self.delivered()
        return 1.0 - (sum(1 for r in d if r.ok) / len(d)) if d else 0.0

    def n_requests(self) -> int:
        return len({r.request_id for r in self.rows})
