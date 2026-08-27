"""Run a policy over a workload and price everything it did.

Escalation is deliberately absent. The spec this replaces verified a routed answer
against the reference model and then, on disagreement, RE-RAN the reference model. The
verify call already produced that answer: re-running pays twice for a result you are
holding. Deleting it collapses the break-even verification rate to a clean identity,
`v* = 1 - c_cheap/c_reference`, which is arithmetic rather than a benchmark.
"""

from __future__ import annotations

from dataclasses import dataclass

from route_regret.fixture import REGISTRY, WorkloadSpec
from route_regret.ledger import Ledger, Row
from route_regret.models import Case, ModelCard, _unit_draw, succeeds
from route_regret.policies import Policy

# Judge overhead: it reads the prompt and both answers, and emits a short verdict.
JUDGE_PROMPT_OVERHEAD = 200
JUDGE_OUTPUT_TOKENS = 50


def reference_model(registry: list[ModelCard] | None = None) -> ModelCard:
    """The model quality is measured AGAINST. By capability, never by price — the
    registry is not price-ordered, and taking the most expensive model here would
    silently pick a weaker reference."""
    return max(registry or REGISTRY, key=lambda m: m.capability["default"])


def verifies(case: Case, spec: WorkloadSpec, rate: float) -> bool:
    """Which requests get verified. Deterministic, so two runs agree to the cent."""
    return _unit_draw(case.case_id, "verify", spec.seed_tag) < rate


def run(policy: Policy, cases: list[Case], spec: WorkloadSpec, *,
        verify_rate: float = 0.0, registry: list[ModelCard] | None = None) -> Ledger:
    registry = registry or REGISTRY
    ref = reference_model(registry)
    ledger = Ledger()

    for case in cases:
        chosen = policy.choose(case, spec)
        routed_ok = succeeds(chosen, case)
        ledger.add(Row(
            request_id=case.case_id, call_kind="route", model=chosen.name,
            input_tokens=case.input_tokens, output_tokens=case.output_tokens,
            cost_usd=chosen.cost(case.input_tokens, case.output_tokens),
            ok=routed_ok, policy_version=policy.name,
            propensity=policy.propensity(case, spec, chosen),
        ))

        # Verifying a request the reference model itself served buys nothing: it would
        # compare a model against itself and agree by construction. That self-comparison
        # is how the original spec reported quality it had not measured.
        if chosen.name == ref.name or not verifies(case, spec, verify_rate):
            continue

        ledger.add(Row(
            request_id=case.case_id, call_kind="verify", model=ref.name,
            input_tokens=case.input_tokens, output_tokens=case.output_tokens,
            cost_usd=ref.cost(case.input_tokens, case.output_tokens),
            ok=succeeds(ref, case), policy_version=policy.name,
        ))
        judge_in = case.input_tokens + 2 * case.output_tokens + JUDGE_PROMPT_OVERHEAD
        ledger.add(Row(
            request_id=case.case_id, call_kind="judge", model=ref.name,
            input_tokens=judge_in, output_tokens=JUDGE_OUTPUT_TOKENS,
            cost_usd=ref.cost(judge_in, JUDGE_OUTPUT_TOKENS),
            ok=True, policy_version=policy.name,
        ))
    return ledger


def break_even_verify_rate(cheap: ModelCard, ref: ModelCard,
                           input_tokens: int = 800, output_tokens: int = 400,
                           with_judge: bool = True) -> float:
    """The verification rate above which routing stops saving money.

    Arithmetic, not a benchmark, and it does not depend on the fixture. With the judge
    counted — and an LLM-as-judge call is itself an expensive LLM call, which the
    original spec never priced — the rate is materially lower than the naive figure.
    """
    c_l = cheap.cost(input_tokens, output_tokens)
    c_h = ref.cost(input_tokens, output_tokens)
    if c_h <= 0:
        return 0.0
    per_verify = c_h
    if with_judge:
        judge_in = input_tokens + 2 * output_tokens + JUDGE_PROMPT_OVERHEAD
        per_verify += ref.cost(judge_in, JUDGE_OUTPUT_TOKENS)
    return max(0.0, (c_h - c_l) / per_verify)


def references(cases: list[Case], spec: WorkloadSpec,
               registry: list[ModelCard] | None = None):
    """The two reference lines, recomputed for THIS workload.

    Recomputed rather than cached because both move with the mix. A denominator carried
    over from another traffic profile is how a savings figure becomes a statement about
    the fixture instead of about the router.
    """
    from route_regret.metrics import Reference
    from route_regret.policies import AlwaysTop, Oracle

    registry = registry or REGISTRY
    top = run(AlwaysTop(registry), cases, spec, verify_rate=0.0, registry=registry)
    orc = run(Oracle(registry), cases, spec, verify_rate=0.0, registry=registry)
    return Reference(top_cost=top.total_cost(), top_violation=top.violation_rate(),
                     oracle_cost=orc.total_cost())


def marginal_of(ledger: Ledger) -> dict[str, float]:
    """The model mix a policy actually used — the thing a blind control must match."""
    routed = [r for r in ledger.rows if r.call_kind == "route"]
    counts: dict[str, int] = {}
    for r in routed:
        counts[r.model] = counts.get(r.model, 0) + 1
    n = len(routed)
    return {k: v / n for k, v in counts.items()}
