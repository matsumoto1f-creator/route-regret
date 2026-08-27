"""The workload, and the registry of models that serve it.

The registry's prices are Aug-2026 list prices. Its capabilities are hand-authored and
deliberately NOT in price order: sonnet-5 is cheaper per token than gpt-4o and more
capable, and the `structured_extraction` family is one the smallest model handles about
as well as the largest. Both are true of real model lineups, and both are things a
router that only knows the price ladder gets wrong.
"""

from __future__ import annotations

from dataclasses import dataclass

from route_regret.models import Case, Mix, ModelCard, _unit_draw

REGISTRY: list[ModelCard] = [
    ModelCard(name="gpt-4o-mini", price_in_per_mtok=0.15, price_out_per_mtok=0.60,
              capability={"default": 0.30, "structured_extraction": 0.74}),
    ModelCard(name="haiku-4.5", price_in_per_mtok=1.00, price_out_per_mtok=5.00,
              capability={"default": 0.55, "structured_extraction": 0.80}),
    # Cheaper than gpt-4o AND more capable. A price-ordered ladder routes past it.
    ModelCard(name="sonnet-5", price_in_per_mtok=2.00, price_out_per_mtok=10.00,
              capability={"default": 0.82, "structured_extraction": 0.88}),
    ModelCard(name="gpt-4o", price_in_per_mtok=2.50, price_out_per_mtok=10.00,
              capability={"default": 0.74, "structured_extraction": 0.86}),
    ModelCard(name="opus-5", price_in_per_mtok=5.00, price_out_per_mtok=25.00,
              capability={"default": 0.94, "structured_extraction": 0.95}),
]

FAMILIES = ("default", "structured_extraction", "exact_match_list")

MIXES: dict[str, Mix] = {
    "balanced":    Mix(name="balanced",    decile_weights=[1] * 10),
    "mostly_easy": Mix(name="mostly_easy", decile_weights=[5, 4, 3, 2, 1, 1, 1, 1, 1, 1]),
    "mostly_hard": Mix(name="mostly_hard", decile_weights=[1, 1, 1, 1, 1, 2, 3, 4, 5, 6]),
    "bimodal":     Mix(name="bimodal",     decile_weights=[5, 3, 1, 0.3, 0.2, 0.2, 0.3, 1, 3, 5]),
    "midheavy":    Mix(name="midheavy",    decile_weights=[0.5, 1, 2, 4, 5, 5, 4, 2, 1, 0.5]),
}


@dataclass(frozen=True)
class WorkloadSpec:
    """Every knob that changes what the bench measures, in one declared object.

    `leakage` and `signal` are swept axes, not constants. At leakage=1 and signal=1 the
    latent difficulty is fully recoverable from what the router sees, every router is
    perfect, and the bench measures nothing — which is exactly why the headline is
    published as a curve over `signal` rather than as a point.
    """

    mix: str = "balanced"
    n: int = 20_000
    # How strongly input length reveals difficulty. The spec's own first feature is
    # token count, so this is the coupling that decides whether the naive cost metric
    # rewards routing well or routing badly.
    leakage: float = 0.75
    # How much of the latent difficulty a semantic feature carries.
    signal: float = 0.70
    seed_tag: str = "v1"


def _difficulty_from(mix: Mix, u: float) -> float:
    """Map a uniform draw onto the mix's decile distribution."""
    weights = mix.weights()
    acc = 0.0
    for index, w in enumerate(weights):
        if u < acc + w:
            within = (u - acc) / w if w > 0 else 0.0
            return min(0.999999, (index + within) / 10.0)
        acc += w
    return 0.999999


def build_workload(spec: WorkloadSpec) -> list[Case]:
    """Generate the fixture's cases. Pure function of the spec — no RNG, no clock."""
    mix = MIXES[spec.mix]
    cases: list[Case] = []
    for i in range(spec.n):
        cid = f"{spec.seed_tag}:{spec.mix}:{i}"
        d = _difficulty_from(mix, _unit_draw(cid, "difficulty"))

        # Family assignment. exact_match_list carries the n_items dimension, where the
        # cheap-vs-frontier gap is non-monotone and no surface feature tracks it.
        fu = _unit_draw(cid, "family")
        family = ("structured_extraction" if fu < 0.20
                  else "exact_match_list" if fu < 0.35 else "default")
        n_items = 1
        if family == "exact_match_list":
            n_items = 1 + int(_unit_draw(cid, "items") * 120)

        # THE COUPLING. Length is a leaky view of difficulty: at leakage=1 length is a
        # monotone function of d and the router can read difficulty straight off the
        # token count; at leakage=0 length says nothing.
        noise = _unit_draw(cid, "length-noise")
        length_signal = spec.leakage * d + (1.0 - spec.leakage) * noise
        input_tokens = int(200 + 2400 * length_signal)
        output_tokens = int(80 + 520 * length_signal)

        cases.append(Case(case_id=cid, difficulty=d, input_tokens=input_tokens,
                          output_tokens=output_tokens, family=family, n_items=n_items))
    return cases


def observed_signal(case: Case, spec: WorkloadSpec) -> float:
    """What a router can learn about difficulty beyond the raw token count.

    Deliberately lossy. If this returned `case.difficulty` the classifier would be a
    lookup and its accuracy would be a property of this function rather than of the
    router.
    """
    noise = _unit_draw(case.case_id, "signal-noise", spec.seed_tag)
    return spec.signal * case.difficulty + (1.0 - spec.signal) * noise


def features(case: Case, spec: WorkloadSpec) -> dict[str, float]:
    """Exactly what a policy is allowed to see. Never `case.difficulty`."""
    return {
        "input_tokens": float(case.input_tokens),
        "output_tokens": float(case.output_tokens),
        "n_items": float(case.n_items),
        "is_extraction": 1.0 if case.family == "structured_extraction" else 0.0,
        "signal": observed_signal(case, spec),
    }
