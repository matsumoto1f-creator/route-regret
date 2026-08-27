"""The objects the bench measures over.

Two separations are load-bearing here, and both exist because the alternative makes a
measurement come out right by construction rather than by the system working.

1. A model's CAPABILITY is authored independently of its PRICE. If capability were
   derived from price, "route to the cheapest adequate model" would be the same
   instruction as "route to the cheapest model", and the bench could not distinguish a
   router that understands the workload from one that always picks the bottom of the
   ladder. Real price ladders are not capability ladders — at Aug-2026 list prices
   Claude Haiku 4.5 costs 7.8x GPT-4o-mini while being the stronger model.

2. A case's DIFFICULTY is declared, not derived from its text. If difficulty were a
   function of the surface features the router reads, a perfect router would be a
   trivial regression and accuracy would be 1.0 by construction. This is llm-gateway's
   `token_skew` lesson in a new costume: the instrument must not share an expression
   with the thing it measures.
"""

from __future__ import annotations

import hashlib
import math

from pydantic import BaseModel, Field, field_validator

# The logistic steepness of the capability curve. Higher k makes "can this model do
# this case" closer to a step function at theta == d.
CAPABILITY_STEEPNESS = 12.0


class ModelCard(BaseModel):
    """One model on the bench: what it costs, and separately, what it can do."""

    name: str
    price_in_per_mtok: float
    price_out_per_mtok: float

    # Capability by case family, with `default` used for families not listed. Authored
    # by hand in the fixture manifest, never computed from price. A family entry above
    # the default is how "the cheap model is actually better here" is expressed, which
    # the bench needs or the metric is silently capped at agreement-with-frontier.
    capability: dict[str, float] = Field(default_factory=lambda: {"default": 0.5})

    def theta(self, family: str) -> float:
        return self.capability.get(family, self.capability.get("default", 0.5))

    def cost(self, input_tokens: int, output_tokens: int) -> float:
        return (input_tokens * self.price_in_per_mtok / 1e6
                + output_tokens * self.price_out_per_mtok / 1e6)

    @field_validator("capability")
    @classmethod
    def _needs_a_default(cls, v: dict[str, float]) -> dict[str, float]:
        if "default" not in v:
            raise ValueError("capability must carry a 'default' entry")
        return v


class Case(BaseModel):
    """One request in the fixture workload.

    `difficulty` is the latent truth. `input_tokens` correlates with it by `leakage`
    at generation time — which is not a modelling flourish but the single most
    important property of the workload: the spec's own first feature is token count,
    so if hard prompts are long, a router that routes BADLY sends the expensive long
    prompts to the cheap model and books a larger saving. That coupling is why the
    obvious cost metric is anti-correlated with routing correctly.
    """

    case_id: str
    difficulty: float = Field(ge=0.0, le=1.0)
    input_tokens: int = Field(gt=0)
    output_tokens: int = Field(gt=0)
    family: str = "default"
    # Exact-match tasks with N sub-items: per-item accuracy q gives success q**n_items.
    # The cheap-vs-frontier gap is NON-MONOTONE in n_items, which no surface feature
    # captures, and which is the sharpest argument for labelling on observed adequacy
    # rather than on a human's impression of complexity.
    n_items: int = Field(default=1, ge=1)


class Mix(BaseModel):
    """A named traffic profile: how difficulty is distributed over the workload.

    Shipped as several variants because no scalar summary of a router is invariant to
    the mix. Reporting one number without naming the mix it was measured on is the
    defect this class exists to make impossible to commit by accident.
    """

    name: str
    # Piecewise weights over difficulty deciles; normalised on use.
    decile_weights: list[float] = Field(min_length=10, max_length=10)

    def weights(self) -> list[float]:
        total = sum(self.decile_weights)
        if total <= 0:
            raise ValueError(f"mix {self.name!r} has non-positive total weight")
        return [w / total for w in self.decile_weights]


def _unit_draw(*parts: str) -> float:
    """A deterministic number in [0, 1) from the given parts.

    sha256 rather than `random`: the bench must be reproducible from a fresh clone with
    no seed handling, no RNG state threaded through call sites, and no dependence on
    evaluation order. Two runs of the same fixture must agree to the cent.
    """
    digest = hashlib.sha256("|".join(parts).encode()).hexdigest()
    return int(digest[:12], 16) / float(1 << 48)


def per_item_accuracy(card: ModelCard, case: Case) -> float:
    """P(this model gets ONE item of this case right)."""
    gap = card.theta(case.family) - case.difficulty
    return 1.0 / (1.0 + math.exp(-CAPABILITY_STEEPNESS * gap))


def succeeds(card: ModelCard, case: Case) -> bool:
    """Ground truth: did this model produce an acceptable answer for this case?

    Deterministic in (case_id, model). No RNG, no clock, no evaluation-order dependence.
    """
    q = per_item_accuracy(card, case)
    return _unit_draw(case.case_id, card.name) < q ** case.n_items
