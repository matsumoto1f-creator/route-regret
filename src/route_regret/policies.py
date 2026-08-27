"""Routing policies. The classifier is one entrant here, not the product.

Every policy is a function from what a router can SEE to a model choice. `Oracle` is
the exception and is deliberately unbuildable: it is handed the ground truth, because
it exists to be a denominator rather than a target.
"""

from __future__ import annotations

from typing import Protocol

from route_regret.fixture import REGISTRY, WorkloadSpec, features
from route_regret.models import Case, ModelCard, _unit_draw, succeeds


class Policy(Protocol):
    name: str
    def choose(self, case: Case, spec: WorkloadSpec) -> ModelCard: ...
    def propensity(self, case: Case, spec: WorkloadSpec, model: ModelCard) -> float: ...


def _by_price(registry: list[ModelCard]) -> list[ModelCard]:
    return sorted(registry, key=lambda m: m.cost(1000, 500))


class AlwaysTop:
    """The incumbent policy almost everyone actually runs, and the reference line the
    savings figure is measured against. 'Top' is by CAPABILITY, not by price."""
    name = "always_top"

    def __init__(self, registry: list[ModelCard] | None = None) -> None:
        self.registry = registry or REGISTRY
        self.model = max(self.registry, key=lambda m: m.capability["default"])

    def choose(self, case: Case, spec: WorkloadSpec) -> ModelCard:
        return self.model

    def propensity(self, case, spec, model) -> float:
        return 1.0 if model.name == self.model.name else 0.0


class AlwaysCheapest:
    """The degenerate policy that maximises the naive savings metric while failing a
    third of requests. It is on the bench precisely so that metric can be seen losing."""
    name = "always_cheapest"

    def __init__(self, registry: list[ModelCard] | None = None) -> None:
        self.registry = registry or REGISTRY
        self.model = _by_price(self.registry)[0]

    def choose(self, case, spec) -> ModelCard:
        return self.model

    def propensity(self, case, spec, model) -> float:
        return 1.0 if model.name == self.model.name else 0.0


class Oracle:
    """Clairvoyant: the cheapest model that actually succeeds on this case.

    Unbuildable by construction — it reads ground truth. It is the denominator of
    FASC, not something a router is asked to approach.
    """
    name = "oracle"

    def __init__(self, registry: list[ModelCard] | None = None) -> None:
        self.registry = registry or REGISTRY

    def choose(self, case, spec) -> ModelCard:
        for m in _by_price(self.registry):
            if succeeds(m, case):
                return m
        return max(self.registry, key=lambda m: m.capability["default"])

    def propensity(self, case, spec, model) -> float:
        return 1.0


class ContentBlind:
    """The floor a content-aware router must beat to have earned anything.

    Chooses at random with the SAME marginal distribution over models as the policy it
    is being compared to — so any margin over it is attributable to reading the request
    rather than to the tier mix. A router that cannot beat this has learned nothing,
    and no savings number it reports means anything.
    """
    name = "content_blind"

    def __init__(self, marginal: dict[str, float], registry: list[ModelCard] | None = None) -> None:
        self.registry = registry or REGISTRY
        self.marginal = marginal
        self._by_name = {m.name: m for m in self.registry}

    def choose(self, case, spec) -> ModelCard:
        u = _unit_draw(case.case_id, "blind", spec.seed_tag)
        acc = 0.0
        for name, p in self.marginal.items():
            acc += p
            if u < acc:
                return self._by_name[name]
        return self._by_name[next(iter(self.marginal))]

    def propensity(self, case, spec, model) -> float:
        return self.marginal.get(model.name, 0.0)


class ThresholdLadder:
    """A hand-tuned static rule over the observed signal — no training, no labels.

    On the bench as the honest 'did the classifier earn its complexity' baseline. If a
    trained model cannot beat two thresholds, the training set was ceremony.
    """
    name = "threshold_ladder"

    def __init__(self, cuts: list[tuple[float, str]], offset: float = 0.0,
                 registry: list[ModelCard] | None = None) -> None:
        self.registry = registry or REGISTRY
        self.cuts = cuts
        # A single knob that shifts every cut. Policies are compared at MATCHED quality,
        # so each one is first tuned until it just meets the constraint; comparing
        # untuned policies compares their arbitrary operating points, not their logic.
        self.offset = offset
        self._by_name = {m.name: m for m in self.registry}

    def choose(self, case, spec) -> ModelCard:
        s = features(case, spec)["signal"] + self.offset
        for bound, name in self.cuts:
            if s <= bound:
                return self._by_name[name]
        return self._by_name[self.cuts[-1][1]]

    def propensity(self, case, spec, model) -> float:
        return 1.0 if self.choose(case, spec).name == model.name else 0.0
