"""Composition with llm-gateway: this repo CHOOSES a model, the gateway SERVES it.

The two route on different things and neither duplicates the other. llm-gateway routes
on FAILURE -- the caller names a model, the gateway falls through a chain with admission
control, budget reservation and circuit breaking. route-regret routes on CONTENT -- it
reads the request and decides which model is adequate. The seam is a `Decision`.

Three things break at that seam, and all three break in the direction that looks like
good news:

1. **A single preferred model is not enough information.** The gateway may substitute
   during an outage, and it substitutes down its own per-TIER chain -- which knows
   nothing about why this request needed capability. The router's second choice is
   strictly better information, so the seam carries a RANKED list.

2. **Quality must be attributed to `Served.model_served`, never to the choice.** Filing
   the outcome against the model the router REQUESTED credits a model that did not run,
   with another model's results, for the whole duration of an incident -- which is to
   say it poisons the training set exactly when things are going wrong. And a request
   the gateway substituted is not evaluable at all: you cannot score a choice that was
   not honoured, so it leaves the evaluation set rather than being scored leniently.

3. **Shed load is not a saving.** Divide spend by INTENDED requests and a capacity
   squeeze reports a spectacular cost reduction that is entirely requests never served.
   The honest denominator is the SERVED count, printed next to the shed count, and
   `audit_shed_load` proves the gap is shed load by replaying the survivors.

What this module does NOT claim: the response text. `FixtureUpstream` returns
`f"[{provider}/{model}] response"` -- a pure function of the model name, byte-identical
for a case that model aces and one it fails. It therefore cannot express route-regret's
capability claim, and quality ground truth stays with `route_regret.models.succeeds`.
That is not a defect in llm-gateway: its claims are about capacity and the ledger, where
the response text is correctly irrelevant, and inventing quality semantics for it would
have made the fixture an oracle for a question it never asked.

THE ONE PLACE THE SEAM DOES NOT EXIST YET. `llm_gateway.models.Request` carries
`model: str` and nothing else; the chain is `config.chain_for(request.model)`, derived
from the model's tier (router.py:76). A ranked list cannot be handed to the shipped
Router. `Deployment.build` works around it by constructing one `GatewayConfig` per
distinct ranking over SHARED admission, resilience and upstream state -- so capacity,
breakers and budget are genuinely one system and only the chain differs. It is a workaround and
not a design: the honest fix upstream is one optional field, `Request.preferred_models:
list[str] | None`, consumed in place of `chain_for` when present. That change belongs in
llm-gateway, not here, so it is named rather than faked.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from llm_gateway.admission import AdmissionController, Budget
from llm_gateway.clock import ManualClock
from llm_gateway.config import FallbackChain, GatewayConfig
from llm_gateway.models import (ModelPricing, ModelSpec, ProviderSpec, Request, Team,
                                TeamLimits)
from llm_gateway.providers import FixtureUpstream
from llm_gateway.resilience import ResiliencePolicy, RetryBudget
from llm_gateway.router import Router
from prompt_experiments.stats.proportions import wilson_interval

from route_regret.fixture import REGISTRY, WorkloadSpec, features
from route_regret.models import Case, ModelCard, succeeds
from route_regret.policies import Policy

BY_NAME: dict[str, ModelCard] = {m.name: m for m in REGISTRY}

# The provider partition is load-bearing, not scenery. Every scarce resource in
# llm-gateway -- the request window, the token window, the circuit breaker -- is held
# PER PROVIDER, so which vendor a model belongs to decides whether two routing choices
# compete for the same capacity. In this registry the two cheapest models share one
# provider, which is why a price-greedy router concentrates load rather than spreading
# it. Assigning these by vendor rather than by price is the whole point.
PROVIDER_OF: dict[str, str] = {
    "gpt-4o-mini": "openai",
    "gpt-4o": "openai",
    "haiku-4.5": "anthropic",
    "sonnet-5": "anthropic",
    "opus-5": "anthropic",
}

TEAM_ID = "bench"
API_KEY = "bench-key"

# Non-binding by default, so that any capacity effect a test reports is one the test
# asked for. A fixture whose limits bind by accident reports its own defaults back.
UNBOUNDED = 10 ** 9


# ---- the seam ------------------------------------------------------------
@dataclass(frozen=True)
class Decision:
    """What a content router hands a failure router.

    `preferred_models` is RANKED rather than singular because the gateway may substitute
    during an outage, and the router's runner-up beats a generic tier chain. `difficulty`
    and `confidence` are the router's own beliefs, derived only from what a policy is
    allowed to see -- never from `Case.difficulty`, which would make every downstream
    number a property of the fixture.
    """

    case_id: str
    preferred_models: tuple[str, ...]
    difficulty: float
    confidence: float
    reason: str

    @property
    def head(self) -> str:
        return self.preferred_models[0]


def rank_from(head: str, registry: list[ModelCard] | None = None) -> tuple[str, ...]:
    """The router's substitution order, which is not the gateway's.

    Adequate substitutes first -- every model at or above the head's capability, cheapest
    first -- then the inadequate tail ordered by capability descending. The gateway's own
    chain is per tier and would happily substitute DOWN the price ladder, because it has
    no idea the request needed capability; the router does, and this is the only place
    that knowledge can be expressed.
    """
    cards = registry or REGISTRY
    bar = {m.name: m for m in cards}[head].capability["default"]
    rest = [m for m in cards if m.name != head]
    adequate = sorted((m for m in rest if m.capability["default"] >= bar),
                      key=lambda m: m.cost(1000, 500))
    inadequate = sorted((m for m in rest if m.capability["default"] < bar),
                        key=lambda m: -m.capability["default"])
    return (head, *[m.name for m in adequate], *[m.name for m in inadequate])


def _confidence(policy: Policy, case: Case, spec: WorkloadSpec) -> tuple[float, str]:
    """How far the router's own observation sits from its nearest decision boundary.

    A policy that publishes no boundaries gets 0.0, and that default is deliberate:
    refusing an inadequate substitute is a claim that the router knows better than the
    gateway, and a router making no confidence claim has not earned the right to make it.
    """
    cuts = getattr(policy, "cuts", None)
    if not cuts:
        return 0.0, "policy publishes no decision boundary; substitution unrestricted"
    observed = features(case, spec)["signal"] + getattr(policy, "offset", 0.0)
    nearest = min(abs(observed - bound) for bound, _ in cuts)
    return nearest, f"observed signal {observed:.3f}, {nearest:.3f} from the nearest cut"


def decide(policy: Policy, case: Case, spec: WorkloadSpec, *,
           confidence_floor: float = 0.0,
           registry: list[ModelCard] | None = None) -> Decision:
    """Turn one routing choice into the ranked instruction the gateway can act on.

    `confidence_floor` is what makes confidence load-bearing rather than decorative. A
    declared field nothing consumes is a claim of capability the code does not have --
    llm-gateway deleted `Request.stream` for exactly that. Here, a router confident this
    request needs capability truncates its ranking at the last ADEQUATE model, so the
    gateway reports no_capacity rather than quietly serving something the router believes
    cannot do the job. That is `allow_fallback: false` decided per request instead of per
    team, which is the finer grain the content router is the only party able to supply.
    """
    cards = registry or REGISTRY
    head = policy.choose(case, spec).name
    ranked = rank_from(head, cards)
    confidence, why = _confidence(policy, case, spec)

    if confidence >= confidence_floor > 0.0:
        bar = {m.name: m for m in cards}[head].capability["default"]
        ranked = tuple(m for m in ranked
                       if {c.name: c for c in cards}[m].capability["default"] >= bar)
        why += f"; confident at {confidence:.3f} >= {confidence_floor:.3f}, so a model " \
               "below the bar is refused rather than substituted in"
    return Decision(case_id=case.case_id, preferred_models=ranked,
                    difficulty=features(case, spec)["signal"],
                    confidence=confidence, reason=f"{policy.name}: {why}")


@dataclass(frozen=True)
class ServedCall:
    """One route-regret request after the gateway has finished with it.

    `quality_ok` is None whenever nothing was served. A request that was never answered
    has no quality, and defaulting it to False would count a capacity refusal as a model
    failure -- which is how an outage becomes a quality regression in the record.
    """

    case_id: str
    decision: Decision
    outcome: str
    model_requested: str
    model_served: str
    served: bool
    substituted: bool
    quality_ok: bool | None
    billed_usd: float
    wasted_usd: float
    attempts: int
    error: str
    # Kept only so `test_the_gateway_fixtures_text_cannot_carry_the_capability_claim`
    # can demonstrate that it carries no quality information. Nothing else reads it.
    response_text: str
    # The ceiling admission granted, when it granted less than was asked for. A clamped
    # request is served with a SHORTER answer, so it genuinely costs less -- which is a
    # real saving sitting inside a capacity squeeze and looking exactly like shed load.
    # `audit_shed_load` has to separate the two or its headline is an accident of which
    # limit was squeezed and how hard.
    clamped_from: int | None = None

    @property
    def honoured(self) -> bool:
        """Served, by the model the router actually asked for. The only requests that
        say anything about the routing policy."""
        return self.served and not self.substituted

    @property
    def spend_usd(self) -> float:
        """Including failed attempts. A ledger that bills only the successful call
        under-counts precisely during an incident, always in the safe-looking direction."""
        return self.billed_usd + self.wasted_usd


@dataclass
class Deployment:
    """route-regret's registry, wired into a real llm-gateway request path."""

    routers: dict[str, Router]
    upstreams: dict[str, FixtureUpstream]
    admission: AdmissionController
    clock: ManualClock
    confidence_floor: float = 0.0
    # Rankings truncated by confidence are not a pure function of the head model, so
    # their configs are built on demand and kept. Rebuilding one per request would make
    # config construction a cost of the workload rather than of the registry.
    _chain_routers: dict[tuple[str, ...], Router] = field(default_factory=dict)

    @classmethod
    def build(cls, *, registry: list[ModelCard] | None = None,
              provider_rpm: dict[str, int] | None = None,
              provider_tpm: int = UNBOUNDED,
              provider_concurrency: int = UNBOUNDED,
              team_tokens_per_minute: int = UNBOUNDED,
              team_daily_budget_usd: float = 1e6,
              team_concurrency: int = UNBOUNDED,
              outages: dict[str, list[tuple[float, float]]] | None = None,
              allow_fallback: bool = True,
              confidence_floor: float = 0.0) -> "Deployment":
        cards = registry or REGISTRY
        rpm = provider_rpm or {}
        specs = {
            name: ProviderSpec(name=name,
                               requests_per_minute=rpm.get(name, UNBOUNDED),
                               tokens_per_minute=provider_tpm,
                               max_concurrent=provider_concurrency)
            for name in sorted(set(PROVIDER_OF[c.name] for c in cards))
        }
        # One clock for the whole deployment. Two would let a provider's rate window and
        # a breaker's cooldown disagree about what "now" is, which is a bug that only
        # appears once something is degraded.
        clock = ManualClock()
        upstreams = {
            name: FixtureUpstream(spec=spec, clock=clock,
                                  outages=list((outages or {}).get(name, [])))
            for name, spec in specs.items()
        }
        admission = AdmissionController(clock)
        admission.register(TEAM_ID, Budget(tokens_per_minute=team_tokens_per_minute,
                                           daily_budget_usd=team_daily_budget_usd,
                                           max_concurrent=team_concurrency))
        resilience = ResiliencePolicy(clock, RetryBudget())

        # One config per distinct ranking. See the module docstring: the shipped Request
        # has no per-request chain field, and the ranking is a pure function of the head
        # model, so this is bounded by the registry rather than by the workload.
        routers = {}
        for card in cards:
            models = {
                other.name: ModelSpec(
                    name=other.name, provider=PROVIDER_OF[other.name],
                    # `chain_for` only reads the REQUESTED model's tier, so a private
                    # tier on the head is enough to make the chain the router's ranking.
                    tier=f"rr:{card.name}" if other.name == card.name else "rr:other",
                    pricing=ModelPricing(input_per_mtok=other.price_in_per_mtok,
                                         output_per_mtok=other.price_out_per_mtok),
                    # Above anything the fixture emits, so a clamp is always the
                    # gateway's decision rather than this adapter's ceiling.
                    max_output_tokens=UNBOUNDED)
                for other in cards
            }
            ranking = rank_from(card.name, cards)
            config = GatewayConfig(
                providers=specs, models=models,
                teams={TEAM_ID: Team(id=TEAM_ID, api_key=API_KEY,
                                     limits=TeamLimits(), allow_fallback=allow_fallback)},
                fallback={f"rr:{card.name}": FallbackChain(tier=f"rr:{card.name}",
                                                           models=list(ranking))},
                version="route-regret-seam")
            routers[card.name] = Router(config, upstreams, admission, resilience, clock)
        return cls(routers=routers, upstreams=upstreams, admission=admission,
                   clock=clock, confidence_floor=confidence_floor)

    def serve(self, cases: list[Case], policy: Policy,
              spec: WorkloadSpec) -> list[ServedCall]:
        calls: list[ServedCall] = []
        for case in cases:
            decision = decide(policy, case, spec,
                              confidence_floor=self.confidence_floor)
            router = self.routers[decision.head]
            if decision.preferred_models != tuple(
                    router.config.chain_for(decision.head)):
                # The workaround has a failure mode worth catching: a truncated ranking
                # that silently got the untruncated config would let the gateway
                # substitute below a bar the router refused, and the run would look fine.
                router = self._chain_routers.get(decision.preferred_models) \
                    or self._router_for(decision, router)
                self._chain_routers[decision.preferred_models] = router
            request = Request(
                team_id=TEAM_ID, model=decision.head,
                # The gateway estimates input length as len(prompt)//4. Synthesising
                # exactly 4x characters carries the case's own token count into the
                # gateway's admission arithmetic instead of inventing a second one.
                prompt="x" * (4 * case.input_tokens),
                max_tokens=case.output_tokens, request_id=case.case_id)
            result = router.handle(API_KEY, request)
            served = result.served
            ok = served.outcome == "ok"
            calls.append(ServedCall(
                case_id=case.case_id, decision=decision, outcome=served.outcome,
                model_requested=served.model_requested or decision.head,
                model_served=served.model_served, served=ok,
                substituted=served.substituted,
                # ATTRIBUTION. Against `model_served`, always. The model the router asked
                # for is not the model whose answer the user read.
                quality_ok=succeeds(BY_NAME[served.model_served], case) if ok else None,
                billed_usd=served.usage.cost_usd, wasted_usd=result.wasted_usd,
                attempts=served.attempts, error=served.error,
                response_text=served.text,
                clamped_from=next((a.clamped_from for a in reversed(result.attempts)
                                   if a.ok), None)))
        return calls

    def _router_for(self, decision: Decision, template: Router) -> Router:
        """A Router over the same shared state whose chain is this exact ranking.

        Needed only because a truncated ranking is not a pure function of the head model.
        Capacity, breakers and budget are the template's, so this is one gateway with a
        different chain rather than a second gateway.
        """
        base = template.config
        tier = f"rr:{'|'.join(decision.preferred_models)}"
        models = {name: spec.model_copy(update={"tier": tier if name == decision.head
                                                else "rr:other"})
                  for name, spec in base.models.items()}
        config = GatewayConfig(
            providers=base.providers, models=models, teams=base.teams,
            fallback={tier: FallbackChain(tier=tier,
                                          models=list(decision.preferred_models))},
            version=base.version)
        return Router(config, template.upstreams, template.admission,
                      template.resilience, template.clock, template.metrics)


# ---- attribution ---------------------------------------------------------
@dataclass(frozen=True)
class Attribution:
    """Quality filed against one model name, with the interval it was measured to."""

    model: str
    n: int
    ok: int
    rate: float
    low: float
    high: float

    def line(self) -> str:
        return (f"  {self.model:<14} n={self.n:>5}  ok={self.rate:6.1%}  "
                f"[{self.low:.1%}, {self.high:.1%}]")


def attribute(calls: list[ServedCall], *, key: str = "served") -> dict[str, Attribution]:
    """Tally delivered quality by model.

    `key="served"` is the correct one and the only one anything downstream should use.
    `key="requested"` is kept as a visible control, because the defect it produces is
    invisible in a healthy run and enormous during an outage: it files another model's
    outcomes against a model that ran zero times, and it does so for exactly as long as
    the incident lasts.
    """
    if key not in ("served", "requested"):
        raise ValueError(f"attribute by 'served' or 'requested', not {key!r}")
    buckets: dict[str, list[ServedCall]] = {}
    for call in calls:
        if not call.served:
            continue
        name = call.model_served if key == "served" else call.model_requested
        buckets.setdefault(name, []).append(call)

    out: dict[str, Attribution] = {}
    for name, rows in sorted(buckets.items()):
        ok = sum(1 for r in rows if r.quality_ok)
        interval = wilson_interval(ok, len(rows))
        out[name] = Attribution(model=name, n=len(rows), ok=ok, rate=ok / len(rows),
                                low=interval.low, high=interval.high)
    return out


@dataclass(frozen=True)
class EvaluationSet:
    """The requests a routing policy may be scored on, and everything that left."""

    honoured: list[ServedCall]
    excluded: list[ServedCall]
    reasons: Counter

    @property
    def n(self) -> int:
        return len(self.honoured)

    @property
    def violation_rate(self) -> float:
        if not self.honoured:
            return 0.0
        return 1.0 - sum(1 for c in self.honoured if c.quality_ok) / len(self.honoured)

    def line(self) -> str:
        dropped = ", ".join(f"{n} {why}" for why, n in sorted(self.reasons.items()))
        return (f"  evaluable {self.n} of {self.n + len(self.excluded)} requests  "
                f"violation {self.violation_rate:.1%}  excluded: {dropped or 'none'}")


def evaluable(calls: list[ServedCall]) -> EvaluationSet:
    """Split a run into what can and cannot be used to score the ROUTER.

    A substituted request is not a lenient data point, it is not a data point: the answer
    came from a model the router did not choose, for a reason the router had no part in.
    Scoring it credits or blames the policy for the gateway's fallback logic. A request
    that was never served has no quality at all.
    """
    honoured, excluded, reasons = [], [], Counter()
    for call in calls:
        if not call.served:
            excluded.append(call)
            reasons[f"never served ({call.outcome})"] += 1
        elif call.substituted:
            excluded.append(call)
            reasons["substituted by the gateway"] += 1
        else:
            honoured.append(call)
    return EvaluationSet(honoured=honoured, excluded=excluded, reasons=reasons)


# ---- spend ---------------------------------------------------------------
@dataclass(frozen=True)
class SpendReport:
    """What a run cost, with both denominators in the open."""

    label: str
    intended: int
    served: int
    billed_usd: float
    wasted_usd: float

    @property
    def shed(self) -> int:
        return self.intended - self.served

    @property
    def shed_rate(self) -> float:
        return self.shed / self.intended if self.intended else 0.0

    @property
    def spend_usd(self) -> float:
        return self.billed_usd + self.wasted_usd

    @property
    def naive_cost_per_intended(self) -> float:
        """The dashboard number. Falls whenever the gateway serves fewer requests,
        whether or not anything got cheaper."""
        return self.spend_usd / self.intended if self.intended else 0.0

    @property
    def cost_per_served(self) -> float:
        return self.spend_usd / self.served if self.served else float("nan")

    def line(self) -> str:
        return (f"  {self.label:<26} spend ${self.spend_usd:8.4f}  "
                f"served {self.served:>5} of {self.intended:>5} intended "
                f"({self.shed} shed)  ${self.cost_per_served:.6f}/served  "
                f"${self.naive_cost_per_intended:.6f}/intended")


def spend_report(calls: list[ServedCall], *, label: str = "run") -> SpendReport:
    return SpendReport(
        label=label, intended=len(calls), served=sum(1 for c in calls if c.served),
        billed_usd=sum(c.billed_usd for c in calls),
        wasted_usd=sum(c.wasted_usd for c in calls))


@dataclass(frozen=True)
class ShedAudit:
    """A capacity squeeze, priced both ways, with the difference accounted for."""

    baseline: SpendReport
    squeezed: SpendReport
    # The baseline replayed on EXACTLY the requests the squeeze served. Shed requests
    # are refused at admission and never reach an upstream, so the surviving sequence of
    # upstream calls is identical and the replay reproduces the squeezed spend to the
    # cent -- which is what turns "the saving is small" into "the saving is zero".
    # It reproduces it only while every survivor was served the same way. Admission's
    # whole design is that a squeeze CLAMPS rather than refuses, and a clamped survivor
    # is a shorter answer, genuinely cheaper: the exactness is a property of which limit
    # was squeezed and how hard, not of the audit. `clamped_under_squeeze` says which.
    matched: SpendReport
    decisions_unchanged: bool
    # `decisions_unchanged` compares the model the ROUTER asked for, which is a pure
    # function of (policy, case, spec) and therefore cannot move when only the
    # deployment is squeezed -- a control that is true by construction is not a control.
    # These two are the quantities that actually can move, and each one puts a real
    # saving inside the squeeze that is not shed load: a survivor served by a DIFFERENT
    # model, and a survivor served a SHORTER answer.
    served_model_changes: int = 0
    clamped_under_squeeze: int = 0

    @property
    def confounded(self) -> bool:
        return bool(self.served_model_changes or self.clamped_under_squeeze)

    @property
    def naive_saving(self) -> float:
        return 1.0 - (self.squeezed.naive_cost_per_intended
                      / self.baseline.naive_cost_per_intended)

    @property
    def matched_saving(self) -> float:
        """The only one of the three that is a cost measurement: the same requests,
        served the same way, priced against what they cost when nothing was shed."""
        return 1.0 - self.squeezed.cost_per_served / self.matched.cost_per_served

    def lines(self) -> list[str]:
        # Computed from the reports, never typed. The verdict below is a lookup on the
        # sign of the two numbers, so it cannot say something the data does not.
        verdict = {
            (True, True): "both agree a saving happened",
            (True, False): "the entire apparent saving is load that was never served",
            (False, True): "the naive number missed a real saving",
            (False, False): "neither reports a saving",
        }[(self.naive_saving > 1e-9, self.matched_saving > 1e-9)]
        # A matched saving is only a cost measurement while the survivors were served
        # the same way. Under a squeeze that substitutes or truncates them it is a
        # model-mix or answer-length effect, and reporting it as "a saving happened"
        # commits this module's own defect one level up.
        confounds = [f"{n} {why}" for n, why in
                     ((self.served_model_changes, "survivors were served by a different "
                                                  "model than in the baseline"),
                      (self.clamped_under_squeeze, "survivors had their answer clamped "
                                                   "shorter by admission")) if n]
        if confounds:
            verdict += "; but " + " and ".join(confounds) + \
                       ", so the matched figure is that change, not a cost saving"
        return [
            self.baseline.line(), self.squeezed.line(), self.matched.line(),
            f"  naive saving   {self.naive_saving:+7.1%}   "
            f"(spend / intended requests)",
            f"  matched saving {self.matched_saving:+7.1%}   "
            f"(spend / served requests, versus the same requests unsqueezed)",
            f"  shed rate      {self.squeezed.shed_rate:+7.1%}   "
            f"({self.squeezed.shed} of {self.squeezed.intended} never served)",
            f"  verdict: {verdict}"
            + ("; the router asked for the same model in both runs"
               if self.decisions_unchanged else
               "; the routing decisions differed, so the comparison is confounded"),
        ]


def audit_shed_load(factory, cases: list[Case], policy: Policy,
                    spec: WorkloadSpec) -> ShedAudit:
    """Price a capacity squeeze the naive way and the correct way.

    `factory(squeezed: bool)` returns a fresh Deployment. Three runs: unsqueezed over
    everything, squeezed over everything, and unsqueezed over exactly the requests the
    squeeze managed to serve. The third is the control that makes the claim falsifiable
    -- without it, "the per-served cost barely moved" is a judgement about how small
    "barely" is.
    """
    baseline = factory(False).serve(cases, policy, spec)
    squeezed = factory(True).serve(cases, policy, spec)
    # Order preserved. The gateway's upstream fixture seeds its output-length draw on a
    # per-provider call counter, so a replay in a different order is a different run.
    lived = {c.case_id for c in squeezed if c.served}
    matched = factory(False).serve([c for c in cases if c.case_id in lived], policy, spec)

    decided = {c.case_id: c.model_requested for c in baseline}
    unchanged = all(decided[c.case_id] == c.model_requested for c in squeezed)
    # Against the UNSQUEEZED baseline, not against the matched replay: the replay is
    # already restricted to the survivors, so comparing the two would ask whether the
    # squeeze changed the run it was measured against and always answer no.
    base_served = {c.case_id: c.model_served for c in baseline}
    return ShedAudit(
        baseline=spend_report(baseline, label="baseline"),
        squeezed=spend_report(squeezed, label="under a capacity squeeze"),
        matched=spend_report(matched, label="baseline, survivors only"),
        decisions_unchanged=unchanged,
        served_model_changes=sum(1 for c in squeezed if c.served
                                 and base_served[c.case_id] != c.model_served),
        clamped_under_squeeze=sum(1 for c in squeezed
                                  if c.served and c.clamped_from is not None))
