"""The seam between choosing a model and serving one.

Three of these are about the same defect wearing different clothes: a number computed
over the requests you INTENDED to make rather than the ones that were actually served
by the model you actually asked for. Attribution, evaluation and cost all break in that
one place, and all three break in the direction that looks like good news.

The fourth is a negative. The brief asked for the cost-versus-capacity tension via the
mechanism "cheap models are small and therefore fast"; llm-gateway cannot express that,
and the test that says so runs rather than being a sentence in a summary.
"""

from collections import Counter
from statistics import median

import pytest
from prompt_experiments.stats.proportions import two_proportion_test

from llm_gateway.models import ModelSpec

from route_regret.fixture import REGISTRY, WorkloadSpec, build_workload, features
from route_regret.gateway import (BY_NAME, PROVIDER_OF, Deployment, _confidence,
                                  attribute, audit_shed_load, decide, evaluable,
                                  rank_from, spend_report)
from route_regret.models import succeeds
from route_regret.policies import AlwaysCheapest, ThresholdLadder
from route_regret.report import DEFAULT_LADDER

N = 800
SPEC = WorkloadSpec(mix="balanced", n=N)
CASES = build_workload(SPEC)
LADDER = ThresholdLadder(DEFAULT_LADDER)


def _outage_on(provider: str) -> dict[str, list[tuple[float, float]]]:
    # The ManualClock never advances in these runs, so a window opening at 0 is a
    # total outage for the whole run rather than a blip.
    return {provider: [(0.0, 1e9)]}


# ---- 1. the seam ---------------------------------------------------------
def test_the_routers_ranked_list_is_the_chain_the_gateway_actually_walks():
    """A single preferred model would hand the outage to the gateway's generic tier
    chain. The router's own second choice is better information, and the test that it
    survives the seam is that the substitute is the router's runner-up."""
    deployment = Deployment.build(outages=_outage_on("anthropic"))
    calls = deployment.serve(CASES, LADDER, SPEC)

    substituted = [c for c in calls if c.substituted]
    assert substituted, "an outage on the provider holding three models must substitute"
    for call in substituted:
        ranked = call.decision.preferred_models
        healthy = [m for m in ranked if PROVIDER_OF[m] != "anthropic"]
        assert healthy, f"ranking {ranked} left the request nowhere healthy to go"
        assert call.model_served == healthy[0], (
            f"gateway served {call.model_served!r}; the router's own ranked list said "
            f"{healthy[0]!r} was the best surviving choice")


def test_the_ranked_list_prefers_capability_over_the_price_ladder():
    """The gateway's tier chain would substitute by tier. The router knows this request
    needed capability, so its runner-up is the cheapest model still AT OR ABOVE the head
    -- never the next rung down the price ladder."""
    for head in BY_NAME:
        ranked = rank_from(head)
        assert ranked[0] == head
        head_capability = BY_NAME[head].capability["default"]
        above = [m for m in ranked[1:] if BY_NAME[m].capability["default"] >= head_capability]
        below = [m for m in ranked[1:] if BY_NAME[m].capability["default"] < head_capability]
        assert list(ranked[1:]) == above + below, (
            f"{head}: adequate substitutes must be exhausted before inadequate ones")
        prices = [BY_NAME[m].cost(1000, 500) for m in above]
        assert prices == sorted(prices), f"{head}: adequate substitutes must be cheapest-first"


def test_the_decision_carries_the_routers_observation_not_the_fixtures_truth():
    """`Decision.difficulty` is the one field on the seam that could quietly become an
    oracle. Every number downstream of a decision -- what the gateway substituted, what
    the evaluation set contains, what the audit prices -- is then a property of the
    fixture rather than of the router, and nothing in the run looks wrong.

    This is `fixture.features` refusing to expose `case.difficulty`, enforced one layer
    out. Without it the seam is the hole that discipline was built to close.
    """
    decisions = [decide(LADDER, case, SPEC) for case in CASES]

    for decision, case in zip(decisions, CASES):
        assert decision.difficulty == features(case, SPEC)["signal"], (
            f"{case.case_id}: the decision reports difficulty {decision.difficulty!r} "
            f"where the policy could only see {features(case, SPEC)['signal']!r}")

    # The equality above is not enough on its own: at spec.signal = 1.0 the observed
    # signal IS the latent difficulty, so a leak would satisfy it. At the shipped
    # signal the two must disagree, or this test is passing on a degenerate workload.
    leaked = sum(1 for d, c in zip(decisions, CASES) if d.difficulty == c.difficulty)
    assert leaked == 0, (
        f"{leaked} of {len(CASES)} decisions report exactly the latent difficulty at "
        f"signal={SPEC.signal}; either ground truth is leaking or this workload cannot "
        "tell the difference")


# ---- 2. attribution ------------------------------------------------------
def test_quality_is_attributed_to_the_model_that_served_not_the_one_requested():
    """The defect that matters most, and it fires precisely during an incident.

    Filing the outcome against the model the router REQUESTED credits a model that ran
    zero times, with another model's results, for the whole duration of the outage.
    """
    deployment = Deployment.build(outages=_outage_on("anthropic"))
    calls = deployment.serve(CASES, LADDER, SPEC)

    by_served = attribute(calls, key="served")
    by_requested = attribute(calls, key="requested")

    anthropic_heads = [m for m in BY_NAME if PROVIDER_OF[m] == "anthropic"]
    ran = Counter(c.model_served for c in calls if c.served)
    poisoned = [m for m in anthropic_heads if by_requested.get(m)]
    assert poisoned, "the outage must leave requested-attribution rows to poison"

    # The structural half needs no statistics and cannot be noise: outcomes are on file
    # against models that ran zero times, for as long as the outage lasts.
    distortions, pooled_filed_ok, pooled_filed_n, pooled_truth_ok = {}, 0, 0, 0
    for model in poisoned:
        assert ran[model] == 0, f"{model} was supposed to be in a total outage"
        row = by_requested[model]
        assert row.n > 0, (
            f"requested-attribution files {row.n} outcomes against {model}, which ran "
            f"{ran[model]} times")
        # The numerical half: what the record says versus what that model would ACTUALLY
        # have scored on the same cases. Only route-regret can compute the second,
        # because only it holds the ground truth.
        rows = [c for c in calls if c.model_requested == model and c.served]
        truth = sum(succeeds(BY_NAME[model], _case_of(c)) for c in rows)
        test = two_proportion_test(row.ok, row.n, truth, len(rows))
        distortions[model] = test.effect
        pooled_filed_ok += row.ok
        pooled_filed_n += row.n
        pooled_truth_ok += truth
        assert test.p_value < 0.05, (
            f"{model}: the record says {row.rate:.1%} where the model itself would have "
            f"scored {truth / len(rows):.1%} (p={test.p_value:.3g}) -- if these agree, "
            "the misfiling did not distort and this fixture does not show the defect")

    # The distortions point in BOTH directions, because a single substitute is better
    # than some of the models it replaced and worse than others. That is why the fleet
    # average is not a safety net: it nets the errors against each other.
    assert min(distortions.values()) < 0 < max(distortions.values()), (
        f"distortions {distortions} all point the same way, so an aggregate would still "
        "have caught this and the per-model attribution would be a nicety")

    pooled = two_proportion_test(pooled_filed_ok, pooled_filed_n,
                                 pooled_truth_ok, pooled_filed_n)
    assert abs(pooled.effect) < min(abs(d) for d in distortions.values()), (
        f"the pooled view is off by {abs(pooled.effect):.3f} where the least-distorted "
        f"single model is off by {min(abs(d) for d in distortions.values()):.3f} -- the "
        "aggregate is supposed to be the blinder view, not the sharper one")

    # Served-attribution is a recomputation of the ground truth against model_served.
    for model, row in by_served.items():
        cases = [c for c in calls if c.served and c.model_served == model]
        assert row.n == len(cases)
        assert row.ok == sum(succeeds(BY_NAME[model], _case_of(c)) for c in cases)
        assert row.low < row.rate < row.high
        # Wilson, not Wald: its interval is pulled toward 0.5 and so is NOT centred on
        # the point estimate. A Wald interval is centred by construction, which is how a
        # hand-rolled replacement would slip in here without any assertion noticing.
        assert (row.low + row.high) / 2 != pytest.approx(row.rate, abs=1e-9), (
            f"{model}: the interval is centred on the point estimate, which Wilson "
            "never is -- a symmetric normal approximation has replaced it")


def test_substituted_requests_are_excluded_from_policy_evaluation_entirely():
    """You cannot evaluate a choice that was not honoured.

    A substituted request says nothing about the router: it was answered by a model the
    router did not pick, for a reason the router had no part in.
    """
    deployment = Deployment.build(outages=_outage_on("anthropic"))
    calls = deployment.serve(CASES, LADDER, SPEC)
    evaluation = evaluable(calls)

    assert evaluation.excluded, "the outage must exclude something"
    assert not any(c.substituted for c in evaluation.honoured)
    assert not any(not c.served for c in evaluation.honoured)
    assert evaluation.n == len(evaluation.honoured), (
        "the evaluation's denominator must be the honoured requests, not the intended ones")
    assert evaluation.n + len(evaluation.excluded) == len(calls)

    # The exclusion is not cosmetic: keeping the substituted requests in would report a
    # different quality for the same router.
    honoured_rate = evaluation.violation_rate
    all_served = [c for c in calls if c.served]
    naive_rate = 1.0 - sum(1 for c in all_served if c.quality_ok) / len(all_served)
    assert honoured_rate != pytest.approx(naive_rate, abs=1e-9), (
        f"including substitutions changes nothing here ({honoured_rate:.3f} vs "
        f"{naive_rate:.3f}), so this fixture does not demonstrate the exclusion mattering")

    # And the rate's own denominator, recomputed here rather than trusted. `n` being the
    # honoured count says nothing about what `violation_rate` divides by, and dividing
    # the honoured failures by the INTENDED count is the same defect as the cost one --
    # a quality figure that improves whenever the gateway drops more requests.
    ok = sum(1 for c in evaluation.honoured if c.quality_ok)
    assert honoured_rate == pytest.approx(1.0 - ok / len(evaluation.honoured), abs=1e-12)
    over_intended = 1.0 - ok / len(calls)
    assert honoured_rate != pytest.approx(over_intended, abs=1e-6), (
        f"the honoured rate {honoured_rate:.3f} is indistinguishable from the same "
        f"failures over all {len(calls)} intended requests ({over_intended:.3f}), so "
        "this run cannot show which denominator was used")


def test_a_refused_request_has_no_quality_and_still_cost_real_money():
    """Two named defects that only appear when the gateway gives up.

    `quality_ok` defaults to None rather than False because a capacity refusal is not a
    model failure, and counting it as one turns an outage into a quality regression in
    the record. And a refused request is not free: the attempts that failed on the way
    down the chain were billed, and a ledger summing only the successful call
    under-counts precisely during the incident.

    `allow_fallback=False` is what produces refusals at all -- with the chain crossing
    providers the outage is rescued and there is nothing to refuse.
    """
    calls = Deployment.build(outages=_outage_on("anthropic"),
                             allow_fallback=False).serve(CASES, LADDER, SPEC)
    refused = [c for c in calls if not c.served]
    assert refused, "with fallback off, an outage on three of five models must refuse"

    assert all(c.quality_ok is None for c in refused), (
        f"{sum(1 for c in refused if c.quality_ok is not None)} refused requests carry "
        "a quality verdict for an answer nobody received")
    # The size of getting that wrong, rather than the fact of it.
    honoured = evaluable(calls).violation_rate
    as_failures = 1.0 - sum(1 for c in calls if c.quality_ok) / len(calls)
    assert as_failures - honoured > 0.25, (
        f"scoring the {len(refused)} refusals as model failures moves the reported "
        f"violation from {honoured:.1%} to {as_failures:.1%}, which is too small a gap "
        "for this run to show the defect")

    # Nothing may be filed against the empty model name a refusal carries.
    assert "" not in attribute(calls), "refused requests reached the attribution tally"

    report = spend_report(calls, label="fallback off, anthropic down")
    assert report.wasted_usd > 0, (
        "every failed attempt was billed at zero, so this run cannot show that a "
        "ledger counting only successful calls under-counts during an incident")
    assert report.spend_usd > report.billed_usd
    assert report.spend_usd == pytest.approx(sum(c.spend_usd for c in calls), abs=1e-12), (
        "the per-call spend accessor and the report disagree about what a run cost")
    assert report.wasted_usd / report.billed_usd > 0.05, (
        f"waste is {report.wasted_usd / report.billed_usd:.1%} of billed spend -- "
        "small enough that dropping it entirely would round to nothing here")


# ---- 3. the shed-load trap -----------------------------------------------
def test_shed_load_cannot_register_as_a_cost_saving(capsys):
    """The headline. A dashboard dividing spend by INTENDED requests reports a
    spectacular saving under a capacity squeeze that is entirely requests never served.

    The correct calculation replays the surviving requests against an unsqueezed
    deployment. Because the fixture is deterministic and the shed requests never reached
    an upstream at all, that replay reproduces the squeezed run's spend exactly -- so
    the genuine per-served saving is not merely small, it is zero.
    """
    audit = audit_shed_load(
        lambda squeezed: Deployment.build(
            team_tokens_per_minute=8_000 if squeezed else 10 ** 9),
        CASES, LADDER, SPEC)
    for line in audit.lines():
        print(line)

    assert audit.squeezed.shed > 0, "the squeeze must actually shed something"
    assert audit.decisions_unchanged, (
        "the squeeze must not change a single routing decision, or the naive number "
        "would be measuring the router rather than the shedding")
    # `decisions_unchanged` alone is weaker than it reads: the model the router ASKS
    # for is a pure function of (policy, case, spec) and cannot move when only the
    # deployment is squeezed. The two quantities that can move are what actually has to
    # be zero for the exactness below to be a measurement rather than a coincidence.
    assert not audit.confounded, (
        f"{audit.served_model_changes} survivors were served by a different model and "
        f"{audit.clamped_under_squeeze} had their answer clamped shorter; either puts a "
        "genuine cost change inside the squeeze, and the matched figure is that change")
    assert audit.naive_saving > 0.5, (
        f"the naive dashboard reports {audit.naive_saving:.1%} on an unchanged router")
    assert audit.matched_saving == pytest.approx(0.0, abs=1e-12), (
        f"serving those requests cost {audit.matched_saving:+.2%} less than before -- "
        "if that is not exactly zero the replay is not matched and the claim that the "
        "whole apparent saving is shed load is unproven")
    assert audit.squeezed.cost_per_served == pytest.approx(
        audit.matched.cost_per_served, abs=1e-12)
    # The exactness is conditional on the squeeze REFUSING rather than clamping, which
    # is a property of this token ceiling and not of the audit: admission's design is to
    # grant a smaller ceiling wherever it can, and at 2_000 OTPM one survivor is clamped
    # 508 -> 450 and the matched saving becomes +0.79% of genuinely shorter answers.
    clamping = Deployment.build(team_tokens_per_minute=2_000).serve(CASES, LADDER, SPEC)
    assert any(c.clamped_from is not None for c in clamping if c.served), (
        "no ceiling in reach of this workload clamps, so the exactness above is "
        "unconditional and this caveat is dead weight")

    out = capsys.readouterr().out
    assert "served" in out and str(audit.squeezed.shed) in out, (
        "the honest report must print the served denominator and the shed count, or "
        "the gap between the two numbers is invisible to the person reading it")


def test_the_naive_saving_is_the_shed_rate_wearing_a_dollar_sign():
    """The diagnosis behind the trap: with routing unchanged and the surviving requests
    costing exactly what they always did, spend/intended is spend/served scaled by the
    fraction served. The dashboard is reporting its own shed rate."""
    audit = audit_shed_load(
        lambda squeezed: Deployment.build(
            team_tokens_per_minute=8_000 if squeezed else 10 ** 9),
        CASES, LADDER, SPEC)
    # Not an identity: it holds only because the shedding was cost-neutral in WHICH
    # requests it dropped. A squeeze that shed the expensive traffic would break it,
    # and that divergence is the thing worth seeing.
    assert audit.naive_saving == pytest.approx(audit.squeezed.shed_rate, abs=0.05), (
        f"naive saving {audit.naive_saving:.1%} vs shed rate "
        f"{audit.squeezed.shed_rate:.1%}: the shedding was not cost-neutral, so the "
        "gap is a real composition effect and must be reported as one")


def test_the_audits_confound_check_fires_when_the_squeeze_moves_the_served_model(capsys):
    """The control above has to be able to come out the other way, or asserting it
    proves nothing.

    A provider-RPM squeeze is the case that breaks it: the router asks for exactly the
    same model on every request -- `decisions_unchanged` is still True and always will
    be -- while the gateway spreads the overflow onto a different provider. The matched
    figure then prices a model-mix change, and reporting it as a saving is this
    module's own defect committed one level up.
    """
    audit = audit_shed_load(
        lambda squeezed: Deployment.build(
            provider_rpm={"openai": 60, "anthropic": 60} if squeezed else None),
        CASES, LADDER, SPEC)
    for line in audit.lines():
        print(line)

    assert audit.decisions_unchanged, (
        "the requested model is a pure function of the policy and the case; if this is "
        "ever False the control was measuring something other than what it claims")
    assert audit.served_model_changes > 0, (
        "this squeeze was supposed to substitute survivors onto the other provider")
    assert audit.confounded and audit.matched_saving > 1e-9, (
        f"a matched saving of {audit.matched_saving:+.2%} over "
        f"{audit.served_model_changes} substituted survivors is the confound this flag "
        "exists to catch; with it at zero the flag is never exercised")

    out = capsys.readouterr().out
    assert "different model" in out, (
        "the printed verdict credited the squeeze with a saving without naming the "
        f"{audit.served_model_changes} substitutions that produced it")


# ---- 4. cost-optimal versus capacity-optimal -----------------------------
def test_with_fallback_on_throughput_is_a_property_of_the_fleet_not_the_policy():
    """Half of the answer to 'do cost and capacity point in opposite directions'.

    With the gateway's chain crossing providers, a router that piles everything onto the
    cheapest provider does not lose throughput -- the chain spreads the overflow for it.
    Served count is the sum of the provider windows either way. That is arithmetic.
    """
    caps = {"openai": 60, "anthropic": 60}
    served = {}
    for policy in (AlwaysCheapest(), _AlternatingProviders()):
        calls = Deployment.build(provider_rpm=caps).serve(CASES, policy, SPEC)
        served[policy.name] = spend_report(calls, label=policy.name)

    for name, report in served.items():
        assert report.served == sum(caps.values()), (
            f"{name} served {report.served}, not the fleet's {sum(caps.values())}")
    assert len({r.served for r in served.values()}) == 1


def test_with_fallback_off_cost_optimal_and_capacity_optimal_rank_oppositely(capsys):
    """The other half, and the tension is real.

    `allow_fallback: false` is a policy a structured-output team actually sets -- a
    different model returns a different shape, so a hard failure is preferable. With it
    set, the gateway can no longer spread the overflow, and the cheapest model's
    provider window becomes the throughput ceiling.
    """
    caps = {"openai": 60, "anthropic": 60}
    reports = {}
    for policy in (AlwaysCheapest(), _AlternatingProviders()):
        calls = Deployment.build(provider_rpm=caps, allow_fallback=False
                                 ).serve(CASES, policy, SPEC)
        reports[policy.name] = spend_report(calls, label=policy.name)
        print(reports[policy.name].line())

    cheap, spread = reports["always_cheapest"], reports["alternating_providers"]
    assert cheap.served == caps["openai"], (
        f"the cost-optimal policy should be capped at its one provider's window "
        f"({caps['openai']}), served {cheap.served}")
    assert spread.served == sum(caps.values())
    assert spread.served > cheap.served, "capacity-optimal must win on throughput"
    assert cheap.cost_per_served < spread.cost_per_served, "cost-optimal must win on cost"

    out = capsys.readouterr().out
    assert "always_cheapest" in out and "alternating_providers" in out


def test_a_provider_concurrency_limit_cannot_bind_in_a_synchronous_driver():
    """The negative result, run rather than asserted in prose.

    The brief's mechanism -- cheap models are small and fast, so under a concurrency
    limit the throughput-optimal choice differs from the cost-optimal one -- is not
    expressible here. `ModelSpec` carries no latency or throughput field; latency lives
    on the provider fixture. And the one concurrency dimension that exists degenerates:
    the router calls `acquire()` immediately before `call()`, so in a synchronous driver
    exactly one request is ever in flight. The limit is off or total, never binding.
    """
    outcomes = {}
    for limit in (1, 2, 8):
        deployment = Deployment.build(provider_concurrency=limit)
        calls = deployment.serve(CASES[:20], AlwaysCheapest(), SPEC)
        outcomes[limit] = sum(1 for c in calls if c.served)

    assert outcomes[1] == 0, "at a limit of one, the in-flight request refuses itself"
    assert outcomes[2] == 20 and outcomes[8] == 20, (
        "above one, a synchronous driver never reaches the limit at all")

    # The team's concurrency limit degenerates the OTHER way, and the difference is one
    # line of ordering: `acquire()` runs before the provider's check, so `_inflight` is
    # already 1 there, while `admit` tests the team's counter before incrementing it and
    # settlement returns it to zero before the next request. A limit of one is therefore
    # inert rather than total. Both knobs are on `Deployment.build`; neither can bind,
    # and a knob that cannot bind is a capability claim the code does not have.
    team = {k: sum(1 for c in Deployment.build(team_concurrency=k)
                   .serve(CASES[:20], AlwaysCheapest(), SPEC) if c.served)
            for k in (1, 2, 8)}
    assert set(team.values()) == {20}, (
        f"team concurrency bound at {team}; if it can bind, the synchronous-driver "
        "result above is no longer the whole story and this test is out of date")

    # The one capacity dimension on this seam that DOES bind, asserted so that the
    # negative results above are a statement about concurrency rather than about the
    # whole of `Deployment.build` being wired to nothing.
    tpm = {t: sum(1 for c in Deployment.build(provider_tpm=t)
                  .serve(CASES[:200], AlwaysCheapest(), SPEC) if c.served)
           for t in (5_000, 50_000, 10 ** 9)}
    assert tpm[5_000] < tpm[50_000] < tpm[10 ** 9] == 200, (
        f"the provider token window did not throttle: {tpm}")

    # And the other half of why the brief's mechanism is not expressible: neither side
    # of the seam has anywhere to put a model's speed. Both canaries, so that the day
    # one appears this negative result is revisited rather than quietly left standing.
    speed = {"tokens_per_second", "latency_s", "throughput", "speed"}
    assert not speed & set(ModelSpec.model_fields), (
        f"llm_gateway.ModelSpec grew {speed & set(ModelSpec.model_fields)}; the "
        "cost-versus-capacity tension may now be demonstrable through model speed")
    assert not any(speed & set(vars(m)) for m in REGISTRY), (
        "route-regret's registry grew a speed field; same revisit applies")


# ---- 5. what the gateway's fixture can and cannot claim ------------------
def test_the_gateway_fixtures_text_cannot_carry_the_capability_claim():
    """FixtureUpstream returns `[provider/model] response` -- a pure function of the
    model name. Two cases the same model succeeds on and fails on come back byte
    identical, so quality ground truth has to stay with route_regret.models.succeeds.

    Not a defect in llm-gateway. Its claims are about capacity and the ledger, where the
    response text is correctly irrelevant.
    """
    calls = Deployment.build().serve(CASES, AlwaysCheapest(), SPEC)
    by_model: dict[str, dict[bool, str]] = {}
    for call in calls:
        by_model.setdefault(call.model_served, {}).setdefault(call.quality_ok,
                                                              call.response_text)
    disagreeing = {m: t for m, t in by_model.items() if len(t) == 2}
    assert disagreeing, "need a model that both succeeds and fails to make the point"
    for model, texts in disagreeing.items():
        assert texts[True] == texts[False], (
            f"{model}: the gateway fixture's text distinguishes a success from a "
            "failure, which would make it a quality oracle it does not claim to be")


def test_two_runs_of_the_seam_agree_to_the_cent():
    """House rule. The gateway brings a clock and a mutable per-provider call counter
    into a repo that had neither; if either leaked, this is where it shows."""
    first = Deployment.build(outages=_outage_on("anthropic")).serve(CASES, LADDER, SPEC)
    second = Deployment.build(outages=_outage_on("anthropic")).serve(CASES, LADDER, SPEC)
    assert spend_report(first).spend_usd == spend_report(second).spend_usd
    assert [c.model_served for c in first] == [c.model_served for c in second]
    assert [c.quality_ok for c in first] == [c.quality_ok for c in second]


# ---- confidence is consumed, not decorative ------------------------------
def test_confidence_truncates_the_ranked_list_and_changes_what_the_gateway_does():
    """A declared field nothing consumes is a claim of capability the code does not
    have -- llm-gateway deleted `Request.stream` for exactly that. Confidence here
    decides whether the router permits substitution BELOW the bar it set, so it has to
    change outcomes or it should not be in the dataclass."""
    # The floor is a declared policy knob, so the test derives one from the traffic
    # rather than picking a number: the median observed confidence, which treats half
    # the workload as confident. Two earlier versions of this line failed usefully --
    # a floor of 1.0 is unreachable against cuts 0.25 apart, and a floor near the
    # MAXIMUM selects only the easiest requests, which route to the cheapest model and
    # whose ranking is entirely adequate already, so truncation is a no-op there.
    floor = median(_confidence_of(c) for c in CASES)
    trusting = Deployment.build(confidence_floor=0.0, outages=_outage_on("anthropic"))
    strict = Deployment.build(confidence_floor=floor, outages=_outage_on("anthropic"))

    full = [len(decide(LADDER, c, SPEC, confidence_floor=0.0).preferred_models)
            for c in CASES]
    truncated = [len(decide(LADDER, c, SPEC, confidence_floor=floor).preferred_models)
                 for c in CASES]
    assert sum(truncated) < sum(full), (
        f"a floor at the median observed confidence ({floor:.3f}) shortened no ranking")

    # "Shorter rankings appeared" is satisfied by a confidence that carries no
    # information at all: a `_confidence` returning one constant makes the median that
    # constant, clears the floor on every request, and truncates the lot. What has to
    # hold is that confidence SELECTS -- a strict subset of the requests that had
    # something to truncate, since a median splits the traffic in half by construction.
    shortened = sum(1 for t, f in zip(truncated, full) if t < f)
    truncatable = sum(1 for f, c in zip(full, CASES)
                      if f > len(_adequate_prefix(LADDER.choose(c, SPEC).name)))
    assert 0 < shortened < truncatable, (
        f"{shortened} of the {truncatable} requests with an inadequate tail were "
        f"truncated -- at 0 the floor does nothing, at {truncatable} it fires on every "
        "request and the confidence it is thresholding is a constant")

    # And WHERE it cuts. The claim is that a confident router refuses everything below
    # the bar it set; a truncation that merely shortens the tail still lets the gateway
    # serve an inadequate model, which is the outcome the floor exists to prevent.
    for case, was_full, now in zip(CASES, full, truncated):
        if now < was_full:
            ranked = decide(LADDER, case, SPEC, confidence_floor=floor).preferred_models
            assert list(ranked) == _adequate_prefix(LADDER.choose(case, SPEC).name), (
                f"{case.case_id}: truncated ranking {ranked} still permits a model "
                "below the bar the router set")

    permissive = spend_report(trusting.serve(CASES, LADDER, SPEC))
    refusing = spend_report(strict.serve(CASES, LADDER, SPEC))
    assert refusing.served < permissive.served, (
        "refusing inadequate substitutes must cost availability, or the truncation is "
        "doing nothing and confidence is decorative")


# ---- helpers -------------------------------------------------------------
_BY_ID = {c.case_id: c for c in CASES}


def _case_of(call):
    return _BY_ID[call.case_id]


def _confidence_of(case):
    return _confidence(LADDER, case, SPEC)[0]


def _adequate_prefix(head):
    """The ranking a fully confident router would permit: the head and every substitute
    at or above its capability. What is left over is what a floor can take away."""
    bar = BY_NAME[head].capability["default"]
    return [m for m in rank_from(head) if BY_NAME[m].capability["default"] >= bar]


class _AlternatingProviders:
    """A control that spends more per request in exchange for using both providers'
    rate windows. Deliberately not clever: the point is the provider partition, and a
    policy that reasoned about content would confound it."""

    name = "alternating_providers"

    def choose(self, case, spec):
        return BY_NAME["gpt-4o-mini" if int(case.case_id.rsplit(":", 1)[1]) % 2 == 0
                       else "haiku-4.5"]

    def propensity(self, case, spec, model) -> float:
        return 1.0
