"""The classifier as ONE ENTRANT, and the four ways its usual write-up lies.

Every test here corresponds to a defect in the spec this repo replaces, and each one is
a claim that can come back the wrong way:

1. it hand-labelled complexity tiers, so the labels were a function of the features and
   the classifier recovered the labeller instead of any model's behaviour;
2. it reported a bare accuracy scalar against an unstated majority class;
3. it scored the two routing errors as if they cost the same, when one costs quality and
   the other costs money and is invisible to a verifier;
4. its retraining loop only ever labelled the arm it acted on, so it converged to
   always-frontier.

Two of these came back partly against the entrant on measurement, and both are left
asserting the direction measured rather than the direction claimed:
`test_reading_this_requests_own_price_is_the_part_that_does_not_work` refutes the stated
mechanism for defect 3's margin while the margin itself survives, and
`test_the_margin_is_mostly_features_but_the_fitting_is_not_nothing` reports a sign flip
that is a property of EXPLORATION_RATE rather than of what the router learned.
"""

from dataclasses import replace

import numpy as np
import pytest

from route_regret.bench import references, run
from route_regret.classifier import (EXPLORATION_RATE, FEATURE_ORDER,
                                     CostSensitiveRouter, Faceoff, FlatThresholdRouter,
                                     SurfaceLabeller, adequacy, faceoff, fit_head,
                                     quality_gate, report_adequacy, train)
from route_regret.fixture import REGISTRY, WorkloadSpec, build_workload
from route_regret.metrics import Reference, score
from route_regret.models import succeeds
from route_regret.policies import AlwaysTop, ThresholdLadder
from route_regret.report import DEFAULT_LADDER, tune_to_tau

N_TEST = 3000
N_TRAIN = 4000
DELTA = 0.03
VERIFY = 0.05


@pytest.fixture(scope="module")
def spec():
    return WorkloadSpec(mix="balanced", n=N_TEST, seed_tag="v1")


@pytest.fixture(scope="module")
def cases(spec):
    return build_workload(spec)


@pytest.fixture(scope="module")
def ensemble(spec, cases):
    ens = train(spec, n=N_TRAIN)
    ens.prime(cases, spec)
    return ens


@pytest.fixture(scope="module")
def tuned(spec, cases, ensemble):
    """Every entrant pinned to the same delivered quality before anything is compared."""
    ref = references(cases, spec)
    out = {"ref": ref}
    out["classifier"] = tune_to_tau(lambda o: CostSensitiveRouter(ensemble, offset=o),
                                    cases, spec, ref, DELTA, VERIFY)
    out["flat"] = tune_to_tau(lambda o: FlatThresholdRouter(ensemble, offset=o),
                              cases, spec, ref, DELTA, VERIFY)
    out["ladder"] = tune_to_tau(lambda o: ThresholdLadder(DEFAULT_LADDER, offset=o),
                                cases, spec, ref, DELTA, VERIFY)
    out["greedy_only"] = tune_to_tau(
        lambda o: CostSensitiveRouter(ensemble, offset=o, epsilon=0.0), cases, spec, ref,
        DELTA, VERIFY)
    return out


# Four arbitrary exploration streams, generated rather than chosen. The realisation is
# worth a couple of FASC points, so a margin measured on one of them is a statement about
# those draws; every claim below that could plausibly sit inside that band is made across
# all four.
EXPLORE_TAGS = tuple(f"draw-{i}" for i in range(4))


# --------------------------------------------------------------------------------------
# 1. The label
# --------------------------------------------------------------------------------------

def test_a_label_read_off_the_prompt_scores_far_higher_and_knows_nothing(spec, cases):
    """Hand-labelled complexity is a re-encoding of the classifier's own input.

    A person labelling "how hard is this prompt" has only the prompt surface to go on --
    exactly the features the classifier reads -- so the label is a function of the
    features and the classifier recovers the labeller's rule. It scores beautifully and
    carries no information about any model, which this test states twice: once as the
    accuracy gap, and once as the fact that REPLACING EVERY MODEL IN THE REGISTRY leaves
    the surface labels and their score bit-identical.
    """
    train_spec = replace(spec, seed_tag="train", n=N_TRAIN)
    train_cases = build_workload(train_spec)
    cheap = min(REGISTRY, key=lambda m: m.cost(1000, 500))

    labeller = SurfaceLabeller(train_cases, train_spec)
    surface_fit = fit_head(train_cases, train_spec, labeller(train_cases, train_spec))
    adequacy_fit = fit_head(train_cases, train_spec, adequacy(cheap, train_cases))

    surface = report_adequacy(labeller(cases, spec), surface_fit.predict(cases, spec),
                              model="surface complexity label")
    real = report_adequacy(adequacy(cheap, cases), adequacy_fit.predict(cases, spec),
                           model=cheap.name)

    assert surface.accuracy > real.accuracy + 0.08, (
        f"the circular label should be far easier to hit: {surface} vs {real}")
    assert (1 - surface.accuracy) < (1 - real.accuracy) / 4, (
        f"and its error rate should collapse: {1 - surface.accuracy:.1%} vs "
        f"{1 - real.accuracy:.1%}")

    # Now make routing a different problem: a registry where the cheap model is nearly as
    # good as the frontier. Adequacy moves, and the whole validation moves with it. The
    # surface pipeline cannot move, because `SurfaceLabeller` takes no registry argument
    # at all -- which is not an omission to be fixed but the defect itself, stated as a
    # signature. The falsifiable half of the claim is on the adequacy side: if ITS score
    # did not move either, both labels would be equally inert and this test would be
    # asserting a distinction that is not there.
    other_cheap = cheap.model_copy(update={"capability": {"default": 0.93}})
    moved = float(np.mean(adequacy(cheap, cases) != adequacy(other_cheap, cases)))
    assert moved > 0.3, f"the registry swap must actually change adequacy, moved {moved:.1%}"

    stale = report_adequacy(adequacy(other_cheap, cases), adequacy_fit.predict(cases, spec),
                            model=f"{cheap.name} fit carried onto the new registry")
    assert stale.accuracy < stale.majority_baseline, (
        "the adequacy-trained model, carried onto a registry where the cheap model is "
        "strong, must score BELOW always-guessing -- its 88% was a claim about a price "
        f"list, not about prompts: {stale}")

    # How far below depends on how far the registry moved, and the boundary is measured
    # here rather than left implied by the 0.93 above. At capability 0.70 the labels still
    # move on a THIRD of cases -- the guard above passes -- and the stale fit is back over
    # its majority baseline. So "below always-guessing" is a witness at a stated swap and
    # not a law: what is a law is that the fit tracks the registry, and the pair of
    # measurements is what says which of the two this test is entitled to claim.
    milder = cheap.model_copy(update={"capability": {"default": 0.70}})
    mild_moved = float(np.mean(adequacy(cheap, cases) != adequacy(milder, cases)))
    mild = report_adequacy(adequacy(milder, cases), adequacy_fit.predict(cases, spec),
                           model=f"{cheap.name} fit under a milder swap")
    assert mild_moved > 0.3, (
        "the milder swap has to clear the same setup guard, or it is not the same "
        f"experiment with a smaller knob: moved {mild_moved:.1%}")
    assert mild.majority_baseline < stale.majority_baseline, (
        "a smaller swap must shift the class balance less, or the two swaps are not on "
        f"one axis: {mild.majority_baseline:.1%} against {stale.majority_baseline:.1%}")
    assert mild.accuracy > mild.majority_baseline, (
        "and at the milder swap the stale fit is back ABOVE always-guessing, which is "
        f"what makes the sentence above a witness rather than a law: {mild}")
    assert "registry" not in SurfaceLabeller.__init__.__annotations__, (
        "the hand-labelled pipeline reports the same score whatever the models do, and "
        "the reason is visible in its signature: there is nowhere for a model to enter")

    # Two properties of the labeller the demonstration leans on and neither the accuracy
    # gap nor the registry swap can see, because both are invariant to them.
    #
    # It must point the same way as `adequacy` -- 1 means "the cheap model can have this"
    # -- or the 99.4% above is agreement with an inverted label and the two reports are
    # not comparable. And its cut must be the one standard it settled on, not a median
    # recomputed per batch: a labeller who re-centres on every workload calls half of ANY
    # traffic simple, which is a different and much weaker defect than the one being shown.
    assert np.corrcoef(labeller(cases, spec), adequacy(cheap, cases))[0, 1] > 0.2, (
        "the surface label must be oriented like adequacy, or the reports above compare "
        "a label against its own negation")
    harder = WorkloadSpec(mix="mostly_hard", n=600, seed_tag="v1")
    share = float(labeller(build_workload(harder), harder).mean())
    assert share < 0.35, (
        f"the labeller re-centred on new traffic and called {share:.0%} of a mostly_hard "
        "workload simple; a fixed standard has to notice the traffic got harder")


def test_training_on_the_workload_you_score_on_is_refused(spec):
    """Same seed_tag means the same case ids, and accuracy becomes memorisation."""
    with pytest.raises(ValueError):
        train(spec, n=200, seed_tag=spec.seed_tag)


def test_the_fitting_apparatus_reads_the_arguments_it_is_given(spec, cases):
    """The three ways this module can quietly stop depending on its own inputs.

    None of them changes a headline enough for the comparisons above to notice -- a
    smaller training set, a stale cached probability and a swallowed shape error all
    produce numbers that look exactly like numbers. They are asserted here because the
    only alternative is to find out from a figure that was never a measurement.
    """
    # `n` must be the training workload's size. Ignoring it silently trains on whatever
    # the evaluation spec asked for, which is the sample-size axis disappearing.
    cheap = min(REGISTRY, key=lambda m: m.cost(1000, 500))
    small = train(spec, n=400).heads[cheap.name].probability(cases, spec)
    large = train(spec, n=N_TRAIN).heads[cheap.name].probability(cases, spec)
    assert np.abs(small - large).max() > 0.01, (
        "training at n=400 and at n=4000 produced the same fit, so `n` is not reaching "
        "the workload builder")

    # The probability cache is keyed on the spec as well as the case id, because case ids
    # repeat across specs by construction (`seed_tag:mix:index`) while the features under
    # them do not. A cache keyed on the id alone hands a signal=0.70 probability to a
    # signal=0.20 workload and the sweep flattens for a reason nobody can see.
    faint = replace(spec, signal=0.20)
    faint_cases = build_workload(faint)
    ens = train(spec, n=N_TRAIN)
    ens.prime(cases, spec)
    ens.prime(faint_cases, faint)
    same_id = next(c for c in faint_cases if c.case_id == cases[0].case_id)
    assert (ens.probabilities(cases[0], spec)[cheap.name]
            != ens.probabilities(same_id, faint)[cheap.name]), (
        "the same case id under two specs returned one cached probability")

    # And the report refuses a mismatched pair rather than broadcasting it into a number.
    # The shapes here are chosen to BROADCAST: (10,) against (10, 1) is a legal numpy
    # comparison that yields a 10x10 matrix and an accuracy of 100%. A ragged pair like
    # (10,) against (9,) proves nothing, because numpy raises ValueError on it with or
    # without the guard -- the first version of this assertion did exactly that and could
    # not fail.
    with pytest.raises(ValueError):
        report_adequacy(np.zeros(10, dtype=int), np.zeros((10, 1), dtype=int),
                        model="a column vector wearing a row vector's shape")


# --------------------------------------------------------------------------------------
# 2. The report
# --------------------------------------------------------------------------------------

def test_no_accuracy_is_reported_without_its_kappa_and_its_majority_class(spec, cases,
                                                                          ensemble):
    """80% against a 55% majority class is kappa 0.55, not "80% good".

    The report renders as one string containing all four quantities, so there is no way
    to quote the accuracy from it without carrying the baseline that deflates it.
    """
    cheap = min(REGISTRY, key=lambda m: m.cost(1000, 500))
    rep = ensemble.report(cheap, cases, spec)
    text = str(rep)

    for piece in (f"{rep.accuracy:.1%}", f"{rep.majority_baseline:.1%}",
                  f"{rep.kappa:.3f}", str(rep.under_route), str(rep.over_route)):
        assert piece in text, f"{piece!r} missing from the rendered report: {text}"

    assert rep.majority_baseline > 0.5, "a majority class below half is not a baseline"
    assert rep.kappa < rep.accuracy - 0.05, (
        f"kappa must sit well below raw accuracy when a majority class is doing work: {rep}")

    # The degenerate predictor that scores the majority baseline and has learned nothing.
    truth = adequacy(cheap, cases)
    majority = np.full(len(cases), int(truth.mean() >= 0.5))
    dumb = report_adequacy(truth, majority, model="always-say-the-majority")
    assert dumb.accuracy == pytest.approx(dumb.majority_baseline), (
        "the constant predictor scores exactly the majority baseline")
    assert dumb.kappa == pytest.approx(0.0, abs=1e-12), (
        f"and kappa must call that zero agreement above chance: {dumb}")

    # The off-diagonal cells are the reason the interval type matters, and the reason it
    # is imported rather than written here. Feed the report a cell it has seen nothing in:
    # Wald's width is p-hat(1-p-hat) and collapses to ZERO there, reporting perfect
    # certainty from the least informative data available. Wilson keeps a width. Nothing
    # else in this file distinguishes them -- both intervals render, and the render check
    # above compares the report against itself.
    flawless = np.array([1] * 40 + [0] * 40)
    clean = report_adequacy(flawless, flawless, model="never-wrong-on-40-and-40")
    assert clean.under_route.point == 0.0 and clean.over_route.point == 0.0
    for cell in (clean.under_route, clean.over_route):
        assert cell.high > 0.0, (
            f"a zero-count cell reported zero width: {cell} -- that is Wald, and at "
            "n=40 it claims the next 40 cases cannot possibly differ")
        assert cell.high < 0.5, f"and it must still be an interval, not a shrug: {cell}"

    # The two figures do not merely differ in level -- they can move in OPPOSITE
    # directions on the same problem. Refit the same pipeline for a registry where the
    # cheap model is strong: the majority class swells, accuracy goes UP, and agreement
    # above chance goes DOWN. Anyone quoting the accuracy alone reports an improvement
    # that is a worse classifier.
    train_spec = replace(spec, seed_tag="train", n=N_TRAIN)
    train_cases = build_workload(train_spec)
    strong = cheap.model_copy(update={"capability": {"default": 0.93}})
    strong_fit = fit_head(train_cases, train_spec, adequacy(strong, train_cases))
    strong_rep = report_adequacy(adequacy(strong, cases), strong_fit.predict(cases, spec),
                                 model=f"{strong.name} on a registry where it is strong")
    assert strong_rep.accuracy > rep.accuracy and strong_rep.kappa < rep.kappa, (
        f"accuracy and kappa must be able to disagree in sign, or the whole objection to "
        f"the bare scalar is theoretical: {rep} versus {strong_rep}")


# --------------------------------------------------------------------------------------
# 3. The asymmetry
# --------------------------------------------------------------------------------------

def test_one_classifier_two_losses_same_accuracy_materially_different_money(spec, cases,
                                                                            ensemble, tuned):
    """The two errors do not cost the same, and accuracy cannot see the difference.

    Both routers below read the SAME fitted probabilities, so their adequacy accuracy is
    identical by construction rather than by a lucky match. One prices the two errors --
    under-routing costs a violation, over-routing costs the price gap on THIS case -- and
    the other applies a flat probability cut, which is what treating the classifier's
    output as a class label gets you. Both are then tuned to the same delivered quality,
    so the gap between them is money and nothing else.
    """
    ref = tuned["ref"]
    cost_pol, cost_led = tuned["classifier"]
    flat_pol, flat_led = tuned["flat"]

    assert cost_pol.ensemble is flat_pol.ensemble, "the demonstration needs one fitted model"
    cs, fs = score(cost_led, ref, delta=DELTA), score(flat_led, ref, delta=DELTA)
    assert cs.fasc_at_delta is not None and fs.fasc_at_delta is not None
    assert cost_led.violation_rate() == pytest.approx(flat_led.violation_rate(), abs=2e-3), (
        "the comparison is only meaningful at matched delivered quality")

    duel = faceoff(cost_led, flat_led, cases, ref, challenger="cost_sensitive",
                   incumbent="flat_threshold")
    assert duel.low > 0, (
        f"pricing the two errors differently should be worth real money: {duel.verdict()}")
    assert duel.point > 0.02, (
        f"and materially so, not a rounding difference: {duel.verdict()}")


class _FlatPriceRouter(CostSensitiveRouter):
    """The same Lagrangian with the per-case half of the price term removed: it prices
    each tier at the ladder's flat unit rate and never reads this request's token counts.
    Only the price term changes -- same fit, same lambda scale, same exploration."""

    def greedy(self, case, spec):
        p = self.ensemble.probabilities(case, spec)
        best, best_loss = None, None
        for m in self.tiers:
            loss = m.cost(1000, 500) + self.lam * (1.0 - p[m.name])
            if best_loss is None or loss < best_loss - 1e-15:
                best, best_loss = m, loss
        return best


def test_reading_this_requests_own_price_is_the_part_that_does_not_work(spec, cases,
                                                                        ensemble, tuned):
    """The stated mechanism for the margin above, measured, and it comes back NEGATIVE.

    The claim was that the decision is whether the price gap on THIS request is worth its
    incremental risk. The gaps are genuinely spread -- 8.7x across this workload, asserted
    below so the null is not "there was nothing to condition on". Conditioning on them
    still loses money: hold the fit, the knob and the exploration fixed, swap the per-case
    price for the ladder's flat unit price, and the router gets cheaper at the same
    quality bar.

    The reading is the repo's own coupling pointed back at the router. Length leaks
    difficulty, so a long request simultaneously has a wider price gap (which the price
    term reads as "route it down") and a higher P(inadequate) (which the risk term reads
    as "route it up"), and the two arrive on the same cases with the price term winning
    where it is most wrong. So the +7.5 is earned by the price LADDER being in the loss at
    all -- tiers priced against risk rather than thresholded -- and not by per-case
    pricing, which is a separate claim that this test refuses.

    Left red-side-up rather than fixed: changing the policy would move every figure the
    module reports, and the finding is the point.
    """
    ref = tuned["ref"]
    _, flat_cut = tuned["flat"]
    _, per_case = tuned["classifier"]
    _, flat_price = tune_to_tau(
        lambda o: _FlatPriceRouter(ensemble, offset=o, label="flat_unit_price"),
        cases, spec, ref, DELTA, VERIFY)

    top = max(REGISTRY, key=lambda m: m.capability["default"])
    cheap = min(REGISTRY, key=lambda m: m.cost(1000, 500))
    gaps = [top.cost(c.input_tokens, c.output_tokens)
            - cheap.cost(c.input_tokens, c.output_tokens) for c in cases]
    assert max(gaps) / min(gaps) > 5, (
        f"if the per-case price gaps barely varied, conditioning on them could not help "
        f"and this test would prove nothing: spread {max(gaps)/min(gaps):.1f}x")

    kept = faceoff(flat_price, flat_cut, cases, ref, challenger="flat_unit_price",
                   incumbent="flat_threshold")
    dropped = faceoff(per_case, flat_cut, cases, ref, challenger="per_case_price",
                      incumbent="flat_threshold")
    assert kept.low > 0, (
        f"the price ladder alone must already beat the flat cut, or the +7.5 has no "
        f"component this test can attribute: {kept.verdict()}")
    assert kept.point > dropped.point, (
        "reading this request's own price must not be what earns the margin, and it is "
        f"not: {kept.point:+.1%} without it against {dropped.point:+.1%} with it")

    head_to_head = faceoff(per_case, flat_price, cases, ref, challenger="per_case_price",
                           incumbent="flat_unit_price")
    assert head_to_head.point < 0, (
        f"and directly, the per-case term is worth negative money: "
        f"{head_to_head.verdict()}")


# --------------------------------------------------------------------------------------
# 4. Exploration and propensities
# --------------------------------------------------------------------------------------

def test_exploration_goes_both_ways_and_every_chosen_arm_has_a_positive_propensity(
        spec, cases, tuned):
    """The spec's retraining loop only ever labelled the arm it acted on, so it had no
    counterfactual for any cheaper tier and converged to always-frontier. Phase 4 cannot
    reweight a logged run whose chosen action had propensity zero -- the estimator
    divides by it."""
    policy, ledger = tuned["classifier"]

    down = up = greedy = 0
    for case in cases:
        chosen = policy.choose(case, spec)
        g = policy.greedy(case, spec)
        ci, gi = policy.tiers.index(chosen), policy.tiers.index(g)
        down += ci < gi
        up += ci > gi
        greedy += ci == gi
        assert policy.propensity(case, spec, chosen) > 0, (
            f"chosen arm {chosen.name} logged with propensity zero on {case.case_id}")
        total = sum(policy.propensity(case, spec, m) for m in REGISTRY)
        assert total == pytest.approx(1.0, abs=1e-12), f"propensities sum to {total}"

    assert down > 0 and up > 0, (
        f"exploration must go both ways -- down {down}, up {up} of {len(cases)}")
    assert greedy < len(cases), "a policy that never deviates logs no counterfactual"
    assert 0 < (down + up) / len(cases) < 2 * EXPLORATION_RATE

    logged = [r.propensity for r in ledger.rows if r.call_kind == "route"]
    assert min(logged) > 0, "the ledger must never carry a zero-propensity routed row"


def test_the_verdict_sentence_is_read_off_the_interval_and_not_off_the_point(spec, cases,
                                                                             tuned):
    """The narration itself, exercised on all three outcomes it can render.

    Every other test in this file reads `.low` and `.point` and never the sentence, so
    until this existed the entire `verdict()` derivation was unexecuted logic: swapping
    the beats/LOSES branches left all eleven tests green while the module printed a defeat
    as a win. That is the exact failure the class docstring says it exists to prevent, and
    a docstring is not a test. Constructed by hand rather than measured, because the point
    is the mapping from interval to sentence -- a measured Faceoff can only ever exhibit
    whichever branch the data happened to land in.
    """
    ref = tuned["ref"]
    fields = dict(challenger="A", incumbent="B", challenger_fasc=0.5, incumbent_fasc=0.4,
                  n=len(cases))
    beats = Faceoff(point=+0.10, low=+0.05, high=+0.15, **fields).verdict()
    loses = Faceoff(point=-0.10, low=-0.15, high=-0.05, **fields).verdict()
    tie = Faceoff(point=+0.01, low=-0.05, high=+0.07, **fields).verdict()

    assert "A beats B" in beats and "LOSES" not in beats, beats
    assert "A LOSES to B" in loses, loses
    assert "cannot be told apart" in tie and "beats" not in tie, tie
    assert len({beats, loses, tie}) == 3, "three intervals, three different sentences"

    # A missing FASC is not a small margin: an inadmissible policy has no cost anything
    # may be compared against, and the sentence must refuse rather than report the gap.
    for absent in ("challenger_fasc", "incumbent_fasc"):
        said = Faceoff(point=+0.10, low=+0.05, high=+0.15,
                       **{**fields, absent: None}).verdict()
        assert said.startswith("no comparison"), said
        assert "beats" not in said, said

    # And the number the sentence carries is FASC@delta arithmetic, not a dollar mean
    # wearing a percent sign. FASC is affine in total cost over a shared denominator, so
    # the paired difference must reproduce the two scores `metrics.score` computes
    # separately -- which pins both the achievable-savings denominator and the fact that
    # `_cost_per_case` charges verification and judging, not just the routed call.
    duel = faceoff(tuned["classifier"][1], tuned["ladder"][1], cases, ref,
                   challenger="classifier", incumbent="threshold_ladder")
    assert duel.point == pytest.approx(duel.challenger_fasc - duel.incumbent_fasc,
                                       abs=1e-9), (
        f"the interval is on a different quantity from the table: {duel.point:+.4%} "
        f"against {duel.challenger_fasc - duel.incumbent_fasc:+.4%}")

    # On a workload with nothing to save, that denominator is zero or negative and every
    # margin above becomes a division nobody would publish. It has to raise rather than
    # return an enormous percentage, which is what a near-zero denominator manufactures.
    barren = Reference(top_cost=ref.top_cost, top_violation=ref.top_violation,
                       oracle_cost=ref.top_cost)
    with pytest.raises(ValueError):
        faceoff(tuned["classifier"][1], tuned["ladder"][1], cases, barren,
                challenger="classifier", incumbent="threshold_ladder")


def test_the_coverage_is_paid_for_in_savings_not_donated(spec, cases, tuned):
    """What exploration costs, so it is a priced decision rather than a free virtue.

    The same router with exploration switched off is strictly cheaper at the same quality
    bar: deviating downward has to be bought back at the constraint, and deviating upward
    is money spent on a tier the model did not want. If this ever came out flat,
    exploration would be decorative and the propensities it justifies would be too.
    """
    ref = tuned["ref"]
    _, exploring = tuned["classifier"]
    _, greedy_only = tuned["greedy_only"]
    duel = faceoff(greedy_only, exploring, cases, ref, challenger="greedy_only",
                   incumbent="classifier")
    assert duel.low > 0, (
        f"exploration should have a visible price, and this is it: {duel.verdict()}")


# --------------------------------------------------------------------------------------
# 5. It is an entrant like any other
# --------------------------------------------------------------------------------------

def test_the_router_tunes_to_the_quality_bar_and_its_knob_spans_both_ends(spec, cases,
                                                                          ensemble, tuned):
    ref = tuned["ref"]
    policy, ledger = tuned["classifier"]
    tau = ref.top_violation + DELTA
    assert ledger.violation_rate() <= tau + 1e-12, "the tuned router must be admissible"
    assert score(ledger, ref, delta=DELTA).fasc_at_delta is not None

    # A knob that cannot reach both ends cannot be tuned: bisection would return whichever
    # end it started at and the "matched quality" protocol would be comparing whatever
    # operating point this policy's author happened to pick, which is the thing the
    # protocol exists to stop.
    top = max(REGISTRY, key=lambda m: m.capability["default"])
    cheapest = min(REGISTRY, key=lambda m: m.cost(1000, 500))

    aggressive = CostSensitiveRouter(ensemble, offset=-0.6)
    assert {aggressive.greedy(c, spec).name for c in cases} == {cheapest.name}, (
        "at lambda near zero only price is left in the loss, so the aggressive end must "
        "collapse onto always-cheapest")

    # The conservative end is NOT always-frontier, and that is the loss working rather
    # than failing: where both heads saturate -- an easy extraction case the smallest
    # model is as certain to serve as the largest -- the risk term cancels and price
    # decides. What the end must do is reach the constraint, or nothing is tunable.
    conservative = CostSensitiveRouter(ensemble, offset=0.6)
    conservative_led = run(conservative, cases, spec, verify_rate=VERIFY)
    assert conservative_led.violation_rate() <= tau, (
        f"the conservative end must be admissible, got "
        f"{conservative_led.violation_rate():.1%} against tau {tau:.1%}")

    def frontier_share(p):
        return np.mean([p.greedy(c, spec).name == top.name for c in cases])

    assert frontier_share(conservative) > frontier_share(policy) > frontier_share(aggressive), (
        "the knob must move the frontier share monotonically, or bisection is walking a "
        "surface with no order on it")
    assert -0.6 < policy.offset < 0.6, (
        f"the tuned knob landed on a bisection boundary at {policy.offset:+.3f}, which "
        "means the constraint was never actually located")


def test_two_runs_of_the_trained_router_agree_to_the_cent(spec, cases, tuned):
    """The classifier is the one entrant that could smuggle in RNG state. sklearn's lbfgs
    is deterministic, and this is the test that keeps it that way."""
    policy, ledger = tuned["classifier"]
    again = train(spec, n=N_TRAIN)
    again.prime(cases, spec)
    replayed = run(CostSensitiveRouter(again, offset=policy.offset), cases, spec,
                   verify_rate=VERIFY)
    assert replayed.total_cost() == ledger.total_cost()
    assert replayed.violation_rate() == ledger.violation_rate()


# --------------------------------------------------------------------------------------
# 6. The falsification
# --------------------------------------------------------------------------------------

def test_the_margin_is_mostly_features_but_the_fitting_is_not_nothing(spec, cases,
                                                                      ensemble, tuned):
    """Where the margin over two hand cuts comes from, with the confound taken out.

    ThresholdLadder is two hand cuts over one feature and needs no training set, so the
    ablation that matters is the classifier restricted to that same single feature. The
    first version of this test ran that ablation with exploration ON, found the margin's
    sign flipping between arbitrary exploration streams, and concluded that at matched
    information the fitting bought nothing. That comparison was not matched: the ladder
    does not explore, so only the trained arm was paying the coverage tax, and the tax on
    that arm is 7.8 FASC points against a margin of 6.0.

    Turn exploration off on both trained arms -- now all three entrants are greedy, like
    the ladder -- and the signal-only router beats the ladder by +6.0 points with the
    interval clear of zero. The fitting bought something real; it is simply worth less
    than the coverage the shipped policy spends. Most of the headline is still the four
    features the hand rule was never given, and that is asserted as a ratio rather than
    told as a story.

    The sign flip is left in the test, because it is real and it is the phase-4 warning:
    at the shipped EXPLORATION_RATE two draws of the SAME fitted policy return opposite
    verdicts, each with an interval excluding zero. A single stream's bootstrap is not the
    uncertainty of an exploring policy. What is NOT claimed any more is that the flip
    measures the fitting -- move EXPLORATION_RATE to 0.05 and every stream turns positive,
    to 0.20 and every stream turns negative, which makes the sign a property of a constant
    somebody chose rather than of what the router learned.
    """
    ref = tuned["ref"]
    _, lad_led = tuned["ladder"]
    blinkered = train(spec, n=N_TRAIN, reads=("signal",))
    blinkered.prime(cases, spec)

    assert len(set(FEATURE_ORDER) - set(blinkered.reads)) == 4, (
        "the ablation's whole meaning is how many features it gives up, so the count is "
        f"pinned here rather than described in prose: {blinkered.reads} of {FEATURE_ORDER}")

    def duel(ens, label, *, epsilon, tag="", draws=600):
        _, led = tune_to_tau(
            lambda o: CostSensitiveRouter(ens, offset=o, epsilon=epsilon, explore_tag=tag,
                                          label=label),
            cases, spec, ref, DELTA, VERIFY)
        return led, faceoff(led, lad_led, cases, ref, challenger=label,
                            incumbent="threshold_ladder", draws=draws)

    # --- matched exploration: nobody explores, which is the ladder's own regime ---
    blind_greedy, blind_d = duel(blinkered, "signal_only_greedy", epsilon=0.0)
    _, full_d = duel(ensemble, "all_features_greedy", epsilon=0.0)

    assert blind_d.low > 0, (
        "with the coverage tax off both sides, fitting on the hand rule's OWN feature "
        f"still has to beat it, and does: {blind_d.verdict()}")
    assert full_d.low > blind_d.high, (
        "and the two ablations must be separable, or 'the features did it' is a story "
        f"rather than a measurement: {full_d.verdict()} against {blind_d.verdict()}")
    assert full_d.point > 3 * blind_d.point, (
        f"most of the margin is still the four extra features: {full_d.point:+.1%} with "
        f"them against {blind_d.point:+.1%} without")

    # --- the shipped operating point, where exploration is on and only one side pays ---
    explored = [duel(blinkered, "signal_only", epsilon=EXPLORATION_RATE, tag=t)
                for t in EXPLORE_TAGS]
    full_explored = [duel(ensemble, "all_features", epsilon=EXPLORATION_RATE, tag=t)[1]
                     for t in EXPLORE_TAGS]
    matched = [d for _, d in explored]

    assert all(d.low > 0 for d in full_explored), (
        "the full-feature router must clear two hand cuts on EVERY exploration stream, "
        f"not on a lucky one: {[d.verdict() for d in full_explored if d.low <= 0]}")

    tax = faceoff(blind_greedy, explored[0][0], cases, ref, challenger="signal_only_greedy",
                  incumbent="signal_only_exploring", draws=600)
    assert tax.low > 0 and tax.point > blind_d.point, (
        "the explanation for what follows is that coverage costs this arm more than the "
        f"fitting won it: tax {tax.point:+.1%} against margin {blind_d.point:+.1%}")

    assert any(d.low > 0 for d in matched) and any(d.high < 0 for d in matched), (
        "two draws of ONE fitted policy must return opposite verdicts, each excluding "
        "zero -- that is why a single stream's interval is not the uncertainty of an "
        f"exploring policy: {[f'{d.point:+.1%} [{d.low:+.1%}, {d.high:+.1%}]' for d in matched]}")


def test_the_margin_is_not_a_property_of_one_traffic_mix(spec):
    """A margin measured on one mix is a statement about that mix."""
    seen = {}
    for mix in ("mostly_hard", "bimodal"):
        s = WorkloadSpec(mix=mix, n=2000, seed_tag="v1")
        cs = build_workload(s)
        ref = references(cs, s)
        ens = train(s, n=2500)
        ens.prime(cs, s)
        _, cled = tune_to_tau(lambda o: CostSensitiveRouter(ens, offset=o), cs, s, ref,
                              DELTA, VERIFY)
        _, lled = tune_to_tau(lambda o: ThresholdLadder(DEFAULT_LADDER, offset=o), cs, s,
                              ref, DELTA, VERIFY)
        seen[mix] = faceoff(cled, lled, cs, ref, challenger="classifier",
                            incumbent="threshold_ladder")
    for mix, duel in seen.items():
        assert duel.low > 0, f"{mix}: {duel.verdict()}"


# --------------------------------------------------------------------------------------
# 7. What admissibility is not
# --------------------------------------------------------------------------------------

def test_the_admissibility_call_is_a_point_estimate_the_canary_gate_will_not_advance(
        spec, cases, tuned):
    """Bisecting to the CHEAPEST admissible point lands on the constraint boundary.

    `metrics.score` then calls the policy admissible on a point comparison, while the
    non-inferiority gate at the SAME margin refuses to advance -- because a difference
    sitting exactly at -delta can never have a lower bound above -delta, at any sample
    size. More data does not fix it; it tightens the interval around a point the
    bisection keeps putting back on the line. That is a property of the tuning protocol,
    not of this fixture, and it is the argument for tuning to a bound rather than to a
    point estimate.
    """
    ref = tuned["ref"]
    _, led = tuned["classifier"]
    assert score(led, ref, delta=DELTA).admissible, "the point comparison passes"

    verdicts = {}
    for n in (300, N_TEST):
        s = replace(spec, n=n)
        cs = build_workload(s)
        r = references(cs, s)
        ens = train(s, n=N_TRAIN)
        ens.prime(cs, s)
        _, l = tune_to_tau(lambda o: CostSensitiveRouter(ens, offset=o), cs, s, r, DELTA,
                           VERIFY)
        verdicts[n] = quality_gate(l, run(AlwaysTop(), cs, s), delta=DELTA)

    for n, v in verdicts.items():
        # The gate counts DELIVERED answers -- one per request. Handing it `ledger.rows`
        # instead inflates the variant arm with verify and judge rows, and judge rows are
        # ok=True unconditionally, so the policy's measured quality would rise with the
        # verification rate. Both spellings happen to say "rollback" here, which is why
        # the denominator has to be asserted rather than inferred from the decision.
        assert v.n_variant == n and v.n_baseline == n, (
            f"n={n}: the gate scored {v.n_variant} variant rows against {n} requests")
        assert v.decision != "advance", f"n={n}: gate advanced a boundary-tuned policy: {v.reason}"
        assert v.lower_bound < -DELTA, (
            f"n={n}: the lower bound {v.lower_bound:+.3f} should sit below the -{DELTA:.0%} "
            "margin no matter how much data arrives")
    assert verdicts[N_TEST].lower_bound > verdicts[300].lower_bound, (
        "ten times the data must still tighten the interval -- otherwise this test is "
        "measuring nothing about sample size")
