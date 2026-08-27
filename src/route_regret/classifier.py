"""A trained router, entered on the bench beside the hand rules rather than sold as the
product.

Four things the usual "we trained a complexity classifier" write-up gets wrong, and what
is done here instead.

**The label.** Hand-labelled complexity tiers make the label a function of the features.
A person labelling "how hard is this prompt" has only the prompt surface to go on, which
is precisely what the classifier reads, so the fitted model recovers the labeller's own
rule and reports its agreement with a person as though it were a routing result. The
label here is ADEQUACY -- did this model actually produce an acceptable answer for this
case, from `models.succeeds` on a training workload. `SurfaceLabeller` reproduces the
circular version so the difference can be measured rather than argued.

**The score.** A bare accuracy scalar is unreadable without the majority class it is
standing on: 80% against a 55% majority is kappa 0.55, not "80% good". `AdequacyReport`
renders accuracy, the majority baseline, Cohen's kappa and both off-diagonal cells with
Wilson intervals as one string, so quoting the accuracy without its deflator takes
deliberate effort.

**The loss.** The two routing errors are not symmetric and do not cost the same thing.
Under-routing (sent cheap, needed expensive) costs quality and a verifier can see it.
Over-routing (sent expensive, cheap would have done) costs money and is structurally
INVISIBLE to a verifier that compares the top model against itself -- it agrees, so the
request is booked as a success. `CostSensitiveRouter` therefore chooses by minimising
`price + lambda x P(inadequate)` per case, with lambda the tuning knob, and never by the
classifier's argmax. `FlatThresholdRouter` is the argmax version, kept as the foil.

That much survived measurement. The mechanism originally claimed for it did NOT, and the
refutation is left in the suite rather than deleted: see `CostSensitiveRouter.greedy`.

**The loop.** A retraining loop that only labels the arm it acted on has no
counterfactual for any tier it stopped choosing, so it converges to always-frontier and
calls that learning. This policy explores in BOTH directions -- sometimes a case the
model wants routed down goes up, and vice versa -- and `propensity` returns the true
probability including exploration, because phase 4's off-policy estimator divides by it.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from ai_feature_flags.canary import GateConfig, GateVerdict, evaluate_gate
from prompt_experiments.stats.proportions import Interval, wilson_interval
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import cohen_kappa_score
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from route_regret.bench import reference_model
from route_regret.fixture import REGISTRY, WorkloadSpec, build_workload, features
from route_regret.ledger import Ledger
from route_regret.metrics import Reference, paired_bootstrap, score
from route_regret.models import Case, ModelCard, _unit_draw, succeeds

# Exactly what `fixture.features` exposes, in a fixed order so a fitted model and the
# matrix it is later asked to score cannot silently disagree about column meaning.
FEATURE_ORDER: tuple[str, ...] = ("input_tokens", "output_tokens", "n_items",
                                  "is_extraction", "signal")

# How often the policy deviates from its own best guess, split evenly between routing a
# case down a tier and up a tier. Sized to stay well inside the delta budget: at this rate
# the downward half adds roughly EXPLORATION_RATE/2 x P(the cheaper tier fails where the
# chosen one succeeds) to the violation rate, which the tuner then has to buy back. Set it
# large enough to matter to the constraint and the policy is paying for coverage it could
# have had for less.
EXPLORATION_RATE = 0.10


def design_matrix(cases: list[Case], spec: WorkloadSpec,
                  reads: tuple[str, ...] = FEATURE_ORDER) -> np.ndarray:
    """The rows a router is allowed to fit on. `reads` is a subset, not a convenience:
    the falsification test needs a model restricted to the hand rule's own single
    feature, or "trained beats hand-tuned" is a comparison of inputs wearing the costume
    of a comparison of methods."""
    return np.array([[features(c, spec)[k] for k in reads] for c in cases], dtype=float)


def adequacy(card: ModelCard, cases: list[Case]) -> np.ndarray:
    """THE LABEL: did this model actually produce an acceptable answer for this case.

    Observed behaviour, not an impression of difficulty. It moves when the registry
    moves, which is the property a hand-labelled complexity tier does not have and the
    whole reason the hand-labelled version scores so much better while meaning nothing.
    """
    return np.array([succeeds(card, c) for c in cases], dtype=int)


class SurfaceLabeller:
    """The circular label, kept runnable so the circularity is a measurement.

    A human labeller reads the prompt and calls it simple or complex. Everything they can
    see is in `features`, so their rule is some monotone function of those columns and the
    label is a re-encoding of the input. The particular weights below are arbitrary and
    the demonstration does not depend on them -- any rule a person could apply from the
    prompt alone has the same defect. The cut is fixed on the workload the labeller
    "worked through", not recomputed per evaluation, because a labeller settles on a
    standard once and then applies it.
    """

    def __init__(self, cases: list[Case], spec: WorkloadSpec) -> None:
        scores = self._score(cases, spec)
        self.cut = float(np.median(scores))

    @staticmethod
    def _score(cases: list[Case], spec: WorkloadSpec) -> np.ndarray:
        f = [features(c, spec) for c in cases]
        length = np.array([x["input_tokens"] for x in f])
        length = (length - length.min()) / max(1.0, length.max() - length.min())
        items = np.array([min(1.0, x["n_items"] / 120.0) for x in f])
        signal = np.array([x["signal"] for x in f])
        return 0.5 * length + 0.3 * signal + 0.2 * items

    def __call__(self, cases: list[Case], spec: WorkloadSpec) -> np.ndarray:
        """1 where the labeller judged the prompt simple enough for the cheap model --
        oriented like `adequacy` so the two labels are directly comparable."""
        return (self._score(cases, spec) <= self.cut).astype(int)


@dataclass(frozen=True)
class Head:
    """One fitted P(label | features). Carries the columns it was fitted on so a later
    call cannot score it against a differently-shaped matrix."""

    reads: tuple[str, ...]
    pipeline: Pipeline

    def probability(self, cases: list[Case], spec: WorkloadSpec) -> np.ndarray:
        return self.pipeline.predict_proba(design_matrix(cases, spec, self.reads))[:, 1]

    def predict(self, cases: list[Case], spec: WorkloadSpec) -> np.ndarray:
        return self.pipeline.predict(design_matrix(cases, spec, self.reads))


def fit_head(cases: list[Case], spec: WorkloadSpec, labels: np.ndarray,
             reads: tuple[str, ...] = FEATURE_ORDER) -> Head:
    """Logistic regression, deliberately: lbfgs on standardised columns is deterministic,
    so the bench's "two runs agree to the cent" survives a learned entrant. A model with
    a sampled fit would put RNG state into a repo whose whole fixture exists to avoid it.
    """
    pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    pipe.fit(design_matrix(cases, spec, reads), labels)
    return Head(reads=tuple(reads), pipeline=pipe)


@dataclass(frozen=True)
class AdequacyReport:
    """Accuracy is in here, and it never leaves the building alone.

    `under_route` and `over_route` are the two off-diagonal cells conditioned on the truth
    class, so their denominators are the class counts -- which is exactly why the majority
    baseline has to travel with them. Wilson rather than Wald: at the small denominator of
    a rare class, Wald reports zero width for a cell it has seen nothing in.
    """

    model: str
    n: int
    accuracy: float
    kappa: float
    majority_baseline: float
    under_route: Interval   # of the cases this model could NOT serve, the share sent here
    over_route: Interval    # of the cases it COULD serve, the share sent somewhere dearer

    def headline(self) -> str:
        return (f"{self.model}: {self.accuracy:.1%} accurate against a "
                f"{self.majority_baseline:.1%} majority class -> kappa {self.kappa:.3f} "
                f"(n={self.n}). under-routing {self.under_route}; "
                f"over-routing {self.over_route}")

    def __str__(self) -> str:
        return self.headline()


def report_adequacy(truth: np.ndarray, predicted: np.ndarray, model: str) -> AdequacyReport:
    truth = np.asarray(truth).astype(int)
    predicted = np.asarray(predicted).astype(int)
    if truth.shape != predicted.shape:
        raise ValueError("truth and prediction must describe the same cases")
    positives = int(truth.sum())
    negatives = int(len(truth) - positives)
    share = positives / len(truth) if len(truth) else 0.0
    return AdequacyReport(
        model=model,
        n=int(len(truth)),
        accuracy=float((truth == predicted).mean()),
        # sklearn's kappa rather than a hand-rolled (p_o - p_e)/(1 - p_e): the degenerate
        # cases -- one empty class, perfect agreement -- are where a local version is
        # wrong, and they are the cases a degenerate classifier lands in.
        kappa=float(cohen_kappa_score(truth, predicted)),
        majority_baseline=max(share, 1.0 - share),
        under_route=wilson_interval(int(((predicted == 1) & (truth == 0)).sum()), negatives),
        over_route=wilson_interval(int(((predicted == 0) & (truth == 1)).sum()), positives),
    )


class AdequacyEnsemble:
    """One adequacy head per model on the bench, plus the scale the router prices in.

    Per model rather than one multiclass "which tier" head: the routing decision needs
    P(adequate) for each candidate separately in order to price the two errors against
    that candidate's own price gap, and a single argmax head answers a question
    ("which tier is best") that has already folded in a cost trade-off nobody declared.
    """

    def __init__(self, heads: dict[str, Head], registry: list[ModelCard],
                 stake_scale: float, reads: tuple[str, ...]) -> None:
        self.heads = heads
        self.registry = registry
        self.reads = reads
        self.top = reference_model(registry)
        self.tiers = sorted(registry, key=lambda m: m.cost(1000, 500))
        # Median dollars at stake on one request, from the TRAINING workload. It sets the
        # units of the router's lambda so one bisection range works for this policy and
        # the hand ladder alike; a typed constant here would make the knob's reachable
        # range a property of the price table.
        self.stake_scale = stake_scale
        self._cache: dict[tuple, dict[str, float]] = {}

    def prime(self, cases: list[Case], spec: WorkloadSpec) -> None:
        """Score a whole workload at once and keep it.

        Not an optimisation detail: `tune_to_tau` rebuilds the policy at every bisection
        step, so a per-call `predict_proba` would re-score the same workload eighteen
        times and put a three-order-of-magnitude cost on being tunable, which is the one
        property that makes this comparable to the other entrants.
        """
        probs = {name: head.probability(cases, spec) for name, head in self.heads.items()}
        for i, case in enumerate(cases):
            self._cache[(spec, case.case_id)] = {n: float(p[i]) for n, p in probs.items()}

    def probabilities(self, case: Case, spec: WorkloadSpec) -> dict[str, float]:
        key = (spec, case.case_id)
        if key not in self._cache:
            self.prime([case], spec)
        return self._cache[key]

    def report(self, card: ModelCard, cases: list[Case], spec: WorkloadSpec) -> AdequacyReport:
        head = self.heads[card.name]
        return report_adequacy(adequacy(card, cases), head.predict(cases, spec), card.name)


def train(spec: WorkloadSpec, *, n: int, seed_tag: str = "train",
          reads: tuple[str, ...] = FEATURE_ORDER,
          registry: list[ModelCard] | None = None) -> AdequacyEnsemble:
    """Fit adequacy heads on a workload DISJOINT from the one that will be scored.

    Case ids are `seed_tag:mix:index`, so the same tag is the same cases and the reported
    accuracy becomes memorisation. Refusing it here rather than documenting it, because
    the failure is silent and flattering: the number goes up.
    """
    if seed_tag == spec.seed_tag:
        raise ValueError(
            f"training and evaluation both tagged {seed_tag!r}: identical case ids, so "
            "held-out accuracy would be a lookup")
    registry = registry or REGISTRY
    train_spec = replace(spec, seed_tag=seed_tag, n=n)
    cases = build_workload(train_spec)

    top = reference_model(registry)
    cheapest = min(registry, key=lambda m: m.cost(1000, 500))
    stake = np.array([top.cost(c.input_tokens, c.output_tokens)
                      - cheapest.cost(c.input_tokens, c.output_tokens) for c in cases])
    heads = {m.name: fit_head(cases, train_spec, adequacy(m, cases), reads)
             for m in registry}
    return AdequacyEnsemble(heads, registry, float(np.median(stake)), tuple(reads))


class CostSensitiveRouter:
    """Route by minimising `price + lambda x P(inadequate)`, per case, per candidate.

    Not the classifier's argmax. Argmax answers "is this model probably adequate", which
    prices both errors at one unit each, and the +7.5 FASC points this beats
    `FlatThresholdRouter` by is what pricing them separately is worth. lambda is
    dollars-per-violation -- the shadow price of the quality constraint -- and it is the
    knob `report.tune_to_tau` bisects, so this entrant is pinned to the same delivered
    quality as every other one before any cost is compared.

    The reason first written here for that margin was wrong, and the wrong version is
    worth keeping visible because it is the plausible one. It said the decision is whether
    the price gap on THIS request is worth its incremental risk, the gaps differing 8.7x
    across the workload because prompt lengths do. The gaps do differ 8.7x. Conditioning
    on them is worth -1.1 FASC points [-2.0, -0.1]: replace `case.input_tokens` below with
    the ladder's flat unit price and the router gets BETTER, and the whole +8.6 over the
    flat cut turns out to be the price ladder entering the loss at all, not this case's
    place on it. The mechanism is the repo's own coupling turned back on the router --
    length leaks difficulty, so a long prompt simultaneously widens the price gap (routing
    it down) and raises P(inadequate) (routing it up), and the price term amplifies the
    wrong one exactly where it costs most. `test_reading_this_requests_own_price_is_the_
    part_that_does_not_work` holds that number; per-case pricing is kept because it is
    what the reported figures were measured on, not because it earned its place.
    """

    def __init__(self, ensemble: AdequacyEnsemble, offset: float = 0.0,
                 epsilon: float = EXPLORATION_RATE, label: str = "classifier",
                 explore_tag: str = "") -> None:
        self.ensemble = ensemble
        self.name = label
        self.offset = offset
        self.epsilon = epsilon
        # Which exploration stream this run drew from. Deliberately NOT the policy name:
        # two entrants compared on one workload should explore identically, so the paired
        # bootstrap between them is an interval on their routing and not on their coin
        # flips. It is a parameter rather than a constant because the realisation is worth
        # a couple of FASC points on its own -- a margin smaller than that is a statement
        # about one set of draws, and phase 4 has to log which set a run used to replay it.
        self.explore_tag = explore_tag
        self.tiers = ensemble.tiers
        self.top = ensemble.top
        # Log-scaled so one bisection over [-0.6, 0.6] -- the range the other entrants are
        # tuned over -- spans from "price is all that matters" to "risk is all that
        # matters". A linear knob would spend most of its range in one regime.
        self.lam = ensemble.stake_scale * (10.0 ** (4.0 * offset))

    def greedy(self, case: Case, spec: WorkloadSpec) -> ModelCard:
        """The action before exploration. Ties break toward the cheaper tier because the
        list is in price order and the comparison is strict."""
        p = self.ensemble.probabilities(case, spec)
        best, best_loss = None, None
        for m in self.tiers:
            loss = (m.cost(case.input_tokens, case.output_tokens)
                    + self.lam * (1.0 - p[m.name]))
            if best_loss is None or loss < best_loss - 1e-15:
                best, best_loss = m, loss
        return best

    def _plan(self, case: Case, spec: WorkloadSpec):
        """Greedy arm, its two price neighbours, and the mass on each.

        Neighbours in PRICE order rather than "the tier the model ranked second": phase 4
        can only reweight arms the logged run actually covered, and price-adjacent arms
        are the ones a counterfactual policy is most likely to want. Where a neighbour
        does not exist its mass returns to the greedy arm, so the propensities still sum
        to one and the top of the ladder still explores downward.
        """
        g = self.greedy(case, spec)
        gi = self.tiers.index(g)
        down = gi - 1 if gi > 0 else None
        up = gi + 1 if gi < len(self.tiers) - 1 else None
        return (g, down, up,
                self.epsilon / 2 if down is not None else 0.0,
                self.epsilon / 2 if up is not None else 0.0)

    def choose(self, case: Case, spec: WorkloadSpec) -> ModelCard:
        g, down, up, p_down, p_up = self._plan(case, spec)
        u = _unit_draw(case.case_id, "explore", spec.seed_tag, self.explore_tag)
        if u < p_down:
            return self.tiers[down]
        if u < p_down + p_up:
            return self.tiers[up]
        return g

    def propensity(self, case: Case, spec: WorkloadSpec, model: ModelCard) -> float:
        g, down, up, p_down, p_up = self._plan(case, spec)
        total = 0.0
        if model.name == g.name:
            total += 1.0 - p_down - p_up
        if down is not None and model.name == self.tiers[down].name:
            total += p_down
        if up is not None and model.name == self.tiers[up].name:
            total += p_up
        return total


class FlatThresholdRouter(CostSensitiveRouter):
    """The same fitted probabilities, read as class labels: route to the cheapest tier
    whose P(adequate) clears one flat cut. At offset 0 the cut is 0.5 and this IS the
    classifier's argmax.

    On the bench as the foil for the asymmetry claim. Because it shares the fit, its
    adequacy accuracy is identical to the cost-sensitive router's by construction, so any
    difference between them at matched quality is the loss function and nothing else.
    """

    def __init__(self, ensemble: AdequacyEnsemble, offset: float = 0.0,
                 epsilon: float = EXPLORATION_RATE, label: str = "classifier_flat",
                 explore_tag: str = "") -> None:
        super().__init__(ensemble, offset=offset, epsilon=epsilon, label=label,
                         explore_tag=explore_tag)
        self.threshold = min(1.0, max(0.0, 0.5 + offset))

    def greedy(self, case: Case, spec: WorkloadSpec) -> ModelCard:
        p = self.ensemble.probabilities(case, spec)
        for m in self.tiers:
            if m.name == self.top.name:
                break
            if p[m.name] >= self.threshold:
                return m
        return self.top


def quality_gate(policy_ledger: Ledger, top_ledger: Ledger, *,
                 delta: float = 0.03) -> GateVerdict:
    """Ask whether the delivered quality is non-inferior to always-frontier AS EVIDENCE.

    `metrics.score` decides admissibility by comparing two point estimates to tau, which
    is the right definition of the constraint and the wrong question to stop at: a policy
    bisected to the CHEAPEST admissible point sits on the boundary by construction, where
    a one-sided lower bound can never clear -delta at any sample size. Routing the same
    two proportions through the canary gate is how that gets said out loud instead of
    being an unstated caveat on every FASC figure in the repo.
    """
    base = top_ledger.delivered()
    var = policy_ledger.delivered()
    return evaluate_gate(sum(1 for r in base if r.ok), len(base),
                         sum(1 for r in var if r.ok), len(var),
                         GateConfig(margin=delta))


@dataclass(frozen=True)
class Faceoff:
    """Two entrants at matched quality, and the interval that says whether the gap is
    real. The verdict sentence is derived from the interval, never written alongside it --
    a typed conclusion is how a repo ends up publishing a claim its own numbers stopped
    supporting three commits ago."""

    challenger: str
    incumbent: str
    challenger_fasc: float | None
    incumbent_fasc: float | None
    point: float
    low: float
    high: float
    n: int

    def verdict(self) -> str:
        gap = (f"{self.point:+.1%} FASC@delta [{self.low:+.1%}, {self.high:+.1%}] "
               f"over n={self.n} paired cases")
        if self.challenger_fasc is None or self.incumbent_fasc is None:
            missing = self.challenger if self.challenger_fasc is None else self.incumbent
            return (f"no comparison: {missing} did not reach the quality bar, so its cost "
                    f"is not a number anything may be compared against ({gap})")
        if self.low > 0:
            return (f"{self.challenger} beats {self.incumbent} by {gap}; the interval "
                    "excludes zero")
        if self.high < 0:
            return (f"{self.challenger} LOSES to {self.incumbent} by {gap}; the interval "
                    "excludes zero, so the extra machinery bought negative value")
        return (f"{self.challenger} and {self.incumbent} cannot be told apart: {gap} "
                "straddles zero")

    def __str__(self) -> str:
        return self.verdict()


def _cost_per_case(ledger: Ledger, cases: list[Case]) -> list[float]:
    """Everything the system spent on each request, verification and judging included."""
    totals = {c.case_id: 0.0 for c in cases}
    for row in ledger.rows:
        totals[row.request_id] += row.cost_usd
    return [totals[c.case_id] for c in cases]


def faceoff(challenger_ledger: Ledger, incumbent_ledger: Ledger, cases: list[Case],
            ref: Reference, *, challenger: str, incumbent: str, delta: float = 0.03,
            draws: int = 1000) -> Faceoff:
    """Paired interval on FASC@delta(challenger) - FASC@delta(incumbent).

    Paired on the case, because both policies saw the same workload and an unpaired
    interval here would be wide enough to hide any margin either could earn. FASC is an
    affine function of total cost against a shared denominator, so the per-case cost
    difference carries straight over: multiply the mean by n/achievable and the interval
    is in FASC points rather than dollars.
    """
    if ref.achievable <= 0:
        raise ValueError("no achievable savings on this workload: FASC is undefined, and "
                         "a difference of undefined quantities is not a finding")
    a = _cost_per_case(incumbent_ledger, cases)
    b = _cost_per_case(challenger_ledger, cases)
    point, low, high = paired_bootstrap(a, b, draws=draws, tag="faceoff")
    scale = len(cases) / ref.achievable
    # Admissibility comes from `metrics.score` rather than being re-derived here: two
    # places that decide what counts as admissible eventually disagree, and the one in the
    # comparison function is the one that would quietly win.
    return Faceoff(challenger=challenger, incumbent=incumbent,
                   challenger_fasc=score(challenger_ledger, ref, delta=delta).fasc_at_delta,
                   incumbent_fasc=score(incumbent_ledger, ref, delta=delta).fasc_at_delta,
                   point=point * scale, low=low * scale, high=high * scale, n=len(cases))
