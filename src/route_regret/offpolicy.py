"""What a policy you did NOT run would have cost, and would have delivered.

Two estimands travel together in every routing post-mortem and they are not the same
kind of object, which is the mistake this module is shaped to prevent.

**Cost is a closed-form plug-in.** Under a deterministic candidate the counterfactual
bill has no sampling error at all: the log already carries the context, so the model
the candidate would have picked and the tokens it would have spent are both known
exactly. Running it through IPS -- reweighting the subset of logged requests where the
two policies happened to agree -- estimates a number you could have computed, and pays
for it with an interval several times wider. That is reporting uncertainty you do not
have.

**Quality is not.** You observe whether the model that ACTUALLY ran succeeded, and
nothing about the models that did not. There is no plug-in; the only route to the
candidate's success rate is reweighting by the logged propensities, and that route is
open only where the logging policy actually explored.

**Positivity is therefore a precondition, not a caveat.** If the logging policy never
sent a case type to a model, the log contains zero information about that cell and no
weighting scheme invents it. The estimators here refuse -- they return no number --
rather than quietly averaging over the cells that happen to be covered and calling the
result an estimate of the whole. A production routing log, which is almost always a
deterministic policy, fails this for nearly every candidate; that is the honest answer
and it is the reason exploration has to be bought deliberately.

**The interval is the deliverable, and its coverage is measured.** A point estimate
that lands near the truth on the seed its author looked at has demonstrated nothing.
`measure_coverage` runs the whole pipeline over hundreds of independent workloads and
reports what fraction of nominal 95% intervals contained a truth obtained by actually
running the candidate. That measured rate is a first-class output, published whatever
it says.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from prompt_experiments.stats.proportions import Interval, wilson_interval

from route_regret.bench import (JUDGE_OUTPUT_TOKENS, JUDGE_PROMPT_OVERHEAD,
                                reference_model, run, verifies)
from route_regret.fixture import REGISTRY, WorkloadSpec, build_workload, features
from route_regret.ledger import Ledger
from route_regret.metrics import paired_bootstrap
from route_regret.models import Case, ModelCard, _unit_draw
from route_regret.policies import Policy

# The order the reward model reads features in. Pinned rather than taken from a dict's
# iteration order so a refit on a different Python build cannot silently permute the
# design matrix and produce a differently-wrong reward model.
FEATURE_KEYS = ("input_tokens", "output_tokens", "n_items", "is_extraction", "signal")

# Context cells for the positivity report. The band EDGES are the log's own quantiles
# rather than fixed token counts, so the cell that ends up empty is a property of the
# traffic and not of a threshold someone picked. The band COUNT is still a constant
# chosen here, and it is load-bearing in the direction that matters: coarser cells hide
# holes, and at LENGTH_BANDS=1 the empirical check stops seeing the very hole
# `test_support_can_be_established_from_the_log_alone_when_the_logger_is_unknown`
# depends on. The empirical branch is therefore a LOWER bound on unsupportedness, which
# is why it is reported as strictly weaker than the exact check rather than as a
# substitute for it.
LENGTH_BANDS = 5


# --------------------------------------------------------------------------------
# The log
# --------------------------------------------------------------------------------

@dataclass(frozen=True)
class LoggedRecord:
    """One request as it survives into a log.

    `case` is the CONTEXT and nothing in this module reads its `difficulty` -- only
    `Policy.choose` touches it, and a policy sees the workload through `features()`.
    The asymmetry that matters is between `model`/`ok`: the reward is observed for the
    action that ran and for no other, which is the entire reason quality needs
    reweighting while cost does not.
    """

    case: Case
    model: str
    ok: bool
    propensity: float
    route_cost: float
    # Verify and judge spend charged to this request. Kept separate because it is a
    # function of the ROUTED model -- a counterfactual that picks the reference model
    # pays none of it -- so a single cost column would misprice every candidate that
    # routes differently.
    overhead_cost: float

    @property
    def total_cost(self) -> float:
        return self.route_cost + self.overhead_cost


@dataclass(frozen=True)
class Log:
    records: list[LoggedRecord]
    spec: WorkloadSpec
    logging_policy: str
    verify_rate: float

    def __len__(self) -> int:
        return len(self.records)


def collect(ledger: Ledger, cases: list[Case], spec: WorkloadSpec, *,
            verify_rate: float = 0.0) -> Log:
    """Turn a bench run into the log an off-policy estimator is allowed to see."""
    by_id = {c.case_id: c for c in cases}
    overhead: dict[str, float] = {}
    routes: dict[str, tuple[str, bool, float, float]] = {}
    policy_name = ""
    for r in ledger.rows:
        if r.call_kind == "route":
            routes[r.request_id] = (r.model, r.ok, r.cost_usd, r.propensity)
            policy_name = r.policy_version
        else:
            overhead[r.request_id] = overhead.get(r.request_id, 0.0) + r.cost_usd
    records = [LoggedRecord(case=by_id[rid], model=model, ok=ok, propensity=p,
                            route_cost=cost, overhead_cost=overhead.get(rid, 0.0))
               for rid, (model, ok, cost, p) in routes.items()]
    return Log(records=records, spec=spec, logging_policy=policy_name,
               verify_rate=verify_rate)


# --------------------------------------------------------------------------------
# The logging policy that makes any of this possible
# --------------------------------------------------------------------------------

class EpsilonExploring:
    """A deterministic router with a fixed exploration budget spent uniformly.

    This class is the price of admission, not a convenience. Off-policy evaluation of a
    router is impossible from the logs of a router that never explores -- every
    counterfactual asks about an action the log never contains -- so the exploration has
    to be paid for in production, before the question is asked. `epsilon` is that bill,
    stated as a number someone signed off on.

    Its `propensity` is the exact sampling distribution of `choose`, and the two are
    written adjacent because a propensity that drifts from the sampling law turns every
    downstream estimate into arithmetic on a fiction that no amount of data corrects.
    """

    def __init__(self, base: Policy, epsilon: float,
                 registry: list[ModelCard] | None = None) -> None:
        if not 0.0 < epsilon <= 1.0:
            raise ValueError("epsilon must be positive; a zero-exploration logger "
                             "cannot identify any counterfactual but its own")
        self.registry = registry or REGISTRY
        self.base = base
        self.epsilon = epsilon
        self.name = f"explore{epsilon:g}_{base.name}"

    def choose(self, case: Case, spec: WorkloadSpec) -> ModelCard:
        if _unit_draw(case.case_id, "explore", spec.seed_tag) < self.epsilon:
            k = int(_unit_draw(case.case_id, "explore-pick", spec.seed_tag)
                    * len(self.registry))
            return self.registry[min(k, len(self.registry) - 1)]
        return self.base.choose(case, spec)

    def propensity(self, case: Case, spec: WorkloadSpec, model: ModelCard) -> float:
        p = self.epsilon / len(self.registry)
        if self.base.choose(case, spec).name == model.name:
            p += 1.0 - self.epsilon
        return p


def tv_divergence(logging_policy: Policy, candidate: Policy, cases: list[Case],
                  spec: WorkloadSpec, registry: list[ModelCard] | None = None) -> float:
    """Mean total-variation distance between the two action distributions.

    The axis along which an off-policy estimate is allowed to be believed. Computed
    from the policies' own propensities rather than from whatever knob generated the
    difference, so a claim that error grows with divergence is a claim about two
    measured quantities instead of a restatement of the sweep.
    """
    registry = registry or REGISTRY
    total = 0.0
    for case in cases:
        total += 0.5 * sum(abs(candidate.propensity(case, spec, m)
                               - logging_policy.propensity(case, spec, m))
                           for m in registry)
    return total / len(cases) if cases else 0.0


# --------------------------------------------------------------------------------
# Positivity
# --------------------------------------------------------------------------------

@dataclass(frozen=True)
class PositivityReport:
    """Whether the log can answer the question at all, and if not, where the hole is."""

    identifiable: bool
    supported_fraction: float
    min_propensity: float
    n_cells: int
    # (context cell, action) pairs the candidate needs and the log never produced.
    # Naming them is the difference between a refusal and a diagnosis: this list is the
    # exploration the log would have had to buy.
    unsupported: list[tuple[str, str]]
    reason: str


def _cell_of(record: LoggedRecord, edges: np.ndarray) -> str:
    band = int(np.searchsorted(edges, record.case.input_tokens))
    return f"{record.case.family}|len{band}"


def _support(policy: Policy, case: Case, spec: WorkloadSpec,
             registry: list[ModelCard]) -> tuple[ModelCard, ...]:
    """Every action this policy would take here with positive probability.

    The short-circuit is not an optimisation for its own sake: a deterministic policy
    puts all its mass on one action, and enumerating the registry to rediscover that on
    every logged request is the difference between a coverage sweep that runs and one
    nobody waits for. The general branch exists because a stochastic candidate needs
    support at EVERY action it might take, not just its modal one -- checking only the
    modal action would declare a randomised policy identifiable off a log that covers
    one of its arms.
    """
    chosen = policy.choose(case, spec)
    if policy.propensity(case, spec, chosen) >= 1.0 - 1e-12:
        return (chosen,)
    return tuple(m for m in registry if policy.propensity(case, spec, m) > 0.0)


def positivity(log: Log, candidate: Policy, *, logging_policy: Policy | None = None,
               registry: list[ModelCard] | None = None) -> PositivityReport:
    """Can this log identify the candidate's value?

    Two checks, because two situations occur. When the logging policy is a known object
    the answer is exact: every action the candidate might take needs positive
    probability under the logger at every logged context. When it is not -- a log
    inherited from a system nobody instrumented -- support has to be established
    empirically, by having OBSERVED that action in a comparable context. The empirical
    check is strictly weaker and is reported as such rather than dressed up as the
    exact one.
    """
    registry = registry or REGISTRY
    if not log.records:
        return PositivityReport(False, 0.0, 0.0, 0, [],
                                "empty log: nothing to reweight")

    lengths = np.array([r.case.input_tokens for r in log.records])
    edges = np.percentile(lengths, [100 * i / LENGTH_BANDS
                                    for i in range(1, LENGTH_BANDS)])

    observed: dict[str, set[str]] = {}
    for rec in log.records:
        observed.setdefault(_cell_of(rec, edges), set()).add(rec.model)

    supported = 0
    holes: set[tuple[str, str]] = set()
    min_p = 1.0
    for rec in log.records:
        cell = _cell_of(rec, edges)
        covered = True
        for action in _support(candidate, rec.case, log.spec, registry):
            if logging_policy is not None:
                p = logging_policy.propensity(rec.case, log.spec, action)
                ok = p > 0.0
                min_p = min(min_p, p)
            else:
                ok = action.name in observed.get(cell, set())
            if not ok:
                covered = False
                holes.add((cell, action.name))
        supported += int(covered)

    fraction = supported / len(log.records)
    identifiable = supported == len(log.records)
    # The empirical branch never reads a propensity, so `min_p` there is still the 1.0 it
    # was initialised to. Printing it would render a figure nothing measured -- the exact
    # defect this repo has shipped three times -- and it would contradict the
    # `min_propensity` field, which correctly reports 0.0 for that branch.
    basis = "known logging policy" if logging_policy is not None else "observed actions"
    detail = (f"min propensity {min_p:.4f}, {basis}" if logging_policy is not None
              else f"{basis}; no propensity was read, so none is reported")
    reason = (
        f"every logged context gives the candidate's action positive probability "
        f"({detail})" if identifiable else
        f"{len(log.records) - supported} of {len(log.records)} logged contexts "
        f"({1 - fraction:.1%}) send the candidate to an action this log never "
        f"observed there; {len(holes)} (cell, action) pairs are empty. No reweighting "
        f"recovers a cell with no data in it -- this needs exploration, not statistics."
    )
    return PositivityReport(identifiable=identifiable, supported_fraction=fraction,
                            min_propensity=min_p if logging_policy is not None else 0.0,
                            n_cells=len(observed), unsupported=sorted(holes),
                            reason=reason)


# --------------------------------------------------------------------------------
# Estimates
# --------------------------------------------------------------------------------

@dataclass(frozen=True)
class Estimate:
    estimand: str
    estimator: str
    value: float | None
    low: float | None
    high: float | None
    n: int
    n_effective: float
    identifiable: bool
    reason: str
    # Populated for cost, where the per-request mean is the estimate and the bill is
    # what anyone actually asks about.
    total: float | None = None

    @property
    def width(self) -> float:
        if self.low is None or self.high is None:
            return float("nan")
        return self.high - self.low

    def line(self) -> str:
        if not self.identifiable:
            return f"{self.estimator:<12}{self.estimand:<16}  not identifiable: {self.reason}"
        return (f"{self.estimator:<12}{self.estimand:<16}{self.value:>10.4f}"
                f"  [{self.low:.4f}, {self.high:.4f}]  n_eff={self.n_effective:.0f}/{self.n}")


def _refused(estimand: str, estimator: str, n: int, report: PositivityReport) -> Estimate:
    return Estimate(estimand=estimand, estimator=estimator, value=None, low=None,
                    high=None, n=n, n_effective=0.0, identifiable=False,
                    reason=report.reason)


def _weights(log: Log, candidate: Policy,
             registry: list[ModelCard] | None = None) -> np.ndarray:
    """IPS weights: pi_1(a|x) / pi_0(a|x) on the action that actually ran.

    Written as a probability RATIO rather than as an indicator over the candidate's
    argmax, because the indicator form silently assumes the candidate is deterministic
    and returns a weight of zero for every arm of a randomised candidate it did not
    happen to pick. For the deterministic policies on this bench the two agree exactly;
    the ratio is what keeps that agreement from being an assumption.

    The weight is zero wherever the candidate would never take the logged action. Those
    requests carry no information about it, and averaging over them anyway is what makes
    an unexplored log look like an answer.
    """
    by_name = {m.name: m for m in (registry or REGISTRY)}
    w = np.zeros(len(log.records))
    for i, rec in enumerate(log.records):
        p1 = candidate.propensity(rec.case, log.spec, by_name[rec.model])
        if p1 > 0.0:
            w[i] = p1 / rec.propensity
    return w


def _wilson_at_effective_n(point: float, sum_w: float, sum_w2: float,
                           confidence: float = 0.95) -> tuple[float, float, float]:
    """A Wilson interval at the sample size the reweighting actually left behind.

    The estimand is a proportion, so the interval comes from the proportions library
    rather than from a normal approximation written here. What is NOT free is `n`: after
    reweighting, a handful of high-weight requests carry most of the estimate, and
    Wilson at the raw row count reports an interval the evidence does not support. Kish's
    effective size (sum w)^2 / sum w^2 is the discount, and whether it is the RIGHT
    discount is not asserted -- `measure_coverage` measures it.
    """
    n_eff = (sum_w * sum_w) / sum_w2 if sum_w2 > 0 else 0.0
    n_int = int(round(n_eff))
    if n_int < 1:
        return float("nan"), float("nan"), n_eff
    successes = int(round(min(1.0, max(0.0, point)) * n_int))
    iv = wilson_interval(successes, n_int, confidence)
    return iv.low, iv.high, n_eff


def quality_snips(log: Log, candidate: Policy, *, logging_policy: Policy | None = None,
                  clip: float | None = None, confidence: float = 0.95) -> Estimate:
    """Self-normalised IPS for the candidate's success rate.

    SELF-NORMALISED, said out loud because the two variants disagree in exactly the
    regime this bench runs in -- but NOT for the reason it is tempting to write down.
    Plain IPS divides by the row count, and the weights sum to n only in expectation; the
    natural next sentence, that it therefore reports a rate "biased toward zero", is
    wrong and was in this docstring until it was measured. Under a correct propensity
    plain IPS is unbiased for the candidate's success rate on the logged cases -- the
    realised weight mass misses n in a direction that is itself random, so what the miss
    buys is VARIANCE, not bias, and the shipped test now states it that way with a
    bootstrap interval on each estimator's mean error.
    What is left after that correction is still decisive: the plain form's support is
    [0, inf), so it can return a "rate" above 1, a diagnostic no reader ever receives
    because it looks like a number. The ratio form is slightly biased and bounded in
    [0, 1]; the bound is worth the bias here because the estimate is fed to an
    admissibility comparison that has no meaning outside it.

    `clip` truncates weights when set: variance for bias, deliberately, and off by
    default because a silent clip makes an unidentifiable estimand look estimable.
    """
    report = positivity(log, candidate, logging_policy=logging_policy)
    if not report.identifiable:
        return _refused("success_rate", "snips", len(log.records), report)

    w = _weights(log, candidate)
    if clip is not None:
        w = np.minimum(w, clip)
    r = np.array([1.0 if rec.ok else 0.0 for rec in log.records])
    sum_w = float(w.sum())
    if sum_w <= 0:
        return _refused("success_rate", "snips", len(log.records), report)

    point = float((w * r).sum() / sum_w)
    low, high, n_eff = _wilson_at_effective_n(point, sum_w, float((w * w).sum()),
                                              confidence)
    return Estimate(estimand="success_rate", estimator="snips", value=point, low=low,
                    high=high, n=len(log.records), n_effective=n_eff, identifiable=True,
                    reason=report.reason)


def quality_ips(log: Log, candidate: Policy, *,
                logging_policy: Policy | None = None) -> Estimate:
    """Plain (unnormalised) IPS, unbounded above and shipped without an interval.

    Here so the self-normalised choice is visible as a choice. It carries no interval on
    purpose: its support is [0, inf), a Wilson interval on it would be a category error,
    and giving it a plausible-looking interval anyway is how the wrong estimator gets
    adopted.
    """
    report = positivity(log, candidate, logging_policy=logging_policy)
    if not report.identifiable:
        return _refused("success_rate", "ips", len(log.records), report)
    w = _weights(log, candidate)
    r = np.array([1.0 if rec.ok else 0.0 for rec in log.records])
    n = len(log.records)
    _, _, n_eff = _wilson_at_effective_n(0.5, float(w.sum()), float((w * w).sum()))
    return Estimate(estimand="success_rate", estimator="ips",
                    value=float((w * r).sum() / n), low=None, high=None, n=n,
                    n_effective=n_eff, identifiable=True, reason=report.reason)


def counterfactual_cost(log: Log, candidate: Policy, *,
                        registry: list[ModelCard] | None = None,
                        draws: int = 600, confidence: float = 0.95) -> Estimate:
    """The candidate's bill, computed rather than estimated.

    No propensities appear anywhere in this function, and that is the point. The
    candidate is deterministic, the context is logged, the price table is known -- so
    the counterfactual cost of every logged request is exact, including the verify and
    judge calls the candidate's own choice would or would not have triggered. What
    remains uncertain is only which requests tomorrow's traffic draws, and that is what
    the interval covers.

    Positivity does not gate this. A cell the logger never explored still has a known
    price, so refusing to price it would be refusing arithmetic.
    """
    registry = registry or REGISTRY
    ref = reference_model(registry)
    per_case = []
    for rec in log.records:
        case = rec.case
        chosen = candidate.choose(case, log.spec)
        cost = chosen.cost(case.input_tokens, case.output_tokens)
        # Mirrors bench.run exactly: verifying a request the reference model itself
        # served would compare a model against itself, so it is never charged.
        if chosen.name != ref.name and verifies(case, log.spec, log.verify_rate):
            cost += ref.cost(case.input_tokens, case.output_tokens)
            judge_in = (case.input_tokens + 2 * case.output_tokens
                        + JUDGE_PROMPT_OVERHEAD)
            cost += ref.cost(judge_in, JUDGE_OUTPUT_TOKENS)
        per_case.append(cost)

    n = len(per_case)
    point, low, high = paired_bootstrap(per_case, [0.0] * n, draws=draws)
    return Estimate(estimand="cost_per_request", estimator="closed_form", value=point,
                    low=low, high=high, n=n, n_effective=float(n), identifiable=True,
                    reason="deterministic candidate over a logged context: no "
                           "reweighting is involved and none is needed",
                    total=point * n)


def ips_cost(log: Log, candidate: Policy, *, logging_policy: Policy | None = None,
             draws: int = 600) -> Estimate:
    """The same bill, estimated by importance weighting -- the version not to ship.

    Kept as the control for `counterfactual_cost`. It discards every logged request
    where the two policies disagreed and inflates the survivors, so it answers a
    question that had an exact answer with an interval several times wider. Reaching
    for it is the reflex this module exists to interrupt.
    """
    report = positivity(log, candidate, logging_policy=logging_policy)
    if not report.identifiable and logging_policy is not None:
        return _refused("cost_per_request", "ips", len(log.records), report)
    w = _weights(log, candidate)
    per_case = [float(w[i] * rec.total_cost) for i, rec in enumerate(log.records)]
    n = len(per_case)
    point, low, high = paired_bootstrap(per_case, [0.0] * n, draws=draws)
    _, _, n_eff = _wilson_at_effective_n(0.5, float(w.sum()), float((w * w).sum()))
    return Estimate(estimand="cost_per_request", estimator="ips", value=point, low=low,
                    high=high, n=n, n_effective=n_eff, identifiable=True,
                    reason=report.reason, total=point * n)


# --------------------------------------------------------------------------------
# Doubly robust
# --------------------------------------------------------------------------------

class ConstantReward:
    """A reward model that is flatly, knowably wrong.

    Not a stub: it is the instrument that shows DR's advertised property is real. A
    reward model whose bias is unknown cannot demonstrate anything, because a small
    error is indistinguishable from a lucky fit.
    """

    def __init__(self, value: float) -> None:
        self.value = value
        self.name = f"constant{value:g}"

    def predict(self, cases: list[Case], spec: WorkloadSpec,
                model_names: list[str]) -> np.ndarray:
        return np.full(len(cases), self.value)


class LogisticReward:
    """P(success | features, model), fit on the log by logistic regression.

    Deliberately the simplest thing that could work, and deliberately misspecified: the
    truth is a logistic in (capability - difficulty) and this model never sees
    difficulty, only the leaky surface features a router is allowed to read. Its
    residual bias is therefore real, and DR has to survive it -- which is the property
    being bought, not an inconvenience to be tuned away.
    """

    def __init__(self, pipeline, model_names: list[str]) -> None:
        self.pipeline = pipeline
        self.model_names = model_names
        self.name = "logistic"

    def _design(self, cases: list[Case], spec: WorkloadSpec,
                model_names: list[str]) -> np.ndarray:
        index = {name: i for i, name in enumerate(self.model_names)}
        rows = np.zeros((len(cases), len(FEATURE_KEYS) + len(self.model_names)))
        for i, (case, name) in enumerate(zip(cases, model_names)):
            f = features(case, spec)
            for j, key in enumerate(FEATURE_KEYS):
                rows[i, j] = f[key]
            rows[i, len(FEATURE_KEYS) + index[name]] = 1.0
        return rows

    def predict(self, cases: list[Case], spec: WorkloadSpec,
                model_names: list[str]) -> np.ndarray:
        if not cases:
            return np.zeros(0)
        return self.pipeline.predict_proba(self._design(cases, spec, model_names))[:, 1]


def fit_reward_model(log: Log, *, registry: list[ModelCard] | None = None
                     ) -> LogisticReward:
    """Fit the outcome model on the log, and only on the log.

    Scaling is not cosmetic here: token counts run to thousands while the one-hot model
    indicators are 0/1, and an unscaled fit spends its whole penalty budget on the
    length coefficients. sklearn is a lazy import because it is an optional extra --
    the reweighting estimators must stay usable without it, since DR is the one that
    needs a model and IPS is the one that must always be available.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    registry = registry or REGISTRY
    names = [m.name for m in registry]
    model = LogisticReward(None, names)
    cases = [r.case for r in log.records]
    x = model._design(cases, log.spec, [r.model for r in log.records])
    y = np.array([1 if r.ok else 0 for r in log.records])

    # lbfgs on a fixed design is deterministic, so a refit reproduces to the last digit
    # -- the same guarantee the fixture makes, extended to the estimator.
    pipe = make_pipeline(StandardScaler(),
                         LogisticRegression(max_iter=2000, solver="lbfgs"))
    pipe.fit(x, y)
    model.pipeline = pipe
    return model


def quality_direct(log: Log, candidate: Policy, reward) -> Estimate:
    """The direct method: ask the reward model and believe it.

    On the bench as the thing DR is measured against. It uses no propensities, so it
    never refuses and never widens -- it inherits the reward model's bias in full and
    reports it with the same confident face it would report a correct answer.
    """
    cases = [r.case for r in log.records]
    actions = [candidate.choose(r.case, log.spec).name for r in log.records]
    rhat = reward.predict(cases, log.spec, actions)
    n = len(cases)
    return Estimate(estimand="success_rate", estimator=f"direct[{reward.name}]",
                    value=float(rhat.mean()), low=None, high=None, n=n,
                    n_effective=float(n), identifiable=True,
                    reason="no reweighting: this is the reward model's opinion, and it "
                           "is exactly as good as the reward model")


def quality_dr(log: Log, candidate: Policy, reward, *,
               logging_policy: Policy | None = None, draws: int = 600) -> Estimate:
    """Doubly robust: the reward model's answer, corrected on the data that can correct it.

    The correction term is evaluated against the SAME reward model that produced the
    first term, so a model that is uniformly wrong has its error subtracted straight
    back out and DR degrades gracefully toward IPS. That is the whole property, and it
    is why a mediocre reward model is acceptable here while a mediocre propensity is
    not -- one of the two is allowed to be wrong, and it is not the propensity.
    """
    report = positivity(log, candidate, logging_policy=logging_policy)
    if not report.identifiable:
        return _refused("success_rate", f"dr[{reward.name}]", len(log.records), report)

    cases = [r.case for r in log.records]
    target_actions = [candidate.choose(r.case, log.spec).name for r in log.records]
    rhat_target = reward.predict(cases, log.spec, target_actions)
    rhat_logged = reward.predict(cases, log.spec, [r.model for r in log.records])

    w = _weights(log, candidate)
    r = np.array([1.0 if rec.ok else 0.0 for rec in log.records])
    pseudo = rhat_target + w * (r - rhat_logged)

    n = len(cases)
    point, low, high = paired_bootstrap(list(pseudo), [0.0] * n, draws=draws)
    _, _, n_eff = _wilson_at_effective_n(0.5, float(w.sum()), float((w * w).sum()))
    return Estimate(estimand="success_rate", estimator=f"dr[{reward.name}]", value=point,
                    low=low, high=high, n=n, n_effective=n_eff, identifiable=True,
                    reason=report.reason)


# --------------------------------------------------------------------------------
# Coverage: the number this module is judged on
# --------------------------------------------------------------------------------

@dataclass(frozen=True)
class CoverageTrial:
    seed_tag: str
    truth: float
    # The candidate's realised success rate on THIS seed's cases. A different target
    # from `truth`, and keeping both is the point -- see CoverageReport.narrate.
    sample_truth: float
    point: float
    low: float
    high: float
    n_effective: float
    covered: bool
    covered_sample: bool
    # What the obvious implementation -- Wilson at the raw row count -- would have said.
    # Carried beside the corrected one so the design effect is visible as a measured
    # difference rather than as a paragraph claiming it matters.
    naive_covered: bool


@dataclass(frozen=True)
class CoverageReport:
    nominal: float
    n: int
    trials: list[CoverageTrial]
    refused: int
    truth: float
    truth_n: int

    @property
    def n_used(self) -> int:
        return len(self.trials)

    @property
    def covered_fraction(self) -> float:
        return sum(t.covered for t in self.trials) / self.n_used if self.trials else 0.0

    @property
    def naive_covered_fraction(self) -> float:
        return (sum(t.naive_covered for t in self.trials) / self.n_used
                if self.trials else 0.0)

    @property
    def sample_covered_fraction(self) -> float:
        return (sum(t.covered_sample for t in self.trials) / self.n_used
                if self.trials else 0.0)

    @property
    def implied_width_scale(self) -> float:
        """The factor every interval would have to be multiplied by to hit nominal.

        Empirical and assumption-free: the `nominal` quantile of |error| / half-width.
        A calibrated interval puts that quantile at 1.0, so 0.8 reads as "20% wider than
        the evidence requires" and 1.3 as "30% too narrow". This exists because a
        coverage number alone says an interval is wrong without saying by how much,
        and the size of a miscalibration decides whether anyone should care.
        """
        if not self.trials:
            return float("nan")
        ratios = [abs(t.point - t.truth) / ((t.high - t.low) / 2)
                  for t in self.trials if t.high > t.low]
        return float(np.percentile(ratios, 100 * self.nominal)) if ratios else float("nan")

    @property
    def mean_width(self) -> float:
        return float(np.mean([t.high - t.low for t in self.trials])) if self.trials else 0.0

    @property
    def mean_n_effective(self) -> float:
        return float(np.mean([t.n_effective for t in self.trials])) if self.trials else 0.0

    @property
    def mean_absolute_error(self) -> float:
        return (float(np.mean([abs(t.point - t.truth) for t in self.trials]))
                if self.trials else 0.0)

    def coverage_interval(self) -> Interval:
        """The band the measured coverage is judged against.

        Not a threshold anyone picked: it is the Monte-Carlo uncertainty of the coverage
        measurement itself. A verdict of miscalibration then means the miscalibration is
        larger than the experiment that found it, which is the only version of that
        claim worth publishing.
        """
        return wilson_interval(sum(t.covered for t in self.trials), self.n_used)

    @property
    def calibrated(self) -> bool:
        band = self.coverage_interval()
        return band.low <= self.nominal <= band.high

    def narrate(self) -> str:
        """Every clause below is read off the trials. Nothing here is typed from what
        the author expected the sweep to show -- including the verdict, which flips on
        the measurement rather than on a comment being updated."""
        band = self.coverage_interval()
        side = "under" if self.covered_fraction < self.nominal else "over"
        verdict = ("calibrated" if self.calibrated
                   else f"MIS-CALIBRATED, {side}-covering")
        survives = self.mean_n_effective / self.n if self.n else 0.0
        correction = ("the effective-n discount earns its place"
                      if self.naive_covered_fraction < self.covered_fraction
                      else "the effective-n discount changed nothing measurable here")
        target = ("looser" if self.sample_covered_fraction > self.covered_fraction
                  else "tighter")
        return "\n".join([
            f"{self.n_used} seeds x n={self.n}, {self.refused} refused for positivity.",
            f"truth {self.truth:.4f}, from running the candidate on {self.truth_n} "
            f"independent cases.",
            f"nominal {self.nominal:.0%} -> measured {self.covered_fraction:.1%} "
            f"[{band.low:.1%}, {band.high:.1%}]: {verdict}.",
            f"intervals are {self.implied_width_scale:.2f}x the width that would hit "
            f"nominal exactly.",
            f"mean interval width {self.mean_width:.1%}, mean |error| "
            f"{self.mean_absolute_error:.1%}.",
            f"reweighting leaves {self.mean_n_effective:.0f} of {self.n} effective "
            f"observations ({survives:.1%} of the log survives).",
            f"same point estimate with Wilson at the raw n covers "
            f"{self.naive_covered_fraction:.1%} -- {correction}.",
            f"scored against each seed's OWN cases instead of the population, coverage "
            f"reads {self.sample_covered_fraction:.1%} -- a {target} target, because "
            f"the estimator and that truth share their workload draw.",
        ])


def measure_coverage(make_logging_policy, make_candidate, *, seeds: int = 240,
                     n: int = 800, mix: str = "balanced", nominal: float = 0.95,
                     tag: str = "cov", signal: float = 0.70, leakage: float = 0.75,
                     truth_n: int = 200_000) -> CoverageReport:
    """Run the whole estimator end to end on `seeds` independent workloads.

    The truth is not a formula: the candidate policy is actually run, on a large
    workload drawn from the same spec but a disjoint seed, and its delivered success
    rate read off the ledger.

    INDEPENDENT is the load-bearing word, and getting it wrong is the quiet way to
    manufacture a calibrated-looking interval. Scoring each seed against the candidate's
    success rate on that seed's OWN cases is tempting -- it looks like it isolates the
    reweighting -- but the estimator and that truth are computed from the same workload
    draw, so the workload's sampling noise appears in both and cancels. The interval was
    built to cover the POPULATION value and includes that noise, so measuring it against
    a target that has it removed reports coverage the interval has not earned. Both
    numbers are recorded; only the independent one is the headline.

    Because every draw of the fixture is a hash of the case id, the candidate's
    population value does not depend on `seed_tag` -- so one large run serves as the
    truth for every seed, at a Monte-Carlo error far below the intervals being tested.
    """
    truth_spec = WorkloadSpec(mix=mix, n=truth_n, leakage=leakage, signal=signal,
                              seed_tag=f"{tag}-truth")
    truth_cases = build_workload(truth_spec)
    truth = 1.0 - run(make_candidate(), truth_cases, truth_spec,
                      verify_rate=0.0).violation_rate()
    del truth_cases

    trials: list[CoverageTrial] = []
    refused = 0
    for k in range(seeds):
        spec = WorkloadSpec(mix=mix, n=n, leakage=leakage, signal=signal,
                            seed_tag=f"{tag}{k}")
        cases = build_workload(spec)
        logger = make_logging_policy()
        candidate = make_candidate()

        log = collect(run(logger, cases, spec, verify_rate=0.0), cases, spec)
        # `nominal` has to reach the estimator, not just the yardstick. It did not: the
        # interval under test was pinned at quality_snips' 0.95 default while the naive
        # control below was built at `nominal`, so every call with nominal != 0.95
        # measured a 95% interval against a bar it was never asked to meet -- and the two
        # coverages were computed at different confidence levels, which makes the design
        # -effect comparison meaningless. Measured before the fix: mean interval width
        # was byte-identical at nominal 0.80, 0.95 and 0.99.
        est = quality_snips(log, candidate, logging_policy=logger, confidence=nominal)
        if not est.identifiable:
            refused += 1
            continue

        sample_truth = 1.0 - run(candidate, cases, spec, verify_rate=0.0).violation_rate()
        naive = wilson_interval(int(round(est.value * n)), n, nominal)
        trials.append(CoverageTrial(
            seed_tag=spec.seed_tag, truth=truth, sample_truth=sample_truth,
            point=est.value, low=est.low, high=est.high, n_effective=est.n_effective,
            covered=bool(est.low <= truth <= est.high),
            covered_sample=bool(est.low <= sample_truth <= est.high),
            naive_covered=bool(naive.low <= truth <= naive.high),
        ))
    return CoverageReport(nominal=nominal, n=n, trials=trials, refused=refused,
                          truth=truth, truth_n=truth_n)
