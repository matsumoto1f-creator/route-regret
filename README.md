# route-regret

A bench for LLM routing policies. **The obvious way to measure a cost router rewards
routing badly** — this measures the thing that doesn't.

Runs entirely offline: a deterministic capability-graded fixture, no API key, no
container, no clock.

---

## Start here

```bash
pip install -e ".[dev,classifier,gateway]"
route-regret arithmetic
```

```
Break-even verification rate  v* = 1 - c_cheap/c_ref

cheap model          $/req   v* no judge   v* with judge
--------------------------------------------------------
gpt-4o-mini       0.000360        97.4%           56.2%
haiku-4.5         0.002800        80.0%           46.2%
sonnet-5          0.005600        60.0%           34.6%
gpt-4o            0.006000        57.1%           33.0%
```

**That is arithmetic, not a benchmark.** Verify a routed answer against a reference
model more often than `v*` and routing costs more than never routing at all. The judge
is itself a reference-model call, and pricing it roughly halves the rate — which is why
the common design of *verify every routed request* is on the wrong side of this line for
every model pair in the registry.

---

## The problem this exists for

The standard way to score a cost router is *"percent saved versus sending everything to
the frontier model."* On a workload where harder prompts are longer — which is the
normal case, and the assumption behind using token count as a routing feature — that
number is **anti-correlated with the router being right**.

The mechanism is simple once seen: a router that routes badly sends the long, expensive
prompts to the cheap model, and books a larger saving for it.

```bash
route-regret bench --n 6000
```

```
policy                    cost $  violation   naive sav         FASC@d
------------------------------------------------------------------------
oracle                   39.1222      10.2%       57.9%           100.0%
always_top               92.9187      14.2%        0.0%             0.0%
always_cheapest          10.9344      63.1%       88.2%   not admissible
threshold_ladder         70.6425      17.2%       24.0%            41.4%
content_blind            82.3352      17.2%       11.4%            19.7%
```

`always_cheapest` — no routing logic whatsoever — **wins the naive metric outright at
88.2%**, while failing 63% of requests. Any system tuned to that number learns to stop
routing.

Costs scale with `--n`; the ratios do not. Every figure in this README is from
`--n 6000` unless stated, and reruns reproduce to the cent because the fixture is a
pure function of `(case_id, model)`.

## What actually fixes it, and it is not a better metric

Pin the operating point. Tune every policy until it delivers the **same quality**, then
compare only cost. Under that protocol the perverse incentive disappears — a worse
router can no longer buy a cheaper bill, because it has to spend its way back to the
quality bar.

Measured over a signal-degradation sweep (`tests/test_metric_honesty.py`):

| regime | corr(router accuracy, naive savings) | corr(accuracy, FASC@δ) |
|---|---|---|
| fixed operating point | **negative** | — |
| fixed operating point, oracle-normalised | **negative** | — |
| pinned to matched quality | positive | positive |

Oracle normalisation alone does **not** fix it: it rescales a numerator whose sign is
already wrong. The constraint does all the work. That correction is in the test suite
because the author proposed the normalisation as the fix and was wrong.

**FASC@δ** — fraction of achievable savings captured, scored only for policies whose
delivered failure rate stays within δ of always-frontier:

```
τ      = violation(always_top) + δ          δ = 3pp, pre-registered
FASC@δ = (cost(always_top) − cost(policy)) / (cost(always_top) − cost(oracle))
         evaluated only where violation(policy) ≤ τ
```

The oracle is clairvoyant — it picks the cheapest model that *actually* succeeded on
each case. It is unbuildable by construction, because it is a denominator rather than a
target.

## The claim, and the test that kills it

The router captures **41.4%** of achievable savings against **19.7%** for a
content-blind control given the router's own model marginal and tuned to the same
quality bar — a margin of **21.7 points** it had to earn by reading the request.

`test_the_router_earns_its_margin_over_a_blind_control_at_matched_quality` goes red if
that margin disappears. The control is anchored so it can always reach the quality bar;
without that anchor "the router is admissible and the control is not" would be true by
construction, and the headline would be a test that cannot fail.

## No number here is mix-invariant

Nothing summarising a router in one scalar can be. What the bench does is show the size
of the effect rather than claim it was handled. Same router, five traffic mixes:

| how it is reported | spread across mixes |
|---|---|
| naive savings (scalar) | 23.2pp |
| FASC@δ (scalar) | 21.2pp |
| per-stratum, re-tuned per mix | 15.1pp |
| per-stratum, policy frozen | 8.1pp |

```bash
route-regret mixes     # the scalar view, with the retired metric beside it
route-regret strata    # FASC per difficulty decile
```

A stratum whose achievable savings are negligible reports `not_estimable` rather than a
percentage — dividing by a denominator near zero manufactures findings. An **untuned**
policy exceeds 100% in the hard strata, which is not a bug: it is inadmissibility
showing up per-stratum, and the reason a stratum table without a quality column is as
misleading as the scalar it replaced.

## The instrument has a ceiling, and it is published

Two independent samples of the *same* model at accuracy `p` agree only
`p² + (1−p)²/(L−1)` of the time. At 95% accuracy with binary labels that is **90.5%** —
so a pipeline comparing a model against itself and reporting "9.5% routing failures" is
reporting its own noise floor. Every parity figure here is a distance from that ceiling,
never from 100%.

## The fixture, and why it is the load-bearing part

Two separations decide whether any of this measures anything:

- **Capability is authored independently of price.** If capability were derived from
  price, "route to the cheapest adequate model" would be the same instruction as "route
  to the cheapest model". Real ladders are not capability ladders: in this registry
  `sonnet-5` is cheaper per token than `gpt-4o` *and* stronger, and at Aug-2026 list
  prices Claude Haiku 4.5 costs 7.8× GPT-4o-mini while being the better model.
- **Difficulty is declared, not derived from the text.** If latent difficulty were
  recoverable from the features the router reads, routing accuracy would be 1.0 by
  construction. `leakage` and `signal` are therefore **declared, swept parameters**, and
  `tests/test_fixture_can_break.py` demonstrates the degenerate state: at full leakage a
  router reading nothing but token count matches the oracle, and the bench measures
  nothing.

Success is `sha256(case_id|model)` against a logistic capability curve — no RNG, no
clock, no evaluation-order dependence. Two runs agree to the cent.

Break-the-fixture tests are part of the suite, not a footnote: make the cheap model
near-perfect and the oracle must move onto it; make every non-frontier model useless and
achievable savings must vanish and FASC must become undefined rather than large.

## What this reuses

Real dependencies, not decorative ones:

- [`prompt-experiments`](https://github.com/matsumoto1f-creator/prompt-experiments) —
  Wilson intervals and sample-size functions. Intervals are never hand-rolled here.
- [`ai-feature-flags`](https://github.com/matsumoto1f-creator/ai-feature-flags) — the
  non-inferiority gate. `insufficient_data` is a required verdict, not a failure.
- [`llm-gateway`](https://github.com/matsumoto1f-creator/llm-gateway) (extra) — pricing,
  the cost ledger, admission. That gateway routes on **failure**; this routes on
  **content**. The seam: this chooses a model, the gateway serves it.

## The classifier is an entrant, not the product

```bash
route-regret classify
```

Trained on **adequacy** — did the cheap model actually produce an acceptable answer —
never on a human's impression of complexity. That matters because a person hand-labelling
has only the prompt's surface to go on, which are precisely the features the classifier
reads, so the labels become a function of the features and the model merely recovers the
labeller's own rule. In the suite that circular label scores **99.4%** against **88.0%**
for the honest target: a 20x lower error rate, carrying no information about any model.

Never a bare accuracy scalar. Every figure ships with its majority-class baseline, its
kappa, and the two off-diagonal cells with Wilson intervals — because 88% against a 66%
majority is a kappa of 0.73, and under-routing and over-routing are not the same mistake:

```
gpt-4o-mini: 88.1% accurate against a 65.3% majority class -> kappa 0.736 (n=2500).
             under-routing 8.5% [7.3%, 10.0%]; over-routing 18.2% [15.8%, 20.9%]
```

The decision rule is a per-case Lagrangian argmin on `price + λ·P(inadequate)`, not
argmax. Same fitted model, two losses, accuracy identical by construction: the
cost-sensitive rule is worth **+7.5 FASC points [+5.5, +9.7]**.

Against the two-cut hand rule at matched quality it wins by **+17.0pp [+14.5, +19.7]** —
but the suite is explicit about where that comes from. Restricted to the *same single
feature* the hand rule reads, the margin's sign flips across exploration draws. Most of
the win is features the hand rule was never given, not the fitting.

## Estimating a policy you did not run

```bash
route-regret offpolicy
```

The headline is not a point estimate — it is whether a nominal 95% interval actually
contains the truth 95% of the time, with truth taken from running the candidate for real
on 200,000 independent cases:

```
nominal 95%  ->  measured 95.0% [83.5%, 98.6%]   CALIBRATED
reweighting leaves 33 of 500 effective observations (6.7% of the log survives)
same point estimate, Wilson at the RAW n -> covers 27.5%
```

**That last line is the control.** Wilson at the raw row count — the obvious thing to
write — undercovers by 68 points. The effective-sample discount is doing all the work.

Cost is a closed-form plug-in, not an importance-weighted estimate: exact to `7.1e-15`
against actually running the policy, with an interval **3.95x tighter** than IPS of the
same quantity. Only the quality estimand needs propensities. Positivity is checked rather
than assumed — under a deterministic logger the estimator **refuses 6 of 7 candidates**
and says so, instead of returning a number no reweighting could support.

## Composed with llm-gateway

```bash
route-regret gateway
```

This repo chooses a model; [llm-gateway](https://github.com/matsumoto1f-creator/llm-gateway)
serves it. Two failures that only appear once they are composed:

**Attribution.** Under a provider outage the gateway substitutes. File the quality
outcome against the model the router *requested* and the record inverts the ladder —
`haiku-4.5` reads 88.6% and `opus-5` reads 18.9%, and **neither model ran at all**. The
pooled average sees nothing (p=0.64) because the distortions have opposite signs. Quality
is attributed to `Served.model_served`, and substituted requests are excluded from policy
evaluation entirely: you cannot evaluate a choice that was not honoured.

**Shed load.** Under a capacity squeeze a naive dashboard divides spend by *intended*
requests and reports a triumph:

```
naive saving    +91.5%   (spend / intended requests)
matched saving   +0.0%   (spend / served requests, versus the same requests unsqueezed)
shed rate       +89.7%   (538 of 600 never served)
```

The 91.5% is the shed rate wearing a dollar sign. Replaying the survivors against an
unsqueezed deployment reproduces the squeezed spend to the cent.

## Status

All six phases built. **65 tests**, runs from a fresh clone with no API key.

Known caveat, found while building phase 3 and not yet resolved: `tune_to_tau` bisects to
the *cheapest admissible* point, so a tuned policy sits on the constraint by construction
and a one-sided non-inferiority bound can never clear −δ at any n. That is a property of
the tuning protocol, not of the fixture, and it is the argument for tuning to a bound
rather than to a point estimate. Every FASC figure here carries it.
