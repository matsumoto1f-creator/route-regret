"""`route-regret` — the whole bench from a fresh clone, no API key, no container."""

from __future__ import annotations

import argparse

from route_regret.bench import break_even_verify_rate, reference_model, references, run
from route_regret.fixture import MIXES, REGISTRY, WorkloadSpec, build_workload
from route_regret.metrics import by_stratum, score, self_agreement_ceiling
from route_regret.policies import Oracle, AlwaysTop, ThresholdLadder

UNBOUNDED_TPM = 10 ** 9
from route_regret.report import DEFAULT_LADDER, bench_mix, mix_invariance_table, spread


def cmd_bench(args) -> int:
    ref, scores = bench_mix(args.mix, n=args.n, verify_rate=args.verify, delta=args.delta)
    print(f"\nmix={args.mix}  n={args.n}  verify_rate={args.verify:.0%}  "
          f"non-inferiority margin={args.delta:.0%}")
    print(f"quality bar: violations must stay at or below {scores[0].tau:.1%} "
          f"(always-frontier {ref.top_violation:.1%} + {args.delta:.0%})\n")
    print(f"{'policy':<20}{'cost $':>12}{'violation':>11}{'naive sav':>12}{'FASC@d':>17}")
    print("-" * 72)
    for s in scores:
        print(s.line())
    by = {s.policy: s for s in scores}
    r, b = by["threshold_ladder"].fasc_at_delta, by["content_blind"].fasc_at_delta
    if r is not None and b is not None:
        print(f"\nrouter {r:.1%} vs content-blind {b:.1%} at MATCHED quality "
              f"-> margin {100*(r-b):+.1f}pp")
    print("\nnaive savings is printed as the control, not as a result: the policy that "
          "\nmaximises it is always_cheapest, which fails "
          f"{by['always_cheapest'].violation:.0%} of requests.")
    return 0


def cmd_mixes(args) -> int:
    table = mix_invariance_table(n=args.n, verify_rate=args.verify, delta=args.delta)
    print(f"\nSame router on every row. Only the traffic changes.\n")
    print(f"{'mix':<14}{'naive savings':>15}{'FASC@d':>12}{'violation':>11}")
    print("-" * 52)
    nv, fa = [], []
    for mix, sc in table.items():
        s = sc["threshold_ladder"]
        nv.append(s.naive_savings)
        fa.append(s.fasc_at_delta)
        fd = "n/a" if s.fasc_at_delta is None else f"{s.fasc_at_delta:.1%}"
        print(f"{mix:<14}{s.naive_savings:>14.1%}{fd:>12}{s.violation:>11.1%}")
    print("-" * 52)
    print(f"{'spread':<14}{100*spread(nv):>13.1f}pp{100*spread(fa):>10.1f}pp")
    print("\nNeither is mix-invariant and this repo does not claim otherwise — no scalar"
          "\nsummary of a router can be. FASC@d is the tighter of the two, and the"
          "\nquantity that IS invariant is per-stratum: `route-regret strata`.")
    return 0


def cmd_strata(args) -> int:
    spec = WorkloadSpec(mix=args.mix, n=args.n)
    cases = build_workload(spec)
    d_of = {c.case_id: c.difficulty for c in cases}
    ref = references(cases, spec)
    from route_regret.report import tune_to_tau
    policy, led = tune_to_tau(lambda off: ThresholdLadder(DEFAULT_LADDER, offset=off),
                              cases, spec, ref, args.delta, args.verify)
    top = run(AlwaysTop(), cases, spec)
    orc = run(Oracle(), cases, spec)
    print(f"\nFASC by difficulty decile — mix={args.mix}\n")
    for s in by_stratum(led.rows, top.rows, orc.rows, lambda rid: d_of[rid]):
        print(s.line())
    print("\nStrata whose achievable savings are negligible report not_estimable rather"
          "\nthan a percentage: dividing by a denominator near zero manufactures findings.")
    return 0


def cmd_arithmetic(args) -> int:
    """The numbers that do not depend on the fixture at all."""
    ref = reference_model()
    print(f"\nReference model (chosen by CAPABILITY, not price): {ref.name}\n")
    print("Break-even verification rate  v* = 1 - c_cheap/c_ref")
    print("  above this rate, routing stops saving money — arithmetic, not a benchmark\n")
    print(f"{'cheap model':<16}{'$/req':>10}{'v* no judge':>14}{'v* with judge':>16}")
    print("-" * 56)
    for m in REGISTRY:
        if m.name == ref.name:
            continue
        print(f"{m.name:<16}{m.cost(800,400):>10.6f}"
              f"{break_even_verify_rate(m, ref, with_judge=False):>13.1%}"
              f"{break_even_verify_rate(m, ref):>16.1%}")
    print("\nThe judge is itself an expensive model call. Pricing it roughly halves the"
          "\nrate at which verification is still worth doing — the original spec never"
          "\ncounted it, and specified verifying EVERY routed request, which is on the"
          "\nwrong side of this line for every model pair in the registry.\n")
    print("Instrument ceiling  p^2 + (1-p)^2/(L-1) — two samples of the SAME model:")
    for p in (0.99, 0.95, 0.90, 0.85):
        c = self_agreement_ceiling(p)
        print(f"  a model at {p:.0%} accuracy agrees with ITSELF {c:.1%} of the time"
              f"  -> {1-c:.1%} phantom 'routing failures'")
    print("\nEvery parity figure must be read as a distance from that ceiling, never"
          "\nfrom 100%.")
    return 0


def cmd_classify(args) -> int:
    """The classifier as ONE ENTRANT, not the product."""
    from route_regret.classifier import CostSensitiveRouter, faceoff, train
    # train() takes the EVALUATION spec and derives a disjoint training workload from
    # seed_tag. It refuses if the two tags match, because identical case ids make
    # held-out accuracy a lookup -- a guard that caught this call site being written
    # backwards.
    test_spec = WorkloadSpec(mix=args.mix, n=args.n, seed_tag="test")
    cases = build_workload(test_spec)
    ref = references(cases, test_spec)
    from route_regret.report import tune_to_tau

    ensemble = train(test_spec, n=args.n, seed_tag="train")
    print(f"\ntrained on n={args.n} (seed_tag=train), scored on a DISJOINT n={args.n} "
          f"(seed_tag=test)\n")
    for card in REGISTRY[:2]:
        print("  " + str(ensemble.report(card, cases, test_spec)).replace("\n", "\n  "))
    # Computed, not typed. Writing the illustrative numbers by hand is how a narration
    # line becomes an argument wearing a result's clothes.
    r0 = ensemble.report(REGISTRY[0], cases, test_spec)
    print(f"\n  Never a bare accuracy scalar: {r0.accuracy:.0%} against a "
          f"{r0.majority_baseline:.0%} majority class is a kappa of {r0.kappa:.2f}, and"
          "\n  the two off-diagonal cells are not the same mistake.\n")

    clf, clf_led = tune_to_tau(
        lambda off: CostSensitiveRouter(ensemble, offset=off), cases, test_spec,
        ref, args.delta, args.verify)
    hand, hand_led = tune_to_tau(
        lambda off: ThresholdLadder(DEFAULT_LADDER, offset=off), cases, test_spec,
        ref, args.delta, args.verify)
    for led in (clf_led, hand_led):
        print(score(led, ref, delta=args.delta).line())
    f = faceoff(clf_led, hand_led, cases, ref, delta=args.delta,
                challenger="classifier", incumbent="threshold_ladder")
    print(f"\n  classifier over the hand rule: {100*f.point:+.1f}pp "
          f"[{100*f.low:+.1f}, {100*f.high:+.1f}] at matched quality")
    print("  A trained model that cannot beat two thresholds means the training set was"
          "\n  ceremony. See tests/test_classifier.py for where that margin comes from —"
          "\n  most of it is features the hand rule was never given, not the fitting.")
    return 0


def cmd_offpolicy(args) -> int:
    """Estimate a policy you did not run, from logs of the one you did."""
    from route_regret.classifier import CostSensitiveRouter, train
    from route_regret.offpolicy import EpsilonExploring, measure_coverage

    print(f"\nMeasuring INTERVAL COVERAGE over {args.seeds} independent workloads.")
    print("The headline is not a point estimate — it is whether a nominal 95% interval")
    print("actually contains the truth 95% of the time. Truth comes from running the")
    print("candidate for real on a large disjoint draw.\n")
    rep = measure_coverage(
        lambda: EpsilonExploring(ThresholdLadder(DEFAULT_LADDER), 0.15),
        lambda: ThresholdLadder(DEFAULT_LADDER, offset=0.12),
        seeds=args.seeds, n=args.n, nominal=0.95)
    iv = rep.coverage_interval()
    verdict = ("CALIBRATED" if iv.low <= rep.nominal <= iv.high
               else "UNDER-covering" if rep.covered_fraction < rep.nominal
               else "OVER-covering")
    print(f"  nominal {rep.nominal:.0%}  ->  measured {rep.covered_fraction:.1%} "
          f"[{iv.low:.1%}, {iv.high:.1%}]   {verdict}")
    print(f"  intervals are {rep.implied_width_scale:.2f}x the width that would hit "
          f"nominal exactly")
    print(f"  mean width {rep.mean_width:.1%}   mean |error| {rep.mean_absolute_error:.1%}")
    print(f"  reweighting leaves {rep.mean_n_effective:.0f} of {rep.n} effective "
          f"observations ({rep.mean_n_effective/rep.n:.1%} of the log survives)")
    print(f"  same point estimate, Wilson at the RAW n -> covers "
          f"{rep.naive_covered_fraction:.1%}")
    print(f"\n  That last line is the control. Wilson at the raw row count is the obvious"
          f"\n  thing to write, and it undercovers by "
          f"{100*(rep.covered_fraction - rep.naive_covered_fraction):.0f} points. The"
          f"\n  effective-sample discount is doing all the work.")
    print(f"\n  Truth is not a formula: the candidate was actually run on {rep.truth_n:,} "
          f"independent\n  cases, disjoint from every estimation draw.")
    return 0


def cmd_gateway(args) -> int:
    """Composition with llm-gateway: this repo chooses, the gateway serves."""
    from route_regret.gateway import (Deployment, attribute, audit_shed_load, evaluable)

    spec = WorkloadSpec(mix=args.mix, n=args.n)
    cases = build_workload(spec)
    policy = ThresholdLadder(DEFAULT_LADDER)

    print("\n--- attribution under a provider outage ---\n")
    dep = Deployment.build(outages={"anthropic": [(0.0, 1e9)]})
    calls = dep.serve(cases, policy, spec)
    print("  filed against the model that actually RAN (correct):")
    for a in attribute(calls, key="served").values():
        print("    " + a.line())
    print("\n  filed against the model the router REQUESTED (the defect):")
    for a in attribute(calls, key="requested").values():
        print("    " + a.line())
    ev = evaluable(calls)
    print(f"\n  {ev.line()}")
    print("  A substituted request cannot evaluate the choice that was not honoured.")

    print("\n--- the shed-load trap ---\n")
    audit = audit_shed_load(
        lambda squeezed: Deployment.build(
            team_tokens_per_minute=8000 if squeezed else UNBOUNDED_TPM),
        cases, policy, spec)
    for line in audit.lines():
        print("  " + line)
    print(f"\n  naive saving {audit.naive_saving:+.1%} — which is the shed rate wearing a"
          f"\n  dollar sign. Replaying the survivors against an unsqueezed deployment"
          f"\n  reproduces the squeezed spend exactly: matched saving "
          f"{audit.matched_saving:+.1%}.")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="route-regret",
                                description="A bench for LLM routing policies.")
    sub = p.add_subparsers(dest="cmd", required=True)
    for name, fn, helptext in (
        ("bench", cmd_bench, "score every policy on one traffic mix"),
        ("mixes", cmd_mixes, "what each metric does when only the traffic changes"),
        ("strata", cmd_strata, "FASC per difficulty decile — the mix-invariant view"),
        ("arithmetic", cmd_arithmetic, "the fixture-independent identities"),
        ("classify", cmd_classify, "the trained classifier as one entrant on the bench"),
        ("offpolicy", cmd_offpolicy, "measured coverage of the off-policy estimator"),
        ("gateway", cmd_gateway, "attribution and shed-load, composed with llm-gateway"),
    ):
        s = sub.add_parser(name, help=helptext)
        s.set_defaults(func=fn)
        s.add_argument("--n", type=int, default=8000)
        s.add_argument("--delta", type=float, default=0.03)
        s.add_argument("--verify", type=float, default=0.05)
        if name in ("bench", "strata", "classify", "gateway"):
            s.add_argument("--mix", default="balanced", choices=list(MIXES))
        if name == "offpolicy":
            s.add_argument("--seeds", type=int, default=120)
    args = p.parse_args(argv)
    return args.func(args)
