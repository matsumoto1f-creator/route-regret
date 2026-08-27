"""`route-regret` — the whole bench from a fresh clone, no API key, no container."""

from __future__ import annotations

import argparse

from route_regret.bench import break_even_verify_rate, reference_model, references, run
from route_regret.fixture import MIXES, REGISTRY, WorkloadSpec, build_workload
from route_regret.metrics import by_stratum, score, self_agreement_ceiling
from route_regret.policies import Oracle, AlwaysTop, ThresholdLadder
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


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="route-regret",
                                description="A bench for LLM routing policies.")
    sub = p.add_subparsers(dest="cmd", required=True)
    for name, fn, helptext in (
        ("bench", cmd_bench, "score every policy on one traffic mix"),
        ("mixes", cmd_mixes, "what each metric does when only the traffic changes"),
        ("strata", cmd_strata, "FASC per difficulty decile — the mix-invariant view"),
        ("arithmetic", cmd_arithmetic, "the fixture-independent identities"),
    ):
        s = sub.add_parser(name, help=helptext)
        s.set_defaults(func=fn)
        s.add_argument("--n", type=int, default=8000)
        s.add_argument("--delta", type=float, default=0.03)
        s.add_argument("--verify", type=float, default=0.05)
        if name in ("bench", "strata"):
            s.add_argument("--mix", default="balanced", choices=list(MIXES))
    args = p.parse_args(argv)
    return args.func(args)
