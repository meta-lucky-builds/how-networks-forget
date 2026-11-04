"""Read-only audit of the H6 results: checks and post hoc readings for results/h6/RESULT.md. Trains nothing."""
import json
import sys
from pathlib import Path

import numpy as np

d = Path(sys.argv[1] if len(sys.argv) > 1 else "results/h6")
cal = json.loads((d / "calibration.json").read_text())
summ = json.loads((d / "summary.json").read_text())
ROOT = 20261105
rng = np.random.default_rng([ROOT, 99])
counts = rng.multinomial(20, np.full(20, 1 / 20), size=10000).astype(float)

print("== calibration")
for r, e in cal.items():
    print(r, "config", e["config"], "budget", e.get("budget"), "answerable", e["answerable"], "seeds", e["seeds"])
    print(" advantage_w0", e.get("advantage_w0"))
    print(" checks", {k: round(v["measured"], 3) for k, v in e.get("checks", {}).items()})
    print(" grid")
    for k, g in e["grid"].items():
        print("  ", k, {m: round(x, 3) for m, x in g.items()})

rows = {}
for line in (d / "confirm_jobs.jsonl").read_text().splitlines():
    item = json.loads(line)
    _, regime, seed = item["job"].split("|")
    for w, lag, task, key, n, acc in item["value"]:
        rows[(regime, int(seed), w, lag, task, key, n)] = acc
print("\n== raw rows", len(rows), "non-finite", sum(not np.isfinite(v) for v in rows.values()))
ZERO = {"none", "mask", "restore_head", "restore_body"}
seeds = list(range(800, 820))


def col(regime, lag, task, key, n, which="real"):
    n = 0 if key in ZERO else n
    return np.array([rows[(regime, s, which, lag, task, key, n)] for s in seeds])


def new(regime, lag, key, n, which="real"):
    return np.mean([col(regime, lag, f"C{i}", key, n, which) for i in range(4)], axis=0)


def ci(x):
    return [round(float(v), 3) for v in np.percentile(x, [2.5, 97.5])]


def S(regime, lag, key, n, fixed=False):
    base = 0 if fixed else lag
    return col(regime, lag, "A", key, n) - new(regime, base, key, n), col(regime, 0, "A", key, n) - new(regime, 0, key, n)


def bratio(p):
    return (counts @ p[0]) / (counts @ p[1])


print("\n== summary verdicts and rebuilt values")
for regime, e in summ.items():
    n, m = e["budget"], e["config"]["m_star"]
    print(regime, "H6a", e["probe_verdict"], "H6b", e["split_verdict"], "gates", e["gates_passed"], "same", e["same"]["label"], "fixed", e["fixed"]["label"])
    print(" fixed recoverable_ci", e["fixed"]["recoverable_ci"], "fixed access_ci", e["fixed"]["access_ci"])
    print(" matched erasure", {k: round(e["checks"]["matched_gap"][k], 3) for k in ("matched_S_readout", "matched_S_none")})
    keys = ["none", "bn_recal", "readout", "readout_early", "finetune_low", "restore_head", "restore_body"]
    if regime == "cifar_class":
        keys.insert(0, "mask")
    for lag in (1, 2, 3, 5):
        now = col(regime, lag, "A", "readout", n)
        then = col(regime, 0, "A", "readout", n)
        naive = (counts @ now) / (counts @ then)
        pairs = {k: S(regime, lag, k, n) for k in ("readout", "readout_early", "finetune_low")}
        best = np.max([bratio(v) for v in pairs.values()], axis=0)
        none = S(regime, lag, "none", n)
        point = {k: v[0].mean() / v[1].mean() for k, v in pairs.items()}
        top = max(point.values())
        chance = (now.mean() - 0.1) / (then.mean() - 0.1)
        print(f" lag {lag}: naive {now.mean() / then.mean():.3f} chance-corrected {chance:.3f} S_best {top:.3f} {ci(best)}"
              f" naive-S_best {now.mean() / then.mean() - top:.3f} {ci(naive - best)}"
              f" S_none {none[0].mean() / none[1].mean():.3f}"
              f" S_none-S_readout {none[0].mean() / none[1].mean() - point['readout']:.3f} {ci(bratio(none) - bratio(pairs['readout']))}")
        print("   same S", {k: round(float(S(regime, lag, k, n)[0].mean() / S(regime, lag, k, n)[1].mean()), 3) for k in keys})
        print("   fixed S", {k: round(float(S(regime, lag, k, n, True)[0].mean() / S(regime, lag, k, n, True)[1].mean()), 3) for k in keys})
        print("   acc A", {k: round(float(col(regime, lag, "A", k, n).mean()), 3) for k in keys})
        print("   acc new", {k: round(float(new(regime, lag, k, n).mean()), 3) for k in keys})
    print("   acc A w0", {k: round(float(col(regime, 0, "A", k, n).mean()), 3) for k in keys})
    print("   acc new w0", {k: round(float(new(regime, 0, k, n).mean()), 3) for k in keys})
    ref = "mask" if regime == "cifar_class" else "none"
    for k in ("bn_recal", "restore_head"):
        for lag in (m,):
            diff = bratio(S(regime, lag, k, n)) - bratio(S(regime, lag, ref, n))
            print(f"   {k} minus {ref} at m*: {diff.mean():.3f} {ci(diff)}")
    now, then = col(regime, m, "A", "readout", n), col(regime, 0, "A", "readout", n)
    pos = (now / then - (now - new(regime, m, "readout", n)) / (then - new(regime, 0, "readout", n))).tolist()
    print("   per-seed H6a gap min/median/max", round(min(pos), 3), round(float(np.median(pos)), 3), round(max(pos), 3), "seeds positive", sum(p > 0 for p in pos))
