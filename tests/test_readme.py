"""The README's headline numbers and links must match what is stored in the repo."""
import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
README = " ".join((ROOT / "README.md").read_text(encoding="utf-8").split())
H5 = json.loads((ROOT / "results/h5/summary.json").read_text())
H6 = json.loads((ROOT / "results/h6/summary.json").read_text())


def f(x):
    return f"{x:.2f}"


def interval(ci):
    return f"[{f(ci[0])}, {f(ci[1])}]"


@pytest.mark.parametrize("regime, name", [("cifar_class", "All seen classes compete"), ("cifar_task", "Task identity given")])
def test_probe_table(regime, name):
    t = H6[regime]["probe_test"]
    assert H6[regime]["probe_verdict"] == "probes overstate" and H6[regime]["gates_passed"]
    assert f"| {name} | {f(t['naive_retention'])} | {f(t['S_readout'])} | **{f(t['gap'])}** {interval(t['gap_ci'])} |" in README


def test_probe_scores_after_five_tasks():
    t = H6["cifar_class"]["by_lag"]["5/2"]["probe"]
    assert f"**{t['acc_A_later']:.3f} on the trained task and {t['acc_new_later']:.3f} on never-seen tasks**" in README
    assert f"still says {f(t['naive_retention'])}" in README


@pytest.mark.parametrize("regime, lag", [("cifar_class", 1), ("cifar_class", 5), ("cifar_task", 1), ("cifar_task", 5)])
def test_gap_against_strongest_key(regime, lag):
    v = H6[regime]["by_lag"][f"{lag}/2"]
    assert f(v["probe"]["naive_retention"] - max(v["same"]["S"].values())) in README


def test_mnist_rows():
    for regime, row in (("perm_class", "tasks compete for outputs | 1 task"), ("perm_shared", "shared outputs | 5 tasks")):
        p = H5[regime]["primary"]
        assert f"{row} | {f(p['access'])} {interval(p['access_ci'])} | {f(p['unrecovered'])} |" in README
    assert f"0.46 access on one and {f(H5['perm_shared']['fixed_baseline']['access'])} on the other" in README
    gaps = [H5[r]["naive_against_controlled"]["naive_minus_S"] for r in ("perm_shared", "perm_class")]
    assert f"({f(gaps[0])} and {f(gaps[1])})" in README


def test_cifar_rows():
    same = H6["cifar_class"]["same"]
    assert f"all seen classes compete | 1 task | {f(same['access'])} {interval(same['access_ci'])} | {f(same['unrecovered'])} |" in README
    assert f"task identity given | 2 tasks | no valid reading | {f(H6['cifar_task']['same']['unrecovered'])} |" in README
    assert all(H6[r]["split_verdict"] == "mostly unrecovered" for r in H6)


def test_budget_labels():
    labels = [v.get("verdict") or v.get("label") for v in H5["perm_shared"]["by_budget"].values()]
    assert labels == ["no majority", "mostly repair", "mostly access", "mostly access"]
    assert {v.get("verdict") or v.get("label") for v in H5["perm_class"]["by_budget"].values()} == {"mostly access"}


def test_local_links_exist():
    links = re.findall(r"\]\(([^)#\s]+)", (ROOT / "README.md").read_text(encoding="utf-8"))
    local = [link for link in links if not link.startswith("http")]
    assert local and not [link for link in local if not (ROOT / link).exists()]
