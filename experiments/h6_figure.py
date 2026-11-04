"""Draw the README figure from results/h6/summary.json. Standard library only; writes one SVG."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SERIES = (("naive", "Naive probe retention", "s1"), ("best", "Controlled, best key", "s2"), ("readout", "Controlled, same readout", "s3"))
PANELS = (("cifar_class", "All seen classes compete (cifar_class)"), ("cifar_task", "Task identity given (cifar_task)"))
LAGS = (1, 2, 3, 5)
WIDTH, HEIGHT, PANEL_W, PANEL_H, TOP, LEFT, GAP = 880, 424, 340, 270, 96, 44, 64
Y_MIN, Y_MAX = -0.05, 0.8
STYLE = """
.surface { fill: #fcfcfb } .title { fill: #0b0b0b; font-weight: 600 } .text { fill: #52514e } .grid { stroke: #e4e3df } .zero { stroke: #a8a7a1 }
.s1 { stroke: #2a78d6; fill: #2a78d6 } .s2 { stroke: #eb6834; fill: #eb6834 } .s3 { stroke: #1baf7a; fill: #1baf7a }
@media (prefers-color-scheme: dark) {
  .surface { fill: #1a1a19 } .title { fill: #f0efec } .text { fill: #c3c2b7 } .grid { stroke: #33332f } .zero { stroke: #6b6a64 }
  .s1 { stroke: #3987e5; fill: #3987e5 } .s2 { stroke: #d95926; fill: #d95926 } .s3 { stroke: #199e70; fill: #199e70 }
}
text { font-family: -apple-system, 'Segoe UI', Helvetica, Arial, sans-serif; font-size: 13px }
"""


def values(entry):
    return {"naive": [entry["by_lag"][f"{lag}/{entry['budget']}"]["probe"]["naive_retention"] for lag in LAGS],
            "best": [max(entry["by_lag"][f"{lag}/{entry['budget']}"]["same"]["S"].values()) for lag in LAGS],
            "readout": [entry["by_lag"][f"{lag}/{entry['budget']}"]["same"]["S_readout"] for lag in LAGS]}


def panel(entry, title, x0):
    x = lambda lag: x0 + (lag - 1) / 4 * PANEL_W
    y = lambda v: TOP + (Y_MAX - v) / (Y_MAX - Y_MIN) * PANEL_H
    out = [f'<text class="title" x="{x0}" y="{TOP - 16}">{title}</text>']
    for tick in (0, 0.2, 0.4, 0.6, 0.8):
        out.append(f'<line class="{"zero" if tick == 0 else "grid"}" x1="{x0}" x2="{x0 + PANEL_W}" y1="{y(tick):.1f}" y2="{y(tick):.1f}"/>')
        out.append(f'<text class="text" text-anchor="end" x="{x0 - 8}" y="{y(tick) + 4:.1f}">{tick:.1f}</text>')
    for lag in LAGS:
        out.append(f'<text class="text" text-anchor="middle" x="{x(lag):.1f}" y="{TOP + PANEL_H + 20}">{lag}</text>')
    out.append(f'<text class="text" text-anchor="middle" x="{x0 + PANEL_W / 2}" y="{TOP + PANEL_H + 40}">Tasks learned afterwards</text>')
    data = values(entry)
    for key, name, css in SERIES:
        points = " ".join(f"{x(lag):.1f},{y(v):.1f}" for lag, v in zip(LAGS, data[key]))
        out.append(f'<polyline class="{css}" style="fill:none" stroke-width="2" stroke-linejoin="round" points="{points}"/>')
        for lag, v in zip(LAGS, data[key]):
            out.append(f'<circle class="{css}" cx="{x(lag):.1f}" cy="{y(v):.1f}" r="4.5"><title>{name}, {lag} later: {v:.2f}</title></circle>')
        out.append(f'<text class="text" x="{x(LAGS[-1]) + 9:.1f}" y="{y(data[key][-1]) + 4:.1f}">{data[key][-1]:.2f}</text>')
    return out


def main():
    summary = json.loads((ROOT / "results/h6/summary.json").read_text())
    alt = "Two line charts. Naive probe retention stays between about 0.5 and 0.7 as later tasks are learned, while the controlled measures fall towards zero."
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {WIDTH} {HEIGHT + 20}" width="{WIDTH}" role="img" aria-label="{alt}">', f"<style>{STYLE}</style>",
           f'<rect class="surface" width="{WIDTH}" height="{HEIGHT + 20}" rx="8"/>',
           f'<text class="title" x="{LEFT}" y="26" style="font-size:16px">A plain probe keeps reading about half or more while controlled measures fall</text>']
    legend_x = LEFT
    for _, name, css in SERIES:
        out.append(f'<line class="{css}" stroke-width="2" x1="{legend_x}" x2="{legend_x + 20}" y1="49" y2="49"/><circle class="{css}" cx="{legend_x + 10}" cy="49" r="4"/>')
        out.append(f'<text class="text" x="{legend_x + 26}" y="53">{name}</text>')
        legend_x += 44 + 7.2 * len(name)
    for i, (regime, title) in enumerate(PANELS):
        out += panel(summary[regime], title, LEFT + i * (PANEL_W + GAP))
    out.append(f'<text class="text" x="{LEFT}" y="{HEIGHT + 8}">Split CIFAR-100, 20 seeds, 2 labelled images per class. 1.0 is the old task\'s full advantage. Best key is chosen after the fact, so it favours recovery.</text>')
    out.append("</svg>\n")
    path = ROOT / "docs/figures/h6_probe_vs_memory.svg"
    path.write_text("\n".join(out))
    print(path)


if __name__ == "__main__":
    main()
