"""Build report figures from the saved AIPerf sweep summary."""

import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/hw6-mavletova-matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "figures"
DATA = json.loads((FIGURES / "speculative_sweep_metrics.json").read_text())
RUNS = DATA["runs"]
RPMS = sorted({run["rpm"] for run in RUNS})
COLORS = {"baseline": "#2463A6", "eagle3-k2": "#D85B30"}
LABELS = {"baseline": "Бейзлайн — без спекдека", "eagle3-k2": "EAGLE-3 — 2 токена"}

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.labelcolor": "#263244",
    "text.color": "#263244",
    "xtick.color": "#526174",
    "ytick.color": "#526174",
    "axes.edgecolor": "#C3CCD6",
    "savefig.facecolor": "white",
})


def series(configuration, metric):
    return [next(r[metric] for r in RUNS
                 if r["configuration"] == configuration and r["rpm"] == rpm)
            for rpm in RPMS]


def make_plot(metric, title, ylabel, output_name, ylim):
    baseline = series("baseline", metric)
    eagle = series("eagle3-k2", metric)
    transitions = [(RPMS[i - 1], RPMS[i]) for i in range(1, len(RPMS))
                   if eagle[i - 1] < baseline[i - 1] and eagle[i] >= baseline[i]]
    assert len(transitions) == 1, transitions
    lower, upper = transitions[0]
    fig, ax = plt.subplots(figsize=(10.8, 6.8))
    fig.subplots_adjust(left=0.10, right=0.97, bottom=0.16, top=0.82)
    fig.text(0.10, 0.94, title, fontsize=19, fontweight="bold", va="top")
    fig.text(0.10, 0.885, "Qwen3-8B · vLLM 0.22.0 · одна A100 80 GB", fontsize=11, color="#526174")
    ax.axvspan(lower, upper, color="#E8B04C", alpha=0.14, zorder=0)
    for configuration, marker in (("baseline", "o"), ("eagle3-k2", "s")):
        ax.plot(RPMS, series(configuration, metric), marker=marker,
                color=COLORS[configuration], label=LABELS[configuration],
                linewidth=2.4, markersize=6, zorder=3)
    ax.set(xlabel="Заданная нагрузка, RPM", ylabel=ylabel,
           xlim=(105, 515), ylim=ylim)
    ax.set_xticks(RPMS)
    ax.grid(axis="y", color="#DCE2E9", alpha=0.8, linewidth=0.7)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", frameon=False, fontsize=11)
    ax.text(0.025, 0.72,
            f"Наблюдаемый переход: {lower}–{upper} RPM\n"
            f"С {upper} RPM EAGLE-3 медленнее\nво всех измеренных точках",
            transform=ax.transAxes, fontsize=10, linespacing=1.5,
            bbox={"boxstyle": "round,pad=0.65", "facecolor": "#FFF6E5", "edgecolor": "none"})

    inset = ax.inset_axes([0.55, 0.12, 0.39, 0.31])
    inset.set_facecolor("#F8FAFC")
    zoom_rpms = [rpm for rpm in RPMS if rpm <= 200]
    for configuration, marker in (("baseline", "o"), ("eagle3-k2", "s")):
        vals = series(configuration, metric)[:len(zoom_rpms)]
        inset.plot(zoom_rpms, vals, marker=marker, color=COLORS[configuration],
                   linewidth=1.8, markersize=4)
        for i, (rpm, value) in enumerate(zip(zoom_rpms, vals)):
            # Offset labels to opposite sides of the two curves at each RPM.
            baseline_below = baseline[i] < eagle[i]
            below = baseline_below if configuration == "baseline" else not baseline_below
            inset.annotate(f"{value:.2f}".replace(".", ","), (rpm, value),
                           xytext=(0, -13 if below else 7), textcoords="offset points",
                           ha="center", fontsize=8, color=COLORS[configuration])
    low = min(series(c, metric)[0] for c in COLORS)
    high = max(max(series(c, metric)[:len(zoom_rpms)]) for c in COLORS)
    inset.set(xlim=(108, 212), ylim=(low - 1.8, high + 2.0), xticks=zoom_rpms)
    inset.set_title("120–200 RPM крупнее · задержка, с", fontsize=9, pad=7)
    inset.tick_params(labelsize=8)
    inset.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:g}".replace(".", ",")))
    inset.grid(axis="y", color="#DCE2E9", linewidth=0.6)

    fig.text(0.10, 0.055,
             "640 запросов на прогон; прогрев исключён. Маркеры — измерения, линии соединяют соседние точки.",
             fontsize=9, color="#526174")
    fig.text(0.10, 0.027,
             "Задержки рассчитаны по успешным запросам. Точная нагрузка смены преимущества не измерена.",
             fontsize=9, color="#526174")
    fig.savefig(FIGURES / output_name, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    assert len(RUNS) == len(RPMS) * len(COLORS) == 20
    assert all(r["successful_requests"] + r["error_requests"] == r["total_requests"] == 640 for r in RUNS)
    make_plot("mean_latency_s", "Среднее время полного ответа",
              "Mean E2E Latency, с", "speculative_mean_latency.png", (0, 85))
    make_plot("p90_latency_s", "P90 времени полного ответа",
              "P90 E2E Latency, с", "speculative_p90_latency.png", (0, 115))
