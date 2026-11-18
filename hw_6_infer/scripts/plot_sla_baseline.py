"""Plot the baseline's measured 300-second SLA load tests."""

import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/hw6-mavletova-matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DATA = json.loads((ROOT / "figures/sla_baseline_metrics.json").read_text())
RUNS = sorted(DATA["runs"], key=lambda r: r["rpm"])
LIMIT = DATA["sla_p90_limit_s"]
assert all(r["load_duration_s"] == 300 and r["error_requests"] == 0 for r in RUNS)
best = max((r for r in RUNS if r["sla_met"]), key=lambda r: r["rpm"])
first_fail = min((r for r in RUNS if r["rpm"] > best["rpm"] and not r["sla_met"]),
                 key=lambda r: r["rpm"])

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#C3CCD6", "text.color": "#263244",
    "axes.labelcolor": "#263244", "xtick.color": "#526174", "ytick.color": "#526174",
})
fig, ax = plt.subplots(figsize=(10.8, 6.8))
fig.subplots_adjust(left=0.10, right=0.97, bottom=0.17, top=0.82)
fig.text(0.10, 0.94, "Нагрузочный тест бейзлайна", fontsize=20, fontweight="bold", va="top")
fig.text(0.10, 0.885, "Qwen3-8B · vLLM 0.22.0 · A100 80 GB · 300 секунд нагрузки на точку",
         fontsize=11, color="#526174")
ax.axhspan(0, LIMIT, color="#2C8B67", alpha=0.045)
ax.axhspan(LIMIT, 24, color="#CF493B", alpha=0.045)
ax.plot([r["rpm"] for r in RUNS], [r["p90_latency_s"] for r in RUNS],
        color="#2463A6", linewidth=2.4, label="P90 E2E Latency", zorder=2)
for met, color, label in ((True, "#21805C", "SLA выполнен"), (False, "#CF493B", "SLA нарушен")):
    points = [r for r in RUNS if r["sla_met"] == met]
    ax.scatter([r["rpm"] for r in points], [r["p90_latency_s"] for r in points],
               color=color, s=50, label=label, zorder=3)
ax.axhline(LIMIT, color="#B03D35", linestyle="--", linewidth=1.5, label="Порог SLA: 15 с")
ax.scatter(best["rpm"], best["p90_latency_s"], s=220, marker="*", color="#21805C", zorder=5)
ax.scatter(first_fail["rpm"], first_fail["p90_latency_s"], s=120, marker="X", color="#CF493B", zorder=5)
for r in RUNS:
    if r not in (best, first_fail):
        ax.annotate(f'{r["p90_latency_s"]:.2f}'.replace(".", ",") + " с",
                    (r["rpm"], r["p90_latency_s"]), xytext=(0, 10), textcoords="offset points",
                    ha="center", color="#263244", fontsize=10)
ax.annotate(f'Максимум среди проверенных\n{best["rpm"]} RPM = {best["rpm"]/60:g} RPS\n'
            + f'P90 = {best["p90_latency_s"]:.2f} с'.replace(".", ","),
            (best["rpm"], best["p90_latency_s"]), xytext=(133, 20.0),
            fontsize=10, linespacing=1.5,
            arrowprops={"arrowstyle": "->", "color": "#21805C", "lw": 1.4},
            bbox={"boxstyle": "round,pad=0.6", "facecolor": "#EDF7F1", "edgecolor": "#B6DAC8"})
ax.annotate(f'Первая точка вне SLA\n{first_fail["rpm"]} RPM\n'
            + f'P90 = {first_fail["p90_latency_s"]:.2f} с'.replace(".", ","),
            (first_fail["rpm"], first_fail["p90_latency_s"]), xytext=(200, 7.0),
            fontsize=10, linespacing=1.5,
            arrowprops={"arrowstyle": "->", "color": "#CF493B", "lw": 1.4},
            bbox={"boxstyle": "round,pad=0.6", "facecolor": "#FFF0ED", "edgecolor": "#E9BDB7"})
ax.set(xlabel="Заданная нагрузка, RPM", ylabel="P90 E2E Latency, с",
       xlim=(114, 231), ylim=(0, 24), xticks=[r["rpm"] for r in RUNS])
ax.set_yticks([0, 5, 10, 15, 20])
ax.grid(axis="y", color="#DCE2E9", linewidth=0.7)
ax.set_axisbelow(True)
ax.legend(loc="lower left", frameon=False, fontsize=10)
fig.text(0.10, 0.065, "Условие: P90 < 15 с. Все запросы завершились успешно; прогрев исключён из метрик.",
         fontsize=9, color="#526174")
fig.text(0.10, 0.035,
         "Граница не уточнена между 180 и 200 RPM. Результат относится к бейзлайну; по одному прогону на точку.",
         fontsize=9, color="#526174")
fig.savefig(ROOT / "figures/sla_baseline_p90.png", dpi=180, facecolor="white")
plt.close(fig)
