from pathlib import Path
import csv
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ML = ROOT / "results" / "ml"
OUT = ROOT / "figures" / "poster"
OUT.mkdir(parents=True, exist_ok=True)

image_path = ML / "05_ml_50pct_cfd_vs_prediction.png"
metrics_path = ML / "ml_geometry_holdout_metrics.csv"

if not image_path.exists():
    raise FileNotFoundError(image_path)

if not metrics_path.exists():
    raise FileNotFoundError(metrics_path)

with open(metrics_path, newline="") as f:
    rows = list(csv.DictReader(f))

def get_rmse(model):
    match = [
        row for row in rows
        if row["held_out_geometry"] == "50%"
        and row["model"] == model
    ]
    if len(match) != 1:
        raise ValueError(f"Expected one result for {model}")
    return float(match[0]["RMSE_mm_s"])

physics = get_rmse("Physics baseline")
geometry = get_rmse("Geometry-only ML")
guided = get_rmse("Physics-guided ML")

improvement = 100 * (physics - guided) / physics

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 12,
    "savefig.facecolor": "white"
})

fig = plt.figure(figsize=(18, 10), facecolor="white")

gs = fig.add_gridspec(
    3, 2,
    width_ratios=[1.7, 1],
    height_ratios=[0.55, 5.5, 1.35],
    left=0.04,
    right=0.96,
    top=0.96,
    bottom=0.06,
    wspace=0.08,
    hspace=0.14
)

# Header
ax_title = fig.add_subplot(gs[0, :])
ax_title.axis("off")

ax_title.text(
    0, 0.78,
    "HEMODYN AI  |  MACHINE-LEARNING RESULTS",
    fontsize=23,
    weight="bold",
    transform=ax_title.transAxes
)

ax_title.text(
    0, 0.12,
    "Physics-guided residual learning for CFD axial-velocity prediction",
    fontsize=14,
    color="#555555",
    transform=ax_title.transAxes
)

# Existing actual CFD-versus-ML visualization
ax_img = fig.add_subplot(gs[1, 0])
ax_img.imshow(plt.imread(image_path))
ax_img.axis("off")

# Quantitative findings
ax_results = fig.add_subplot(gs[1, 1])
ax_results.set_xlim(0, 1)
ax_results.set_ylim(0, 1)
ax_results.axis("off")

ax_results.text(
    0.05, 0.96,
    "Held-out 50% stenosis",
    fontsize=19,
    weight="bold",
    va="top"
)

ax_results.text(
    0.05, 0.87,
    "Trained on healthy and 30% CFD geometries",
    fontsize=11,
    color="#555555"
)

ax_results.text(
    0.05, 0.76,
    f"{improvement:.1f}%",
    fontsize=40,
    weight="bold",
    color="#196B58"
)

ax_results.text(
    0.05, 0.69,
    "lower RMSE than physics baseline",
    fontsize=12
)

bar_ax = ax_results.inset_axes([0.05, 0.29, 0.90, 0.32])

models = [
    "Physics baseline",
    "Geometry-only ML",
    "Physics-guided ML"
]

values = [physics, geometry, guided]

bars = bar_ax.barh(
    np.arange(len(models)),
    values,
    color=["#8798A8", "#C0C8D1", "#267D76"],
    height=0.58
)

bar_ax.set_yticks(np.arange(len(models)))
bar_ax.set_yticklabels(models)
bar_ax.invert_yaxis()
bar_ax.set_xlim(0, max(values) * 1.35)
bar_ax.set_xlabel("Axial velocity RMSE (mm/s)", fontsize=10)
bar_ax.tick_params(axis="both", labelsize=10)
bar_ax.spines[["top", "right"]].set_visible(False)
bar_ax.grid(axis="x", alpha=0.15)
bar_ax.set_axisbelow(True)

for bar, value in zip(bars, values):
    bar_ax.text(
        value + 0.5,
        bar.get_y() + bar.get_height() / 2,
        f"{value:.2f}",
        va="center",
        fontsize=11,
        weight="bold"
    )

ax_results.text(
    0.05, 0.19,
    "Regional RMSE improvement:",
    fontsize=12,
    weight="bold"
)

ax_results.text(
    0.05, 0.13,
    "Stenosis throat: 19.9%",
    fontsize=12
)

ax_results.text(
    0.05, 0.07,
    "Post-stenotic region: 24.8%",
    fontsize=12
)

# Scientific interpretation
ax_footer = fig.add_subplot(gs[2, :])
ax_footer.axis("off")

ax_footer.plot(
    [0, 1], [0.96, 0.96],
    transform=ax_footer.transAxes,
    color="#BBBBBB",
    linewidth=1
)

ax_footer.text(
    0.01, 0.74,
    "KEY FINDING",
    fontsize=12,
    weight="bold",
    transform=ax_footer.transAxes
)

ax_footer.text(
    0.01, 0.46,
    "Physics-guided ML reduced axial-velocity prediction error "
    "on the held-out 50% stenosis geometry.",
    fontsize=12,
    transform=ax_footer.transAxes
)

ax_footer.text(
    0.01, 0.20,
    "LIMITATION: Reverse-flow detection was 0/218 CFD points. "
    "Performance varied across geometry holdouts; "
    "broader generalization remains unverified.",
    fontsize=11,
    color="#555555",
    transform=ax_footer.transAxes
)

output = OUT / "07_hemodyn_ml_poster_panel.png"

fig.savefig(
    output,
    dpi=250,
    facecolor="white",
    bbox_inches="tight"
)

plt.close(fig)

print("\n=== HEMODYN AI POSTER PANEL ===")
print("Figure:", output)
print(f"Physics RMSE: {physics:.3f} mm/s")
print(f"Geometry ML RMSE: {geometry:.3f} mm/s")
print(f"Physics-guided RMSE: {guided:.3f} mm/s")
print(f"Improvement: {improvement:.2f}%")
print("\nPoster panel generated successfully.")
