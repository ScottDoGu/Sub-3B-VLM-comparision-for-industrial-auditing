import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

# Force strict academic white background
plt.style.use('default')
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.color'] = '#e0e0e0'
plt.rcParams['grid.linestyle'] = '--'

OUT_DIR = "paper_assets/charts"
os.makedirs(OUT_DIR, exist_ok=True)

# ── Load Data ──
def load_csv(path):
    if os.path.exists(path):
        return pd.read_csv(path).dropna(subset=['Model']).set_index('Model')
    return pd.DataFrame()

df_base      = load_csv("results/baseline/metrics/aggregated_multi_run_metrics.csv")
df_cot       = load_csv("results/innovation/cot/metrics/aggregated_multi_run_metrics.csv")
df_decomp    = load_csv("results/innovation/decomposition/metrics/aggregated_multi_run_metrics.csv")
df_clahe_cot = load_csv("results/innovation/contrast_cot/metrics/aggregated_multi_run_metrics.csv")
df_clahe     = load_csv("results/innovation/contrast/metrics/aggregated_multi_run_metrics.csv")

if df_base.empty:
    print("Error: Baseline data not found.")
    exit(1)

models = list(df_base.index)

# ── CHART 1: The Baseline Danger (Grouped Bar Chart) ──
fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(len(models))
width = 0.35

lcr = [df_base.loc[m, "raw_lcr_mean"] for m in models]
fnr = [df_base.loc[m, "raw_fnr_mean"] for m in models]

ax.bar(x - width/2, lcr, width, label="Logic Compliance (LCR)", color="#2b508f", edgecolor='black', linewidth=0.5)
ax.bar(x + width/2, fnr, width, label="False Negative Rate (FNR) - DANGER", color="#d6604d", edgecolor='black', linewidth=0.5)

ax.set_xticks(x)
ax.set_xticklabels(models, rotation=15)
ax.set_ylabel("Score (0.0 to 1.0)", fontweight='bold')
ax.set_title("Fig 1: Baseline Danger\nZero-Shot Models Suffer Massive FNR Blind Spots", fontweight="bold")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "fig1_baseline_danger.png"), dpi=300, bbox_inches='tight')
plt.close()

# ── CHART 2: Evolution of Modality Collapse (5 stages) ──
if not df_decomp.empty and not df_clahe.empty and not df_cot.empty and not df_clahe_cot.empty:
    fig, ax = plt.subplots(figsize=(9, 6))
    stages = ["Zero-Shot", "CoT", "Decomp", "CLAHE\n+ CoT", "CLAHE\n+ Decomp"]
    
    colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#a65628']
    
    for i, model in enumerate(models):
        if all(model in df.index for df in [df_base, df_cot, df_decomp, df_clahe_cot, df_clahe]):
            y_points = [
                df_base.loc[model, "raw_fnr_mean"],
                df_cot.loc[model, "raw_fnr_mean"],
                df_decomp.loc[model, "raw_fnr_mean"],
                df_clahe_cot.loc[model, "raw_fnr_mean"],
                df_clahe.loc[model, "raw_fnr_mean"]
            ]
            ax.plot(range(5), y_points, marker='o', markersize=8, linewidth=1.5, alpha=0.85, color=colors[i % len(colors)], label=model, zorder=10)
            
    ax.set_xticks(range(5))
    ax.set_xticklabels(stages, fontweight='bold')
    ax.set_ylabel("False Negative Rate (FNR) - Missed Dangers", fontweight='bold')
    ax.set_title("Fig 2: Evolution of Modality Collapse Across All Interventions", pad=20, fontsize=14, fontweight='bold')
    
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., title="Models", title_fontproperties={'weight':'bold'})
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "fig2_modality_collapse.png"), dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()

# ── CHART 3: The Formatting Penalty / Scalability ──
if not df_cot.empty:
    param_sizes = {"SmolVLM": 0.6, "InternVL2": 1.0, "Janus": 1.3, "Qwen2-VL": 2.2, "Gemma 4": 2.6, "MiniCPM": 2.8}
    fig, ax = plt.subplots(figsize=(8, 5))
    for model in models:
        if model in df_cot.index and model in param_sizes:
            base_f1 = df_base.loc[model, "raw_f1_mean"]
            cot_f1  = df_cot.loc[model, "raw_f1_mean"]
            delta   = cot_f1 - base_f1
            size    = param_sizes[model]
            
            color = "#d6604d" if delta < 0 else "#2b508f"
            ax.scatter(size, delta, color=color, s=200, edgecolors='black', linewidth=0.5, zorder=5)
            ax.text(size, delta + 0.03, model, ha='center', fontsize=10, fontweight="bold", zorder=10)
            
    ax.axhline(0, color='black', linestyle='--', alpha=0.8)
    ax.set_xlabel("Model Parameters (Billions)", fontweight="bold")
    ax.set_ylabel("F1 Score Delta (CoT minus Baseline)", fontweight="bold")
    ax.set_title("Fig 3: The Formatting Penalty\nStructured Text Degrades Small Models (<1.5B)", fontweight="bold")
    
    # Expand axis limits to prevent text cutoff
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    ax.set_xlim(x_min - 0.2, x_max + 0.2)
    ax.set_ylim(y_min - 0.1, y_max + 0.1)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "fig3_formatting_penalty.png"), dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

# ── CHART 4: Edge Deployment Reality (Hardware Bubble Chart) ──
hardware_data = {
    "SmolVLM": {"vram": 0.997, "speed": 30.69}, "InternVL2": {"vram": 1.403, "speed": 9.30},
    "Janus": {"vram": 1.923, "speed": 7.91}, "Qwen2-VL": {"vram": 2.161, "speed": 7.88},
    "MiniCPM": {"vram": 4.524, "speed": 5.65}, "Gemma 4": {"vram": 6.913, "speed": 11.29}
}

fig, ax = plt.subplots(figsize=(9, 6))
for model, stats in hardware_data.items():
    peak_f1 = df_base.loc[model, "raw_f1_mean"] if model in df_base.index else 0
    if not df_decomp.empty and model in df_decomp.index:
        peak_f1 = max(peak_f1, df_decomp.loc[model, "raw_f1_mean"])
    if not df_clahe.empty and model in df_clahe.index:
        peak_f1 = max(peak_f1, df_clahe.loc[model, "raw_f1_mean"])
    if not df_cot.empty and model in df_cot.index:
        peak_f1 = max(peak_f1, df_cot.loc[model, "raw_f1_mean"])
    if not df_clahe_cot.empty and model in df_clahe_cot.index:
        peak_f1 = max(peak_f1, df_clahe_cot.loc[model, "raw_f1_mean"])
        
    s_size = stats["speed"] * 40
    ax.scatter(stats["vram"], peak_f1, s=s_size, alpha=0.7, edgecolors='black', linewidth=1.0, label=model)
    ax.text(stats["vram"], peak_f1 + 0.035, f"{model}\n({stats['speed']:.1f} i/m)", ha='center', va='bottom', fontsize=9, fontweight="bold")

ax.set_xlabel("Peak VRAM during Inference (GB)", fontweight="bold")
ax.set_ylabel("Peak Achieved F1 Score (Across all Interventions)", fontweight="bold")
ax.set_title("Fig 4: Edge Deployment Reality\nHardware Limits vs Peak Accuracy (Bubble = Inference Speed)", fontweight="bold")
ax.set_ylim(-0.1, 0.8)
ax.set_xlim(-0.5, 8.5)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "fig4_hardware_bubble.png"), dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

print(f"Generated clean plots in {OUT_DIR}/")
