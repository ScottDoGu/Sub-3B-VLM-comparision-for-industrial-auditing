import os
import sys
import argparse
import pandas as pd
import scipy.stats as stats

sys.path.append(os.path.join(os.path.dirname(__file__)))
from metrics import read_csv_robust

def get_image_lcr_results(df):
    """
    Returns a dictionary mapping image_id -> boolean (True if model was fully LCR compliant, False otherwise)
    Uses the same logic as metrics.compute_lcr but returns image-level success/failure pairs.
    """
    pair_map = {}
    for _, row in df.iterrows():
        img  = str(row.get("image_id", "")).strip()
        rule = str(row.get("rule_id", "")).strip() 
        verdict = str(row.get("model_verdict", "")).strip()
        expected = str(row.get("expected_verdict", "")).strip()

        if not img:
            continue
            
        if img not in pair_map:
            pair_map[img] = {}
        pair_map[img][rule] = (verdict, expected)

    img_results = {}
    for img, rules in pair_map.items():
        if "Rule A" not in rules or "Rule B" not in rules:
            continue
        verdict_a, expected_a = rules["Rule A"]
        verdict_b, expected_b = rules["Rule B"]
        ok = (verdict_a == expected_a) and (verdict_b == expected_b)
        img_results[img] = ok
        
    return img_results

def main():
    parser = argparse.ArgumentParser(description="Compute McNemar's Test for Intervention Efficacy.")
    parser.add_argument("--runs", type=int, default=3, help="Number of runs to aggregate")
    parser.add_argument("--base_mode", default="baseline", help="The mode to use as the baseline for comparison (default: baseline)")
    parser.add_argument("--intervention", choices=["cot", "decomp", "contrast", "contrast_cot"], required=True)
    args = parser.parse_args()

    models = ["smolvlm", "internvl2", "janus", "qwen2_vl", "minicpm", "gemma4_e2b"]
    model_display_names = {
        "smolvlm": "SmolVLM",
        "internvl2": "InternVL2",
        "janus": "Janus",
        "qwen2_vl": "Qwen2-VL",
        "minicpm": "MiniCPM",
        "gemma4_e2b": "Gemma 4"
    }

    # Output formatting logic mimicking multi_run_metrics.py
    MODE_CONFIG = {
        "baseline":     ("results/baseline/parsed",                        ""),
        "cot":          ("results/innovation/cot/parsed",                  "_cot"),
        "decomp":       ("results/innovation/decomposition/parsed",        "_decomp"),
        "contrast":     ("results/innovation/contrast/parsed",             "_contrast"),
        "contrast_cot": ("results/innovation/contrast_cot/parsed",         "_contrast_cot"),
    }
    
    baseline_dir, base_suffix = MODE_CONFIG[args.base_mode]
    inv_dir, suffix = MODE_CONFIG[args.intervention]
    
    print(f"\n==========================================================================")
    print(f" McNemar's Test: {args.base_mode.upper()} vs. {args.intervention.upper()} (Runs Aggregated: {args.runs})")
    print(f"==========================================================================\n")

    results_table = []

    for model_key in models:
        n00, n11, b, c = 0, 0, 0, 0
        valid_pairs_total = 0
        total_baseline_passes = 0
        total_inv_passes = 0
        
        for i in range(1, args.runs + 1):
            base_file = os.path.join(baseline_dir, f"{model_key}{base_suffix}_run_{i}_parsed.csv")
            inv_file = os.path.join(inv_dir, f"{model_key}{suffix}_run_{i}_parsed.csv")
            
            if not os.path.exists(base_file) or not os.path.exists(inv_file):
                continue
                
            base_df = read_csv_robust(base_file)
            inv_df = read_csv_robust(inv_file)
            
            base_res = get_image_lcr_results(base_df)
            inv_res = get_image_lcr_results(inv_df)
            
            # Match strictly on the same unique images
            for img, base_ok in base_res.items():
                if img in inv_res:
                    inv_ok = inv_res[img]
                    valid_pairs_total += 1
                    
                    if base_ok: total_baseline_passes += 1
                    if inv_ok: total_inv_passes += 1
                    
                    if base_ok and inv_ok:
                        n11 += 1
                    elif not base_ok and not inv_ok:
                        n00 += 1
                    elif base_ok and not inv_ok:
                        b += 1  # Passed baseline, Failed intervention (Regression)
                    elif not base_ok and inv_ok:
                        c += 1  # Failed baseline, Passed intervention (Improvement)
        
        if valid_pairs_total > 0:
            # SciPy's binomtest computes the exact binomial p-value.
            # We are testing if the shifts (b and c) are distributed with purely chance (p=0.5).
            # b + c is the number of 'discordant' pairs.
            discordant_total = b + c
            if discordant_total == 0:
                p_value = 1.0
            else:
                try:
                    # In newer SciPy versions, this is binomtest. Older versions use binom_test.
                    # We will try binomtest first, fallback to binom_test.
                    if hasattr(stats, 'binomtest'):
                        res = stats.binomtest(min(b, c), discordant_total, 0.5, alternative='two-sided')
                        p_value = res.pvalue
                    else:
                        p_value = stats.binom_test(min(b, c), discordant_total, 0.5, alternative='two-sided')
                except Exception as e:
                    p_value = 1.0 # Fallback
            
            # Format LCR as percentage
            base_lcr_pct = (total_baseline_passes / valid_pairs_total) * 100
            inv_lcr_pct = (total_inv_passes / valid_pairs_total) * 100
            
            sig_marker = "*" if p_value < 0.05 else ""
            
            results_table.append({
                "Model": model_display_names[model_key],
                "N (Pairs)": valid_pairs_total,
                "Base LCR %": f"{base_lcr_pct:.1f}%",
                "Inv LCR %": f"{inv_lcr_pct:.1f}%",
                "Regressed(b)": b,
                "Improved(c)": c,
                "p-value": f"{p_value:.4f}{sig_marker}"
            })
            
    if not results_table:
        print("No valid paired data found to test.")
        return
        
    df_results = pd.DataFrame(results_table)
    print(df_results.to_markdown(index=False))
    print("\n* indicates p < 0.05 (Statistically Significant Change)")
    print("Regressed(b) = Images where Baseline passed but Intervention failed.")
    print("Improved(c) = Images where Baseline failed but Intervention passed.\n")
    
    # Save Output
    out_dir = os.path.join("results", "metrics")
    os.makedirs(out_dir, exist_ok=True)
    out_name = f"mcnemar_{args.intervention}_significance.csv"
    out_path = os.path.join(out_dir, out_name)
    df_results.to_csv(out_path, index=False)
    print(f"Results saved to: {out_path}")

if __name__ == "__main__":
    main()
