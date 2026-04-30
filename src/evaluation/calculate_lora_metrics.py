import pandas as pd
import numpy as np
import os
import re

# Parse logic
def parse_response(response: str) -> str:
    response = str(response).strip().lower()
    if 'unsafe' in response:
        return 'UNSAFE'
    elif 'safe' in response:
        return 'SAFE'
    return 'NO_VERDICT'

def compute_metrics_for_df(df: pd.DataFrame) -> dict:
    df['model_verdict'] = df['model_response'].apply(parse_response)
    df['expected_verdict'] = df['expected_verdict'].astype(str).str.strip().str.upper()
    
    # F1 Score
    scored = df[df["model_verdict"] != "NO_VERDICT"].copy()
    tp = ((scored["model_verdict"] == "UNSAFE") & (scored["expected_verdict"] == "UNSAFE")).sum()
    fp = ((scored["model_verdict"] == "UNSAFE") & (scored["expected_verdict"] == "SAFE")).sum()
    fn = ((scored["model_verdict"] == "SAFE") & (scored["expected_verdict"] == "UNSAFE")).sum()
    tn = ((scored["model_verdict"] == "SAFE") & (scored["expected_verdict"] == "SAFE")).sum()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0)
    
    # LCR
    pair_map = {}
    for _, row in df.iterrows():
        img = str(row["image_id"]).strip()
        rule = str(row["rule_id"]).strip()
        verdict = str(row["model_verdict"]).strip()
        expected = str(row["expected_verdict"]).strip()

        if img not in pair_map:
            pair_map[img] = {}
        pair_map[img][rule] = (verdict, expected)

    compliant = 0
    total = 0
    for img, rules in pair_map.items():
        if "Rule A" not in rules or "Rule B" not in rules:
            continue
        total += 1
        va, ea = rules["Rule A"]
        vb, eb = rules["Rule B"]
        if va == ea and vb == eb:
            compliant += 1

    lcr = compliant / total if total > 0 else 0.0
    
    return {"f1": f1, "lcr": lcr}

def main():
    configs = ["baseline", "clahe", "decomp", "contrast"]
    runs = [1, 2, 3]
    
    results_dir = "results/innovation/lora"
    print(f"{'Configuration':<25} | {'LCR (Mean ± Std)':<20} | {'F1 (Mean ± Std)':<20}")
    print("-" * 70)
    
    summary = []
    
    for config in configs:
        lcrs = []
        f1s = []
        for run in runs:
            filename = f"qwen2_vl_lora_{config}_run_{run}_results.csv"
            filepath = os.path.join(results_dir, filename)
            
            if os.path.exists(filepath):
                df = pd.read_csv(filepath)
                metrics = compute_metrics_for_df(df)
                lcrs.append(metrics["lcr"])
                f1s.append(metrics["f1"])
        
        if lcrs:
            mean_lcr = np.mean(lcrs)
            std_lcr = np.std(lcrs)
            mean_f1 = np.mean(f1s)
            std_f1 = np.std(f1s)
            
            print(f"LoRA + {config.capitalize():<18} | {mean_lcr*100:>5.1f}% ± {std_lcr*100:<4.1f} | {mean_f1*100:>5.1f}% ± {std_f1*100:<4.1f}")
            summary.append({
                "Configuration": config,
                "LCR_Mean": mean_lcr,
                "LCR_Std": std_lcr,
                "F1_Mean": mean_f1,
                "F1_Std": std_f1
            })
            
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(os.path.join(results_dir, "lora_aggregated_metrics.csv"), index=False)
    print(f"\nAggregated results saved to {results_dir}/lora_aggregated_metrics.csv")

if __name__ == "__main__":
    main()
