import subprocess
import os
import sys

# Agentic Foveation is a Qwen2-VL-only innovation
models_to_run = [
    ("Qwen2-VL (Foveation)", "run_qwen2_vl_cot_foveation.py", r"four_models\Scripts\python.exe"),
]

script_dir = "src/generation_cot_foveation"

print("==================================================")
print("Starting Agentic Foveation Evaluation Suite (N=3)")
print("==================================================")

for model_name, script_name, python_exe in models_to_run:
    script_path = os.path.join(script_dir, script_name)
    
    if not os.path.exists(script_path):
        print(f"\n[ERROR] Script not found for {model_name}: {script_path}")
        continue
    
    if not os.path.exists(python_exe):
        print(f"\n[ERROR] Environment not found for {model_name}: {python_exe}")
        continue
        
    print(f"\n=== Running Foveation Evaluation for: {model_name} ===")
    
    try:
        process = subprocess.Popen(
            [python_exe, "-u", script_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        
        for line in process.stdout:
            print(line, end="")
            
        process.wait()
        
        if process.returncode == 0:
            print(f"=== Successfully completed {model_name} ===")
        else:
            print(f"\n[ERROR] Process for {model_name} failed with return code {process.returncode}")
            
    except Exception as e:
        print(f"\n[EXCEPTION] Failed to run {model_name}: {str(e)}")

print("\n==================================================")
print("Agentic Foveation Evaluation Suite Complete.")
print("==================================================")
