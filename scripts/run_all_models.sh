models=(
    "HuggingFaceTB/SmolVLM-2.2B"
    "Qwen/Qwen2-VL-2B-Instruct"
    "openbmb/MiniCPM-V-2_6"
    "deepseek-ai/Janus-Pro-1.6B"
    "microsoft/Phi-3.5-vision"
    "internvl/internvl2-2b"
)

for m in "${models[@]}"
do
    for i in 1 2 3
    do
        python -m exhaustive_strategies.run_extension \
            --model "$m" \
            --dataset Dataset/ \
            --results results/${m}/run_$i/
    done
done
