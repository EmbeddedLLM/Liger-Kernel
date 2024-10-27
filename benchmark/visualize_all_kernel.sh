#!/bin/bash

kernel_names=("cross_entropy")
#kernel_names=("cross_entropy" "embedding" "fused_linear_cross_entropy" "fused_linear_jsd" "geglu" "jsd" "kl_div" "layer_norm" "rms_norm" "rope" "swiglu")
metric_names=("memory" "speed")
kernel_operation_modes=("forward" "backward" "full")

for kernel in "${kernel_names[@]}"; do
    for metric in "${metric_names[@]}"; do
        for mode in "${kernel_operation_modes[@]}"; do
            echo "Running with kernel-name: $kernel, metric-name: $metric, kernel-operation-mode: $mode"
            python3 benchmarks_visualizer_gpu.py \
            --kernel-name $kernel \
            --metric-name $metric \
            --gpu-name "AMD Instinct MI300X" \
            --kernel-operation-mode $mode --overwrite
            echo "----------------------------------------"
        done
    done
done
