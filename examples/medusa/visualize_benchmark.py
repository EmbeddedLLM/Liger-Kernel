import json
import os
import matplotlib.pyplot as plt
import numpy as np

# Create the visualizations directory if it doesn't exist
os.makedirs("visualizations", exist_ok=True)

def read_data(moh, mnh, lengths, use_liger):
    data = {'memory': [], 'tokens_per_second': []}
    for length in lengths:
        filename = f"benchmark_stats/llama3-8b-medusa_UseLiger{use_liger}_medusaheads{moh}_numheads{mnh}_length{length}.json"
        try:
            with open(filename, 'r') as f:
                json_data = json.load(f)
                if json_data["total_peak_memory_allocated_MB"] == -1:
                    data['memory'].append("OOM")
                    data['tokens_per_second'].append("OOM")
                else:
                    data['memory'].append(json_data["total_peak_memory_allocated_MB"])
                    data['tokens_per_second'].append(json_data["avg_tokens_per_second"])
        except FileNotFoundError:
            data['memory'].append(None)
            data['tokens_per_second'].append(None)
    return data

def plot_metric(moh, mnh, lengths, metric, ylabel, filename_suffix):
    liger_true_data = read_data(moh, mnh, lengths, True)
    liger_false_data = read_data(moh, mnh, lengths, False)

    x = np.arange(len(lengths))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))

    true_color = '#1f77b4'  # Blue
    false_color = '#ff7f0e'  # Orange

    max_value = 0

    for i, (true_val, false_val) in enumerate(zip(liger_true_data[metric], liger_false_data[metric])):
        if true_val == "OOM":
            ax.text(i - width/2, 0, "OOM", ha='center', va='bottom', color=true_color)
        elif isinstance(true_val, dict):
            ax.bar(i - width/2, true_val['mean'], width, yerr=true_val['IQR']/2, 
                   label='UseLiger=True' if i == 0 else "", color=true_color, ecolor=true_color)
            max_value = max(max_value, true_val['mean'] + true_val['IQR']/2)

        if false_val == "OOM":
            ax.text(i + width/2, 0, "OOM", ha='center', va='bottom', color=false_color)
        elif isinstance(false_val, dict):
            ax.bar(i + width/2, false_val['mean'], width, yerr=false_val['IQR']/2, 
                   label='UseLiger=False' if i == 0 else "", color=false_color, ecolor=false_color)
            max_value = max(max_value, false_val['mean'] + false_val['IQR']/2)

    ax.set_ylim(0, max_value * 1.1)
    ax.set_ylabel(ylabel)
    ax.set_xlabel('Length')
    ax.set_title(f'{ylabel} Comparison (medusaheads={moh}, numheads={mnh})')
    ax.set_xticks(x)
    ax.set_xticklabels(lengths)
    ax.legend()

    plt.tight_layout()
    plt.savefig(f"visualizations/llama3-8b-medusa_medusaheads{moh}_numheads{mnh}_{filename_suffix}.png")
    plt.close()

def plot_comparisons(moh, mnh, lengths):
    plot_metric(moh, mnh, lengths, 'memory', 'Total Peak Memory Allocated (MB) Per GPU', 'memory')
    plot_metric(moh, mnh, lengths, 'tokens_per_second', 'Average Tokens per Second Per GPU', 'tokens_per_second')

# Main execution
lengths = [1024, 2048, 4096, 8192, 16384, 32768]
medusaheads = ["False", "True"]  # Example values, adjust as needed
numheads = [3, 5]  # Example values, adjust as needed

for moh in medusaheads:
    for mnh in numheads:
        plot_comparisons(moh, mnh, lengths)