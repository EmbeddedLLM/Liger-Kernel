import os
import json
import matplotlib.pyplot as plt
import numpy as np

# Configuration
input_dir = "benchmark_stats"
output_dir = "visualizations"
fields_to_plot = ["avg_tokens_per_second", "step_peak_memory_allocated_MB"]
batch_sizes = [64, 128, 192, 256, 512]

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

def parse_filename(filename):
    parts = filename.split("_")
    return {
        "model_type": parts[0],
        "use_liger": parts[3] == "True",
        "batch_size": int(parts[-1].split(".")[0])
    }

def extract_data(file_data, field):
    return {
        "mean": file_data[field]["mean"],
        "ci_low": file_data[field]["0.05CI"],
        "ci_high": file_data[field]["0.95CI"]
    }

# Read and parse JSON files
data = {}
for filename in os.listdir(input_dir):
    if filename.endswith(".json"):
        file_info = parse_filename(filename)
        model_type = file_info["model_type"]
        use_liger = file_info["use_liger"]
        batch_size = file_info["batch_size"]

        with open(os.path.join(input_dir, filename), 'r') as f:
            file_data = json.load(f)

        for field in fields_to_plot:
            data.setdefault(field, {}).setdefault(model_type, {}).setdefault(batch_size, {})[use_liger] = extract_data(file_data, field)

# Plotting function
def create_plot(field, model_type, field_data):
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(batch_sizes))
    width = 0.35

    for use_liger in [True, False]:
        means = []
        errors = []
        for batch_size in batch_sizes:
            batch_data = field_data.get(batch_size, {}).get(use_liger, {"mean": 0, "ci_low": 0, "ci_high": 0})
            means.append(batch_data["mean"])
            errors.append((batch_data["ci_high"] - batch_data["ci_low"]) / 2)

        label = 'Liger (USE_LIGER=True)' if use_liger else 'Hugging Face (USE_LIGER=False)'
        ax.bar(x + width/2 if use_liger else x - width/2, means, width, label=label, yerr=errors, capsize=5)

        # Add OOM text for missing data
        for i, (mean, error) in enumerate(zip(means, errors)):
            if mean == 0 and error == 0:
                ax.text(x[i] + (width/2 if use_liger else -width/2), 0, 'OOM', 
                        ha='center', va='bottom', color='red', fontweight='bold')

    ax.set_ylabel(field.replace("_", " ").title())
    ax.set_title(f'{field.replace("_", " ").title()} - Model Type: {model_type}')
    ax.set_xticks(x)
    ax.set_xticklabels(batch_sizes)
    ax.legend()

    plt.xlabel('Batch Size')
    plt.tight_layout()

    # Save the plot as a PNG file
    plt.savefig(os.path.join(output_dir, f"{model_type}_{field}.png"))
    plt.close(fig)

# Create plots for each field and model type
for field in fields_to_plot:
    for model_type in data[field]:
        create_plot(field, model_type, data[field][model_type])

print(f"Plots have been saved in the '{output_dir}' directory.")