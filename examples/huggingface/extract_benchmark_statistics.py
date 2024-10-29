import os
import re
import json
import ast
import numpy as np
from scipy import stats

def extract_data(filename):
    model_type, use_liger, batch_size = re.match(r'(.+)_use_liger_(.+)_batch_size_(\d+)_rep_\d+\.log', filename).groups()

    model_type = model_type.split(os.sep)[-1]

    with open(filename, 'r') as f:
        lines = f.readlines()

    if not lines:
        print(f"Warning: {filename} is empty.")
        return None

    try:
        last_line = ast.literal_eval(lines[-1].strip())
    except (SyntaxError, ValueError):
        print(f"Warning: Unable to parse the last line in {filename}.")
        return None

    if 'step_peak_memory_allocated_MB' not in last_line or 'avg_tokens_per_second' not in last_line:
        print(f"Warning: {filename} does not contain required information.")
        return None

    return {
        'model_type': model_type,
        'use_liger': use_liger,
        'batch_size': int(batch_size),
        'step_peak_memory_allocated_MB': last_line.get('step_peak_memory_allocated_MB'),
        'avg_tokens_per_second': last_line.get('avg_tokens_per_second')
    }

def compute_statistics(data):
    if not data:
        return {
            'mean': None,
            'std': None,
            'median': None,
            'min': None,
            'max': None,
            'range': None,
            '0.05CI': None,
            '0.95CI': None,
            'interquartile_range': None
        }

    mean = np.mean(data)
    std = np.std(data)
    median = np.median(data)
    min_val = np.min(data)
    max_val = np.max(data)
    range_val = max_val - min_val
    if len(data) > 1:
        ci_05, ci_95 = stats.t.interval(0.90, len(data)-1, loc=mean, scale=stats.sem(data))
    else:
        ci_05, ci_95 = None, None
    iqr = stats.iqr(data) if len(data) > 1 else None

    return {
        'mean': mean,
        'std': std,
        'median': median,
        'min': min_val,
        'max': max_val,
        'range': range_val,
        '0.05CI': ci_05,
        '0.95CI': ci_95,
        'interquartile_range': iqr
    }

def main():
    log_files = [os.path.join("./results", fn) for fn in os.listdir('./results') if fn.endswith('.log')]
    data = [extract_data(fn) for fn in log_files]
    data = [d for d in data if d is not None]  # Remove None values

    grouped_data = {}
    for item in data:
        key = (item['model_type'], item['use_liger'], item['batch_size'])
        if key not in grouped_data:
            grouped_data[key] = {'step_peak_memory_allocated_MB': [], 'avg_tokens_per_second': []}
        if item['step_peak_memory_allocated_MB'] is not None:
            grouped_data[key]['step_peak_memory_allocated_MB'].append(item['step_peak_memory_allocated_MB'])
        if item['avg_tokens_per_second'] is not None:
            grouped_data[key]['avg_tokens_per_second'].append(item['avg_tokens_per_second'])

    results = {}
    for key, values in grouped_data.items():
        results[key] = {
            'step_peak_memory_allocated_MB': compute_statistics(values['step_peak_memory_allocated_MB']),
            'avg_tokens_per_second': compute_statistics(values['avg_tokens_per_second'])
        }

    os.makedirs('benchmark_stats', exist_ok=True)

    for key, stats in results.items():
        model_type, use_liger, batch_size = key
        filename = f"{model_type}_use_liger_{use_liger}_batch_size_{batch_size}.json"
        print(results)
        print(filename)
        filepath = os.path.join('benchmark_stats', filename)

        with open(filepath, 'w') as f:
            json.dump(stats, f, indent=2)

        print(f"Statistics saved to {filepath}")

if __name__ == "__main__":
    main()