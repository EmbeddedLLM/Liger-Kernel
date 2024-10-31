import os
import re
import json
import numpy as np

# Define the directory containing the training logs
log_directories = os.listdir("./")

print(log_directories)

directory_pattern = r"llama3-8b-medusa_UseLiger(.*?)_medusaheads(.*?)_numheads(.*?)_length(.*?)_.*"

# Filter the directories that match the pattern
log_directories = [d for d in log_directories if re.match(directory_pattern, d)]


print(log_directories)


output_directory = "benchmark_stats"

# Ensure the output directory exists
os.makedirs(output_directory, exist_ok=True)

# Define a regex pattern to extract the relevant data from the log
pattern = re.compile(r"Step 20: .*total_peak_memory_allocated_MB: (\d+\.\d+), .*avg_tokens_per_second: (\d+\.\d+)")

# Function to calculate statistics
def calculate_statistics(data):
    mean = np.mean(data)
    min_val = np.min(data)
    max_val = np.max(data)
    median = np.median(data)
    std = np.std(data)
    ci_05 = np.percentile(data, 5)
    ci_95 = np.percentile(data, 95)
    iqr = np.percentile(data, 75) - np.percentile(data, 25)
    return {
        "mean": mean,
        "min": min_val,
        "max": max_val,
        "median": median,
        "std": std,
        "0.05CI": ci_05,
        "0.95CI": ci_95,
        "IQR": iqr
    }

# Iterate over each log file in the directory
for log_directory in log_directories:
    log_file = os.path.join(log_directory, "training.log")

    if os.path.exists(log_file):

        # Extract the parameters from the log file name
        match = re.match(r"llama3-8b-medusa_UseLiger(.*?)_medusaheads(.*?)_numheads(.*?)_length(.*?)_.*", log_directory)
        
        print("match: ", match)
        if not match:
            continue

        ul, moh, mnh, mml = match.groups()

        # Read the log file and extract the relevant data
        total_peak_memory_allocated_MB = []
        avg_tokens_per_second = []

        with open(log_file, 'r') as file:
            for line in file:
                match = pattern.search(line)
                if match:
                    total_peak_memory_allocated_MB.append(float(match.group(1)))
                    avg_tokens_per_second.append(float(match.group(2)))

        if len(avg_tokens_per_second) > 0:

            # Calculate statistics
            stats = {
                "total_peak_memory_allocated_MB": calculate_statistics(total_peak_memory_allocated_MB),
                "avg_tokens_per_second": calculate_statistics(avg_tokens_per_second)
            }
        else:
            # Calculate statistics
            stats = {
                "total_peak_memory_allocated_MB": -1,
                "avg_tokens_per_second": -1
            }

        # Define the output file name
        output_file = f"llama3-8b-medusa_UseLiger{ul}_medusaheads{moh}_numheads{mnh}_length{mml}.json"
        output_path = os.path.join(output_directory, output_file)

        # Save the statistics to a JSON file
        with open(output_path, 'w') as json_file:
            json.dump(stats, json_file, indent=4)

        print(f"Statistics saved to {output_path}")