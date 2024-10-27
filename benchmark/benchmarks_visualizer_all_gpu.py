import json
import os
from argparse import ArgumentParser
from dataclasses import dataclass

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

DATA_PATH = "data/all_benchmark_data.csv"
VISUALIZATIONS_PATH = "visualizations/"


@dataclass
class VisualizationsConfig:
    """
    Configuration for the visualizations script.

    Args:
        kernel_name (str): Kernel name to benchmark. (Will run `scripts/benchmark_{kernel_name}.py`)
        metric_name (str): Metric name to visualize (speed/memory)
        kernel_operation_mode (str): Kernel operation mode to visualize (forward/backward/full). Defaults to "full"
        display (bool): Display the visualization. Defaults to False
        overwrite (bool): Overwrite existing visualization, if none exist this flag has no effect as ones are always created and saved. Defaults to False

    """

    kernel_name: str
    metric_name: str
    kernel_operation_mode: str = "full"
    display: bool = False
    overwrite: bool = False


def parse_args() -> VisualizationsConfig:
    """Parse command line arguments into a configuration object.

    Returns:
        VisualizationsConfig: Configuration object for the visualizations script.
    """
    parser = ArgumentParser()
    parser.add_argument(
        "--kernel-name", type=str, required=True, help="Kernel name to benchmark"
    )
    parser.add_argument(
        "--metric-name",
        type=str,
        required=True,
        help="Metric name to visualize (speed/memory)",
    )
    parser.add_argument(
        "--kernel-operation-mode",
        type=str,
        required=True,
        help="Kernel operation mode to visualize (forward/backward/full)",
    )
    parser.add_argument(
        "--display", action="store_true", help="Display the visualization"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing visualization, if none exist this flag has no effect as one are always created",
    )

    args = parser.parse_args()

    return VisualizationsConfig(**dict(args._get_kwargs()))


def load_data(config: VisualizationsConfig) -> pd.DataFrame:
    """Loads the benchmark data from the CSV file and filters it based on the configuration.

    Args:
        config (VisualizationsConfig): Configuration object for the visualizations script.

    Raises:
        ValueError: If no data is found for the given filters.

    Returns:
        pd.DataFrame: Filtered benchmark dataframe.
    """
    df = pd.read_csv(DATA_PATH)
    df["extra_benchmark_config"] = df["extra_benchmark_config_str"].apply(json.loads)
    unique_values = df["kernel_name"].unique()
    print(unique_values)
    filtered_df = df[
        (df["kernel_name"] == config.kernel_name)
        & (df["metric_name"] == config.metric_name)
        & (df["kernel_operation_mode"] == config.kernel_operation_mode)
        # Use this to filter by extra benchmark configuration property
        # & (data['extra_benchmark_config'].apply(lambda x: x.get('H') == 4096))
        # FIXME: maybe add a way to filter using some configuration, except of hardcoding it
    ]

    if filtered_df.empty:
        raise ValueError("No data found for the given filters")

    return filtered_df


def plot_data(df: pd.DataFrame, config: VisualizationsConfig):
    """Plots the benchmark data, saving the result if needed.

    Args:
        df (pd.DataFrame): Filtered benchmark dataframe.
        config (VisualizationsConfig): Configuration object for the visualizations script.
    """
    xlabel = df["x_label"].iloc[0]
    ylabel = f"{config.metric_name} ({df['metric_unit'].iloc[0]})"
    # Sort by "kernel_provider" and "gpu_name" to ensure consistent color and style assignment
    df = df.sort_values(by=["kernel_provider", "gpu_name"])

    plt.figure(figsize=(12, 8))
    sns.set(style="whitegrid")

    # Create a color palette for kernel providers
    kernel_providers = df["kernel_provider"].unique()
    color_palette = sns.color_palette("tab10", n_colors=len(kernel_providers))
    color_dict = dict(zip(kernel_providers, color_palette))

    # Create a list of line styles for different GPUs
    line_styles = ["-", "--", ":", "-."]
    gpu_names = df["gpu_name"].unique()
    style_dict = dict(zip(gpu_names, line_styles[: len(gpu_names)]))

    for (provider, gpu), group in df.groupby(["kernel_provider", "gpu_name"]):
        color = color_dict[provider]
        style = style_dict[gpu]

        ax = sns.lineplot(
            data=group,
            x="x_value",
            y="y_value_50",
            color=color,
            linestyle=style,
            marker="o",
            label=f"{provider} - {gpu}",
        )

        y_error_lower = group["y_value_50"] - group["y_value_20"]
        y_error_upper = group["y_value_80"] - group["y_value_50"]
        y_error = [y_error_lower, y_error_upper]

        plt.errorbar(
            group["x_value"],
            group["y_value_50"],
            yerr=y_error,
            fmt="none",
            color=color,
            capsize=5,
        )

    plt.legend(
        title="Kernel Provider - GPU", bbox_to_anchor=(1.05, 1), loc="upper left"
    )
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()

    out_path = os.path.join(
        VISUALIZATIONS_PATH,
        f"{config.kernel_name}_{config.metric_name}_{config.kernel_operation_mode}.png",
    )

    if config.display:
        plt.show()
    if config.overwrite or not os.path.exists(out_path):
        os.makedirs(VISUALIZATIONS_PATH, exist_ok=True)
        plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def main():
    config = parse_args()
    df = load_data(config)
    plot_data(df, config)


if __name__ == "__main__":
    main()
