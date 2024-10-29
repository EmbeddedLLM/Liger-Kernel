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
    gpu_name: str
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
        "--gpu-name",
        type=str,
        required=True,
        help="Name of the GPU (Check the exisiting value in data/all_benchmark_data.csv)",
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
        & (df["gpu_name"] == config.gpu_name)
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
    if config.metric_name == "speed":
        ylabel = f"Speed Gain Multiplier of Liger over Hugging Face"
        exit()
    else:
        ylabel = f"Memory Cost (Lower is better)"

    # Pivot the dataframe to have separate columns for liger and huggingface
    df_pivot = df.pivot(index="x_value", columns="kernel_provider", values="y_value_50")

    # Calculate the multiple of liger over huggingface
    if config.metric_name == "speed":
        # df_pivot["multiple"] = df_pivot["huggingface"] / df_pivot["liger"]
        pass
    else:
        df_pivot["multiple"] = df_pivot["huggingface"] / df_pivot["liger"]

    plt.figure(figsize=(12, 6))
    sns.set(style="whitegrid")

    # Create bar plot
    ax = sns.barplot(x=df_pivot.index, y="multiple", data=df_pivot, color="skyblue")

    # Add value labels on top of each bar
    for i, v in enumerate(df_pivot["multiple"]):
        ax.text(i, v, f"{v:.2f}x", ha="center", va="bottom")

    plt.axhline(y=1, color="r", linestyle="--", label="Baseline (Liger)", linewidth=3)

    plt.legend()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(
        f"Memory Cost of Hugging Face over Liger: {config.kernel_name} - {config.metric_name} - {config.kernel_operation_mode}"
    )
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    out_path = os.path.join(
        VISUALIZATIONS_PATH,
        f"{config.kernel_name}_{config.metric_name}_{config.kernel_operation_mode}_{config.gpu_name}_multiple_memory_cost.png",
    )

    if config.display:
        plt.show()
    if config.overwrite or not os.path.exists(out_path):
        os.makedirs(VISUALIZATIONS_PATH, exist_ok=True)
        plt.savefig(out_path)
    plt.close()


# def plot_data(df: pd.DataFrame, config: VisualizationsConfig):
#     """Plots the benchmark data, saving the result if needed.

#     Args:
#         df (pd.DataFrame): Filtered benchmark dataframe.
#         config (VisualizationsConfig): Configuration object for the visualizations script.
#     """
#     xlabel = df["x_label"].iloc[0]
#     if config.metric_name == "speed":
#         ylabel = f"Speed Gain Multiplier of Liger over Hugging Face"
#     else:
#         ylabel = f"Memory Efficiency Ratio of Liger over Hugging Face"

#     # Pivot the dataframe to have separate columns for liger and huggingface
#     df_pivot = df.pivot(index='x_value', columns='kernel_provider', values='y_value_50')

#     # Calculate the multiple of liger over huggingface
#     if config.metric_name == "speed":
#         df_pivot['multiple'] =  df_pivot['huggingface'] / df_pivot['liger']
#     else:
#         df_pivot['multiple'] = df_pivot['liger'] / df_pivot['huggingface']

#     plt.figure(figsize=(10, 6))
#     sns.set(style="whitegrid")
#     ax = sns.lineplot(
#         data=df_pivot,
#         x=df_pivot.index,
#         y='multiple',
#         marker="o",
#         color='blue',
#         label='Liger'
#     )

#     plt.axhline(y=1, color='r', linestyle='--', label='Baseline (Hugging Face)')

#     plt.legend()
#     plt.xlabel(xlabel)
#     plt.ylabel(ylabel)
#     plt.title(f"{config.kernel_name} - {config.metric_name} - {config.kernel_operation_mode}")
#     plt.tight_layout()

#     out_path = os.path.join(
#         VISUALIZATIONS_PATH, f"{config.kernel_name}_{config.metric_name}_{config.kernel_operation_mode}_{config.gpu_name}_multiple_gain.png"
#     )

#     if config.display:
#         plt.show()
#     if config.overwrite or not os.path.exists(out_path):
#         os.makedirs(VISUALIZATIONS_PATH, exist_ok=True)
#         plt.savefig(out_path)
#     plt.close()


def main():
    config = parse_args()
    df = load_data(config)
    plot_data(df, config)


if __name__ == "__main__":
    main()
