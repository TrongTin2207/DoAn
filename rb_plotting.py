import matplotlib.pyplot as plt
import numpy as np
import os

def plot_grouped_bar(data, errors=None, labels=None, title="", xlabel="", ylabel="", legend_labels=None,
                     xtick_labels=None, filename="plot.png", save_path="./result", ylim=None):
    """
    Plot a grouped bar chart with optional error bars.

    Parameters:
    - data: 2D list or array of shape (num_groups, num_bars_per_group)
    - errors: same shape as data if provided
    - labels: list of strings for x-ticks
    - title: chart title
    - xlabel, ylabel: axis labels
    - legend_labels: legend entry for each bar in group
    - filename: file to save plot
    - save_path: folder to save in
    - ylim: tuple (ymin, ymax)
    """
    data = np.array(data)
    num_groups, num_bars = data.shape
    x = np.arange(num_groups)
    width = 0.25  # width of each bar

    fig, ax = plt.subplots(figsize=(7, 5))

    # Colors and hatches to match the uploaded images
    colors = ['lightgray', 'lightgreen', 'white']
    edgecolors = ['black'] * num_bars
    hatches = ['...', '', '///']

    for i in range(num_bars):
        y = data[:, i]
        err = errors[:, i] if errors is not None else None
        ax.bar(x + i*width, y, width, label=legend_labels[i] if legend_labels else f'Group {i+1}',
               color=colors[i % len(colors)], edgecolor=edgecolors[i], hatch=hatches[i % len(hatches)],
               yerr=err, capsize=3)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels(xtick_labels if xtick_labels else [f'{i}' for i in range(num_groups)])
    ax.legend()
    if ylim:
        ax.set_ylim(ylim)

    plt.tight_layout()
    os.makedirs(save_path, exist_ok=True)
    full_path = os.path.join(save_path, filename)
    plt.savefig(full_path, dpi=300)
    print(f"Plot saved to {full_path}")
    plt.show()