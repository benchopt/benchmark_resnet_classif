import matplotlib.pyplot as plt
import pandas as pd

from benchopt.plotting.helpers_compat import get_figure
from benchopt.plotting.helpers_compat import add_h_line
from benchopt.plotting.helpers_compat import fill_between_x

CMAP = plt.get_cmap('tab10')


def _remove_prefix(text, prefix):
    return text[len(prefix):] if text.startswith(prefix) else text


def plot_objective_curve(df, obj_col='objective_value'):
    """Plot objective curve for a given benchmark and dataset.
    Plot the objective value F(x) as a function of the time.
    Parameters
    ----------
    df : instance of pandas.DataFrame
        The benchmark results.
    obj_col : str
        Column to select in the DataFrame for the plot.
    Returns
    -------
    fig : matplotlib.Figure or pyplot.Figure
        The rendered figure, used to create HTML reports.
    """
    markers = {i: v for i, v in enumerate(plt.Line2D.markers)}

    df = df.copy()
    solver_names = df['solver_name'].unique()
    dataset_name = df['data_name'].unique()[0]
    objective_name = df['objective_name'].unique()[0]
    title = f"{objective_name}\nData: {dataset_name}"
    df.query(f"`{obj_col}` not in [inf, -inf]", inplace=True)
    y_label = "F(x)"

    fig = get_figure(False)

    if df[obj_col].count() == 0:  # missing values
        plt.text(0.5, 0.5, "Not Available")
        return fig

    for i, solver_name in enumerate(solver_names):
        df_ = df[df['solver_name'] == solver_name]
        curve = df_.groupby('stop_val').median()

        q1 = df_.groupby('stop_val')['time'].quantile(.1)
        q9 = df_.groupby('stop_val')['time'].quantile(.9)

        fill_between_x(
            fig, curve['time'], q1, q9, curve[obj_col], color=CMAP(i % CMAP.N),
            marker=markers[i % len(markers)], label=solver_name, plotly=False
        )

    plt.legend(fontsize=14)
    plt.xlabel("Time [sec]", fontsize=14)
    plt.ylabel(
        f"{_remove_prefix(obj_col, 'objective_')}: {y_label}",
        fontsize=14,
    )
    plt.title(title, fontsize=14)
    plt.tight_layout()

    return fig
