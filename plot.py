from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from benchopt.plotting.helpers_compat import get_figure

CMAP = plt.get_cmap('tab10')


def _remove_prefix(text, prefix):
    return text[len(prefix):] if text.startswith(prefix) else text


def fill_between_x(fig, x, q_min, q_max, y, color, marker, label,
                   plotly=False, alpha=1.0):
    if not plotly:
        plt.plot(
            x,
            y,
            color=color,
            marker=marker,
            label=label,
            linewidth=3,
            alpha=alpha,
        )
        plt.fill_betweenx(y, q_min, q_max, color=color, alpha=.3)
        return fig


def plot_objective_curve(
    df,
    obj_col='objective_value',
    solver_filters=None,
    solvers=None,
    title=None,
    ylabel=None,
):
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
    assert not (solvers is not None and solver_filters is not None), \
        'You can either specify solvers or solver_filters, but not both.'
    markers = {i: v for i, v in enumerate(plt.Line2D.markers)}

    df = df.copy()
    dataset_name = df['data_name'].unique()[0]
    objective_name = df['objective_name'].unique()[0]
    title = f"{objective_name}\nData: {dataset_name}" if title is None else title
    df.query(f"`{obj_col}` not in [inf, -inf]", inplace=True)
    if solvers is not None:
        query = ''
        for solver_dict in solvers:
            solver = solver_dict['id']
            if query:
                query += ' | '
            query += f"solver_name == '{solver}'"
        df.query(query, inplace=True)
        if len(df) == 0:
            raise ValueError(f"No solvers after filters '{solver_filters}'")
    elif solver_filters is not None:
        query = ''
        for solver_filter in solver_filters:
            if query:
                query += ' & '
            query += f"solver_name.str.contains('{solver_filter}')"
        df.query(query, inplace=True)
        if len(df) == 0:
            raise ValueError(f"No solvers after filters '{solver_filters}'")
    solver_names = df['solver_name'].unique()

    fig = plt.figure(figsize=(10, 5))

    if df[obj_col].count() == 0:  # missing values
        plt.text(0.5, 0.5, "Not Available")
        return fig

    for i, solver_name in enumerate(solver_names):
        if solvers:
            solver_dict = [
                solver_dict for solver_dict in solvers
                if solver_dict['id'] == solver_name
            ][0]
            color = solver_dict['color']
            marker = solver_dict['marker']
            label = solver_dict['label']
            alpha = solver_dict['alpha']
        else:
            color = CMAP(i % CMAP.N)
            marker = markers[i % len(markers)]
            label = solver_name
            alpha = 1.0
        df_ = df[df['solver_name'] == solver_name]
        curve = df_.groupby('stop_val').median()

        q1 = df_.groupby('stop_val')['time'].quantile(.1)
        q9 = df_.groupby('stop_val')['time'].quantile(.9)

        fill_between_x(
            fig, curve['time'], q1, q9, curve[obj_col], color=color,
            marker=marker, label=label, plotly=False, alpha=alpha,
        )
    plt.legend(fontsize=14, loc='upper right')
    plt.yscale('log')
    plt.xlabel("Time [sec]", fontsize=14)
    ylabel = f"{_remove_prefix(obj_col, 'objective_')}: F(x)" if ylabel is None else ylabel
    plt.ylabel(
        ylabel,
        fontsize=14,
    )
    plt.title(title, fontsize=14)
    # plt.tight_layout()

    return fig

if __name__ == "__main__":
    results_file = Path("outputs") / "benchopt_run_2022-05-06_10h38m09.csv"
    df = pd.read_csv(results_file)

    common_color = CMAP(0)
    markers = {i: v for i, v in enumerate(plt.Line2D.markers)}
    solvers = [
        {
            'id': 'SGD-torch[batch_size=128,data_aug=False,lr=0.1,lr_schedule=None,momentum=0,nesterov=False,weight_decay=0.0]',
            'color': common_color,
            'marker': markers[0],
            'alpha': 0.2,
            'label': 'Vanilla SGD',
        },
        {
            'id': 'SGD-torch[batch_size=128,data_aug=True,lr=0.1,lr_schedule=None,momentum=0,nesterov=False,weight_decay=0.0]',
            'color': common_color,
            'marker': markers[1],
            'alpha': 0.4,
            'label': 'SGD + data aug.',
        },
        {
            'id': 'SGD-torch[batch_size=128,data_aug=True,lr=0.1,lr_schedule=None,momentum=0.9,nesterov=False,weight_decay=0.0]',
            'color': common_color,
            'marker': markers[2],
            'alpha': 0.6,
            'label': 'SGD + data aug. + momentum',
        },
        {
            'id': 'SGD-torch[batch_size=128,data_aug=True,lr=0.1,lr_schedule=cosine,momentum=0.9,nesterov=False,weight_decay=0.0]',
            'color': common_color,
            'marker': markers[3],
            'alpha': 0.8,
            'label': 'SGD + data aug. + momentum + cosine LR sched.',
        },
        {
            'id': 'SGD-torch[batch_size=128,data_aug=True,lr=0.1,lr_schedule=cosine,momentum=0.9,nesterov=False,weight_decay=0.0005]',
            'color': common_color,
            'marker': markers[4],
            'alpha': 1.0,
            'label': 'Best SGD',
        },
    ]

    fig = plot_objective_curve(
        df,
        obj_col='objective_test_err',
        # solver_filters=["cosine"],
        solvers=solvers,
        title='',
        ylabel='Test error',
    )
    plt.savefig('test_plot.pdf', dpi=300)
    # ax = fig.axes[0]
    # fig_leg = plt.figure(figsize=(10, 40))
    # ax_leg = fig_leg.add_subplot(111)
    # # add the legend from the previous axes
    # ax_leg.legend(*ax.get_legend_handles_labels(), loc='center')
    # # hide the axes frame and the x/y labels
    # ax_leg.axis('off')
    # fig_leg.savefig('legend.pdf', dpi=300)
