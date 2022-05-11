from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import rc
import pandas as pd
import seaborn as sns

from benchopt.plotting.helpers_compat import get_figure
from regex import P


# matplotlib style config
fontsize = 12
rc('font', **{'family': 'sans-serif',
              'sans-serif': ['Computer Modern Roman']})
usetex = matplotlib.checkdep_usetex(True)
params = {'axes.labelsize': fontsize,
          'font.size': fontsize,
          'legend.fontsize': fontsize,
          'xtick.labelsize': fontsize - 2,
          'ytick.labelsize': fontsize - 2,
          'text.usetex': usetex,
          'figure.figsize': (8, 6)}
plt.rcParams.update(params)
sns.set_palette('colorblind')
sns.set_style("ticks")
CMAP = plt.get_cmap('tab20')


def _remove_prefix(text, prefix):
    return text[len(prefix):] if text.startswith(prefix) else text


def fill_between_x(fig, x, q_min, q_max, y, color, marker, label,
                   plotly=False, alpha=1.0, linestyle='-'):
    if not plotly:
        plt.plot(
            x,
            y,
            color=color,
            marker=marker,
            label=label,
            linewidth=2,
            markersize=6,
            alpha=alpha,
            linestyle=linestyle,
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
    percent=False,
    y_lim=None,
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
            linestyle = solver_dict.get('linestyle', '-')
        else:
            color = CMAP(i % CMAP.N)
            marker = markers[i % len(markers)]
            label = solver_name
            alpha = 1.0
            linestyle = '-'
        df_ = df[df['solver_name'] == solver_name]
        curve = df_.groupby('stop_val').median()
        if percent:
            curve[obj_col] = curve[obj_col] * 100

        q1 = df_.groupby('stop_val')['time'].quantile(.1)
        q9 = df_.groupby('stop_val')['time'].quantile(.9)

        fill_between_x(
            fig, curve['time'], q1, q9, curve[obj_col], color=color,
            marker=marker, label=label, plotly=False, alpha=alpha,
            linestyle=linestyle,
        )
    plt.legend(fontsize=14, loc='upper right')
    # plt.yscale('log')
    y_lim = [0.04, 0.2] if y_lim is None else y_lim
    if percent:
        y_lim = [y * 100 for y in y_lim]
    plt.ylim(y_lim)
    plt.xlabel("Time [sec]", fontsize=14)
    ylabel = f"{_remove_prefix(obj_col, 'objective_')}: F(x)" if ylabel is None else ylabel
    if percent:
        ylabel += ' [\%]'
    plt.ylabel(
        ylabel,
        fontsize=14,
    )
    plt.title(title, fontsize=14)
    # plt.tight_layout()

    return fig

if __name__ == "__main__":
    markers = {i: v for i, v in enumerate(plt.Line2D.markers)}
    solvers = [
        {
            'id': 'SGD-torch[batch_size=128,data_aug=False,lr=0.1,lr_schedule=None,momentum=0,nesterov=False,weight_decay=0.0]',
            'color': CMAP(0),
            'marker': markers[0],
            'alpha': 0.3,
            'label': 'Vanilla SGD',
        },
        {
            'id': 'SGD-torch[batch_size=128,data_aug=True,lr=0.1,lr_schedule=None,momentum=0,nesterov=False,weight_decay=0.0]',
            'color': CMAP(1),
            'marker': markers[1],
            'alpha': 0.4,
            'label': 'SGD + data aug.',
        },
        {
            'id': 'SGD-torch[batch_size=128,data_aug=True,lr=0.1,lr_schedule=None,momentum=0.9,nesterov=False,weight_decay=0.0]',
            'color': CMAP(2),
            'marker': markers[2],
            'alpha': 0.6,
            'label': 'SGD + data aug. + momentum',
        },
        {
            'id': 'SGD-torch[batch_size=128,data_aug=True,lr=0.1,lr_schedule=cosine,momentum=0.9,nesterov=False,weight_decay=0.0]',
            'color': CMAP(3),
            'marker': markers[3],
            'alpha': 0.8,
            'label': 'SGD + data aug. + momentum + cosine LR sched.',
        },
        {
            'id': 'SGD-torch[batch_size=128,data_aug=True,lr=0.1,lr_schedule=cosine,momentum=0.9,nesterov=False,weight_decay=0.0005]',
            'color': CMAP(4),
            'marker': markers[4],
            'alpha': 1.0,
            'label': 'Best SGD',
        },
        {
            'id': 'Adam-torch[batch_size=128,coupled_weight_decay=0.0,data_aug=True,decoupled_weight_decay=0.02,lr=0.001,lr_schedule=cosine]',
            'color': CMAP(5),
            'marker': markers[5],
            'alpha': 1.0,
            'label': 'Best Adam',
        },
        {
            'id': 'SGD-tf[batch_size=128,data_aug=True,lr=0.1,lr_schedule=cosine,momentum=0.9,nesterov=False,weight_decay=0.0005]',
            'color': CMAP(6),
            'marker': markers[6],
            'alpha': 1.0,
            'label': 'Best SGD (TF/Keras)',
        },
    ]
    for dataset in ['mnist', 'cifar', 'svhn']:
        results_file = Path("outputs") / f"bench_{dataset}.csv"
        df = pd.read_csv(results_file)
        ylim = {
            'svhn': [0.03, 0.1],
            'cifar': None,
            'mnist': [0.005, 0.1],
        }[dataset]
        fig = plot_objective_curve(
            df,
            obj_col='objective_test_err',
            # solver_filters=["cosine"],
            solvers=solvers,
            title='',
            ylabel='Test error',
            y_lim=ylim,
            percent=True,
        )
        plt.savefig(f'resnet18_sgd_torch_{dataset}.pdf', dpi=300)
