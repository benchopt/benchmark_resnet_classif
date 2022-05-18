
import itertools
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import rc
import pandas as pd
import seaborn as sns


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

fontsize = 20
labelsize = 20


def _remove_prefix(text, prefix):
    return text[len(prefix):] if text.startswith(prefix) else text


def fill_between_x(ax, x, q_min, q_max, y, color, marker, label,
                   plotly=False, alpha=1.0, linestyle='-'):
    if not plotly:
        ax.plot(
            x,
            y,
            color=color,
            marker=marker,
            label=label,
            linewidth=1,
            markersize=4,
            alpha=alpha,
            linestyle=linestyle,
            markevery=10,
        )
        ax.fill_betweenx(y, q_min, q_max, color=color, alpha=.3)


def plot_objective_curve(
    df,
    ax,
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

    if df[obj_col].count() == 0:  # missing values
        ax.text(0.5, 0.5, "Not Available")
        return ax

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
            print(f"{solver_name} {curve[obj_col].min()}%")

        q1 = df_.groupby('stop_val')['time'].quantile(.1)
        q9 = df_.groupby('stop_val')['time'].quantile(.9)

        fill_between_x(
            ax, curve['time'], q1, q9, curve[obj_col], color=color,
            marker=marker, label=label, plotly=False, alpha=alpha,
            linestyle=linestyle,
        )
    # ax.legend(fontsize=14, loc='upper right')
    # plt.yscale('log')
    y_lim = [0.04, 0.2] if y_lim is None else y_lim
    if percent:
        y_lim = [y * 100 for y in y_lim]
    ax.set_ylim(y_lim)
    ax.set_xlabel("Time (s)", fontsize=fontsize - 2)
    if ylabel is not None:
        if percent:
            ylabel += ' (\%)'
        ax.set_ylabel(
            ylabel,
            fontsize=fontsize - 6,
        )
    ax.set_title(title, fontsize=fontsize - 2)
    # plt.tight_layout()

    return ax


if __name__ == "__main__":
    markers = {i: v for i, v in enumerate(list(plt.Line2D.markers)[:-4])}
    solvers = [
        {
            'id': 'SGD-torch[batch_size=128,data_aug=True,lr=0.1,lr_schedule=step,momentum=0.9,nesterov=False,weight_decay=0.0005]',
            'color': CMAP(4),
            'marker': markers[4],
            'alpha': 1.0,
            'label': 'Best SGD',
        },
        {
            'id': 'Adam-torch[batch_size=128,coupled_weight_decay=0.0,data_aug=True,decoupled_weight_decay=0.02,lr=0.001,lr_schedule=step]',
            'color': CMAP(5),
            'marker': markers[5],
            'alpha': 1.0,
            'label': 'Best Adam',
        },
        {
            'id': 'SGD-tf[batch_size=128,coupled_weight_decay=0.0005,data_aug=True,decoupled_weight_decay=0.0,lr=0.1,lr_schedule=step,momentum=0.9,nesterov=True]',
            'color': CMAP(6),
            'marker': markers[6],
            'alpha': 1.0,
            'label': 'Best SGD (TF/Keras)',
        },
    ]

    datasets = ['cifar']
    dataset_repr = {
        'mnist': 'MNIST',
        'cifar': 'CIFAR-10',
        'svhn': 'SVHN',
    }
    fig, axs = plt.subplots(1, len(datasets), figsize=[12, 3.3], constrained_layout=True)
    for i_d, dataset in enumerate(datasets):
        print('='*20)
        print(dataset)
        results_files = Path("outputs").glob('bench_vgg*.csv')
        df = pd.concat([pd.read_csv(f) for f in results_files])
        ylim = {
            'svhn': [0.023, 0.1],
            'cifar': [0.04, 0.3],
            'mnist': [0., 0.05],
        }[dataset]
        try:
            ax = axs[i_d]
        except TypeError:
            ax = axs
        ax.tick_params(axis='both', which='major', labelsize=labelsize)
        plot_objective_curve(
            df,
            ax,
            obj_col='objective_test_err',
            solvers=solvers,
            title=dataset_repr[dataset],
            ylabel='Test error' if i_d == 0 else None,
            y_lim=ylim,
            percent=True,
        )
    plt.savefig('vgg16_sgd_torch.pdf', dpi=300)
    plt.savefig('vgg16_sgd_torch.svg', dpi=300)

    try:
        ax_example = axs[0]  # we take the cifar axis
    except TypeError:
        ax_example = axs
    leg_fig, ax2 = plt.subplots(1, 1, figsize=(20, 4))
    n_col = 3
    lines_ordered = list(itertools.chain(*[ax_example.lines[i::n_col] for i in range(n_col)]))
    legend = ax2.legend(
        lines_ordered, [line.get_label() for line in lines_ordered], ncol=n_col,
        loc="upper center")
    leg_fig.canvas.draw()
    leg_fig.tight_layout()
    width = legend.get_window_extent().width
    height = legend.get_window_extent().height
    leg_fig.set_size_inches((width / 80,  max(height / 80, 0.5)))
    plt.axis('off')
    leg_fig.savefig("vgg16_sgd_torch_legend.pdf", dpi=300)
    leg_fig.savefig("vgg16_sgd_torch_legend.svg", dpi=300)

