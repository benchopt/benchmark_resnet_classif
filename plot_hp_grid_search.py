import itertools
import os
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

titlesize = 22
ticksize = 16
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
            linewidth=2,
            markersize=6,
            alpha=alpha,
            linestyle=linestyle,
            markevery=20,
        )
        # ax.fill_betweenx(y, q_min, q_max, color=color, alpha=.3)


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
        curve = df_.groupby('stop_val').median().ewm(span=20).mean()
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
    ax.set_xlabel("Time (s)", fontsize=labelsize)
    if ylabel is not None:
        if percent:
            ylabel += ' (\%)'
        ax.set_ylabel(
            ylabel,
            fontsize=labelsize,
        )
    ax.set_title(title, fontsize=labelsize)
    ax.tick_params(axis='both', which='major', labelsize=ticksize)
    # plt.tight_layout()

    return ax


if __name__ == "__main__":
    markers = {i: v for i, v in enumerate(list(plt.Line2D.markers)[:-4])}
    sota_resnet = {
        'mnist': 0.09,  # from papers with code
        'cifar': 100 - 95.27,  # from lookahead
        'svhn': 2.95,  # from AMP, with pre act
    }
    optimizers = ['SGD', 'Adam']
    hyperparameters = ['lr', 'wd']
    hyperparameter_key_per_optimizer = {
        'lr': {'Adam': 'lr', 'SGD': 'lr'},
        'wd': {'Adam': 'coupled_weight_decay', 'SGD': 'weight_decay'},
    }
    hyperparameter_repr = {
        'lr': 'Learning rate',
        'wd': 'Weight decay',
    }
    default_hp_values_per_optimizer = {
        'lr': {'Adam': 0.001, 'SGD': 0.1},
        'wd': {'Adam': 0.02, 'SGD': 0.0005},
    }
    results_file = 'outputs/resnet_grid_neurips.csv'
    df = pd.read_csv(results_file)
    fig, axs = plt.subplots(2, 2, figsize=[11, 1+2*1], constrained_layout=True)
    for i_optimizer, optimizer in enumerate(optimizers):
        for i_hyperparameter, hyperparameter in enumerate(hyperparameters):
            print('='*20)
            print(optimizer, hyperparameter)
            ylim = {
                'svhn': [0.023, 0.08],
                'cifar': [0.04, 0.2],
                'mnist': [0., 0.03],
            }['cifar']
            ax = axs[i_optimizer, i_hyperparameter]
            ax.tick_params(axis='both', which='major', labelsize=labelsize)
            constant_hp = 'lr' if hyperparameter == 'wd' else 'wd'
            constant_hp_repr = hyperparameter_key_per_optimizer[constant_hp][optimizer]
            constant_hp_value = default_hp_values_per_optimizer[constant_hp][optimizer]
            plot_objective_curve(
                df,
                ax,
                obj_col='objective_test_err',
                solver_filters=[optimizer, f'{constant_hp_repr}={constant_hp_value}'],
                title=f"{optimizer} - {hyperparameter_repr[hyperparameter]}",
                ylabel='Test error' if i_optimizer == 0 else None,
                y_lim=ylim,
                percent=True,
            )
            sota = sota_resnet['cifar']
            if sota is not None:
                ax.axhline(sota, color='k', linestyle='--')
    plt.savefig('cifar_hp_sens.pdf', dpi=300)
    plt.savefig('cifar_hp_sens.svg', dpi=300)

    ax_example = axs[0, 0]  # we take the cifar axis
    leg_fig, ax2 = plt.subplots(1, 1, figsize=(20, 4))
    n_col = 3
    legend = ax2.legend(
        ax_example.lines, [line.get_label() for line in ax_example.lines], ncol=n_col,
        loc="upper center")
    leg_fig.canvas.draw()
    leg_fig.tight_layout()
    width = legend.get_window_extent().width
    height = legend.get_window_extent().height
    leg_fig.set_size_inches((width / 80,  max(height / 80, 0.5)))
    plt.axis('off')
    fig2_name = "cifar_hp_sens_legend.pdf"
    leg_fig.savefig(fig2_name, dpi=300)
    os.system(f"pdfcrop {fig2_name} {fig2_name}")
    leg_fig.savefig("cifar_hp_sens_legend.svg", dpi=300)

