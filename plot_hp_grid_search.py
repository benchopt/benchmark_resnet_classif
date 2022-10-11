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
        handles = ax.plot(
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
        return handles


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

    handles = []
    labels = []
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

        handles.extend(fill_between_x(
            ax, curve['time'], q1, q9, curve[obj_col], color=color,
            marker=marker, label=label, plotly=False, alpha=alpha,
            linestyle=linestyle,
        ))
        labels.append(label)
    y_lim = [0.04, 0.2] if y_lim is None else y_lim
    if percent:
        y_lim = [y * 100 for y in y_lim]
    ax.set_ylim(y_lim)
    if ylabel is not None:
        if percent:
            ylabel += ' (\%)'
        ax.set_ylabel(
            ylabel,
            fontsize=labelsize,
        )
    if title is not None:
        ax.set_title(title, fontsize=labelsize)
    ax.tick_params(axis='both', which='major', labelsize=ticksize)

    return ax, handles, labels


def extract_hp_value_from_solver_name(solver_name, hp_name):
    solver_attributes = solver_name.split(',')
    for attribute in solver_attributes:
        if attribute.startswith(hp_name):
            hp_value = attribute.split('=')[1]
            # strip from potential ] at the end
            hp_value = hp_value.strip(']')
            # if float use the scientific notation
            try:
                hp_value = float(hp_value)
                hp_value = f'{hp_value:.1e}'
            except ValueError:
                pass
            return hp_value


if __name__ == "__main__":
    markers = {i: v for i, v in enumerate(list(plt.Line2D.markers)[:-4])}
    sota_resnet = {
        'mnist': 0.09,  # from papers with code
        'cifar': 100 - 95.27,  # from lookahead
        'cifar100': 100 - 78.49,
        'svhn': 2.95,  # from AMP, with pre act
    }
    dataset = 'svhn'
    optimizers = ['SGD', 'Adam']
    hyperparameters = ['lr', 'wd']
    hyperparameter_key_per_optimizer = {
        'lr': {'Adam': 'lr', 'SGD': 'lr'},
        'wd': {'Adam': 'decoupled_weight_decay', 'SGD': 'weight_decay'},
    }
    hyperparameter_repr = {
        'lr': 'Learning rate',
        'wd': 'Weight decay',
    }
    default_hp_values_per_optimizer = {
        'lr': {'Adam': 0.001, 'SGD': 0.1},
        'wd': {'Adam': 0.02, 'SGD': 0.0005},
    }
    results_file = f'outputs/resnet_grid_neurips_{dataset}.csv'
    df = pd.read_csv(results_file)
    fig = plt.figure(figsize=[11, 1+2*1.5])
    outer_gs = fig.add_gridspec(2, 2, hspace=0.8)
    ref_ax = None
    for i_optimizer, optimizer in enumerate(optimizers):
        for i_hyperparameter, hyperparameter in enumerate(hyperparameters):
            inner_gs = outer_gs[i_optimizer, i_hyperparameter].subgridspec(
                nrows=2,
                ncols=2,
                height_ratios=[0.1, 2],
                hspace=0.3,
            )
            print('='*20)
            print(optimizer, hyperparameter)
            ylim = {
                'svhn': [0.023, 0.08],
                'cifar': [0.04, 0.2],
                'cifar100': [0.2, 1],
                'mnist': [0., 0.03],
            }[dataset]
            ax = fig.add_subplot(inner_gs[1, :])
            ax.tick_params(axis='both', which='major', labelsize=labelsize)
            constant_hp = 'lr' if hyperparameter == 'wd' else 'wd'
            constant_hp_repr = hyperparameter_key_per_optimizer[constant_hp][optimizer]
            constant_hp_value = default_hp_values_per_optimizer[constant_hp][optimizer]
            _, handles, labels = plot_objective_curve(
                df,
                ax,
                obj_col='objective_test_err',
                solver_filters=[optimizer, f'{constant_hp_repr}={constant_hp_value}'],
                title=None,
                ylabel=None,
                y_lim=ylim,
                percent=True,
            )
            sota = sota_resnet[dataset]
            ax.axhline(sota, color='k', linestyle='--')
            ax_legend = fig.add_subplot(inner_gs[0, :])
            ax_legend.axis('off')
            hp_repr = hyperparameter_key_per_optimizer[hyperparameter][optimizer]
            extracted_labels = [extract_hp_value_from_solver_name(label, hp_repr) for label in labels]
            sorted_handles, sorted_labels = zip(*sorted(zip(handles, extracted_labels), key=lambda x: x[1]))
            ax_legend.legend(
                sorted_handles,
                sorted_labels,
                loc='center',
                ncol=4,
                handlelength=1.5,
                handletextpad=.2,
                frameon=False,
            )
            ax_title = fig.add_subplot(outer_gs[i_optimizer, i_hyperparameter])
            ax_title.axis('off')
            ax_title.set_title(
                f"{optimizer} - {hyperparameter_repr[hyperparameter]}",
                fontsize=labelsize,
                y=1.08,
            )
    fig.supylabel('Test error (\%)', fontsize=labelsize, x=0.05)
    fig.supxlabel("Time (s)", fontsize=labelsize, y=-0.03)
    plt.savefig(f'{dataset}_hp_sens.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(f'{dataset}_hp_sens.svg', dpi=300, bbox_inches='tight')

    # ax_example = axs[0, 0]  # we take the cifar axis
    # leg_fig, ax2 = plt.subplots(1, 1, figsize=(20, 4))
    # n_col = 3
    # legend = ax2.legend(
    #     ax_example.lines, [line.get_label() for line in ax_example.lines], ncol=n_col,
    #     loc="upper center")
    # leg_fig.canvas.draw()
    # leg_fig.tight_layout()
    # width = legend.get_window_extent().width
    # height = legend.get_window_extent().height
    # leg_fig.set_size_inches((width / 80,  max(height / 80, 0.5)))
    # plt.axis('off')
    # fig2_name = f"{dataset}_hp_sens_legend.pdf"
    # leg_fig.savefig(fig2_name, dpi=300)
    # os.system(f"pdfcrop {fig2_name} {fig2_name}")
    # leg_fig.savefig(f"{dataset}_hp_sens_legend.svg", dpi=300)

