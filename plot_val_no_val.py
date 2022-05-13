
import enum
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
            markersize=3,
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
    compute_best=False,
    compute_best_at_time=None,
    plot=True,
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

    if compute_best:
        best_dict = {}

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
        if compute_best:
            if compute_best_at_time is not None:
                best_obj = curve[curve['time'] == compute_best_at_time[solver_name]['time']][obj_col].values[0]
                best_obj_time = compute_best_at_time[solver_name]['time']
            else:
                best_obj = curve[obj_col].min()
                best_obj_time = curve[curve[obj_col] == best_obj]['time'].values[0]
            best_dict[solver_name] = {
                'obj': best_obj,
                'time': best_obj_time,
            }

        q1 = df_.groupby('stop_val')['time'].quantile(.1)
        q9 = df_.groupby('stop_val')['time'].quantile(.9)

        if plot:
            fill_between_x(
                ax, curve['time'], q1, q9, curve[obj_col], color=color,
                marker=marker, label=label, plotly=False, alpha=alpha,
                linestyle=linestyle,
            )
    if plot:
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
    if compute_best:
        return ax, best_dict
    return ax


if __name__ == "__main__":
    markers = {i: v for i, v in enumerate(list(plt.Line2D.markers)[:-4])}
    solvers = [
        {
            'id': 'SGD-torch[batch_size=128,data_aug=True,lr=0.1,lr_schedule=cosine,momentum=0.9,nesterov=False,weight_decay=0.0005]',
            'label': 'Best SGD with cosine LR schedule',
        },
        {
            'id': 'SGD-torch[batch_size=128,data_aug=True,lr=0.1,lr_schedule=step,momentum=0.9,nesterov=False,weight_decay=0.0005]',
            'label': 'Best SGD with step LR schedule',
        },
    ]

    # datasets = ['cifar_no_val', 'svhn', 'mnist']
    datasets = ['cifar', 'svhn', 'mnist']
    dataset_repr = {
        'mnist': 'MNIST',
        'cifar': 'CIFAR-10',
        'svhn': 'SVHN',
    }
    sota_resnet = {
        'mnist': 0.09,  # from papers with code
        'cifar': 100 - 93.27,  # from lookahead
        'svhn': 2.95,  # from AMP, with pre act
    }
    curve_types = {
        'no_val_test': (False, 'test_err'),
        'val_val': (True, 'val_err'),
        'val_test': (True, 'test_err'),
    }
    linestyles_dict = {
        'no_val_test': '-',
        'val_val': '--',
        'val_test': '-.',
    }
    # alpha_dict = {
    #     'no_val_test': 0.5,
    #     'val_val': 1.0,
    #     'val_test': 0.8,
    # }
    n_markers_color = {
        (solver['id'], curve_type): i
        for i, (solver, curve_type) in enumerate(itertools.product(
            solvers,
            curve_types,
        ))
    }
    fig, axs = plt.subplots(1, 3, figsize=[12, 3.3], constrained_layout=True)
    for i_d, dataset in enumerate(datasets):
        best_dict = {}
        for curve_type, (with_val, objective) in curve_types.items():
            results_name = f"bench_{dataset}"
            if with_val:
                results_name += '_val'
            else:
                results_name += '_no_val'
            results_name += '.csv'
            results_file = Path("outputs") / results_name
            df = pd.read_csv(results_file)
            ylim = {
                'svhn': [0.028, 0.1],
                'cifar': [0.04, 0.1],
                'mnist': [0., 0.05],
            }[dataset]
            xlim_left = {
                'svhn': 700,
                'cifar': 700,
                'mnist': 700,
            }[dataset]
            ax = axs[i_d]
            ax.tick_params(axis='both', which='major', labelsize=labelsize)
            solvers_curve_type = [
                dict(
                    **solver,
                    linestyle=linestyles_dict[curve_type],
                    alpha=1.0,
                    marker=markers[n_markers_color[(solver['id'], curve_type)]],
                    color=CMAP(n_markers_color[(solver['id'], curve_type)]),

                ) for solver in solvers
            ]
            for solver in solvers_curve_type:
                solver['label'] = f"{solver['label']} ({curve_type})"
            best_dict[curve_type] = plot_objective_curve(
                df,
                ax,
                obj_col=f'objective_{objective}',
                solvers=solvers_curve_type,
                title=dataset_repr[dataset],
                ylabel='Test/Val error' if i_d == 0 else None,
                y_lim=ylim,
                percent=True,
                compute_best=True,
                compute_best_at_time=best_dict.get('val_val', None) if curve_type == 'val_test' else None,
                plot=curve_type != 'val_val',
            )[1]
            ax.set_xlim(left=xlim_left)
        sota = sota_resnet[dataset]
        if sota is not None:
            ax.axhline(sota, color='k', linestyle='--')
        # show val case in best dict
        for curve_type in ['val_test', 'no_val_test']:
            b = best_dict[curve_type]
            left = True
            for solver_name, solver_dict in b.items():
                test_err = solver_dict['obj']
                xy = (solver_dict['time'], test_err)
                color = CMAP(n_markers_color[(solver_name, curve_type)])
                ax.scatter(*xy, marker='x', color=color, s=100, zorder=4)
                # ax.text(
                #     *xy,
                #     f"{test_err:.2f}",
                #     color=color,
                # )
                ax.axhline(test_err, color=color, linestyle='--')
    plt.savefig('resnet18_sgd_torch_val.pdf', dpi=300)
    plt.savefig('resnet18_sgd_torch_val.svg', dpi=300)

    ax_example = axs[0]  # we take the cifar axis
    leg_fig, ax2 = plt.subplots(1, 1, figsize=(20, 4))
    n_col = 2
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
    leg_fig.savefig("resnet18_sgd_torch_legend_val.pdf", dpi=300)
    leg_fig.savefig("resnet18_sgd_torch_legend_val.svg", dpi=300)

