from typing import List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
from sklearn.metrics import r2_score
from tensordict import TensorDict

from context_general_bci.config import DatasetConfig
from context_general_bci.plotting.styleguide import prep_plt

def rasterplot(spike_arr: np.ndarray, bin_size_s=0.02, ax=None):
    r""" spike_arr: Time x Neurons """
    if ax is None:
        ax = plt.gca()
    for idx, unit in enumerate(spike_arr.T):
        ax.scatter(
            np.where(unit)[0] * bin_size_s,
            np.ones((unit != 0).sum()) * idx,
            s=1, c='k', marker='|',
            linewidths=1., alpha=1.)
    ax.set_yticks(np.arange(0, spike_arr.shape[1], 40))
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Neuron #')

def plot_logits(ax, logits, title, bin_size_ms, vmin=-20, vmax=20, truth=None):
    ax = prep_plt(ax, big=True)
    ax.invert_yaxis()
    sns.heatmap(logits.cpu().T, ax=ax, cmap="RdBu_r", vmin=vmin, vmax=vmax)
    if truth is not None:
        ax.plot(truth.cpu().T, color="k", linewidth=2, linestyle="--", alpha=0.2)
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Bhvr (class)")
    ax.set_title(title)
    ax.set_yticks([])
    ax.set_xticks(np.linspace(0, logits.shape[0], 3))
    ax.set_xticklabels(np.linspace(0, logits.shape[0] * bin_size_ms, 3).astype(int))

    cbar = ax.collections[0].colorbar
    cbar.ax.set_ylabel('Logit')

def plot_split_logits_flat(full_logits, labels, cfg: DatasetConfig, truth=None, subset_labels=None):
    r"""
        Plot flat stream, flat presumed tiled
    """
    #
    if subset_labels:
        indices = [labels.index(l) for l in subset_labels]
    else:
        indices = range(len(labels))
    f, axes = plt.subplots(len(indices), 1, figsize=(15, 10), sharex=True, sharey=True, layout='constrained')
    if len(indices) == 1:
        axes = [axes]
    # Split logits
    stride = len(labels)
    for i, label in enumerate(labels):
        if i not in indices:
            continue
        logits = full_logits[i::stride]
        if truth is not None:
            truth_i = truth[i::stride]
        else:
            truth_i = None
        plot_logits(axes[indices.index(i)], logits, label, cfg.bin_size_ms, truth=truth_i)
    return f, axes

def plot_split_logits(
        full_logits: torch.Tensor,
        labels,
        cfg: DatasetConfig,
        truth: torch.Tensor | None=None
    ):
    r"""
        Plot reconstructed stream, from `simple_unflatten_batch`
        full_logits: B x K x H
    """
    f, axes = plt.subplots(full_logits.shape[1], 1, figsize=(15, 10), sharex=True, sharey=True, layout='constrained')
    if full_logits.shape[1] == 1:
        axes = [axes]
    for i, label in enumerate(labels):
        logits = full_logits[:, i]
        if truth is not None:
            truth_i = truth[:, i]
        else:
            truth_i = None
        plot_logits(axes[i], logits, label, cfg.bin_size_ms, truth=truth_i)
    return f, axes


def plot_prediction_spans_dict(
        ax,
        plot_dict: TensorDict,
        color,
        model_label,
        model_key='behavior_pred',
        plot_trial_markers=True,
        alpha=0.9,
        linestyle="-",
):
    # Convert boolean tensor to numpy for easier manipulation
    is_student_np = plot_dict['is_student'].numpy()
    change_points = np.where(is_student_np[:-1] != is_student_np[1:])[0] + 1
    change_points = np.concatenate(([0], change_points, [len(is_student_np)]))
    first_line = True

    # Plot the lines
    for start, end in zip(change_points[:-1], change_points[1:]):
        if is_student_np[start]:  # Check if the span is True
            label = model_label if first_line else None  # Label only the first line
            ax.plot(
                np.arange(start, end),
                plot_dict['kin'][model_key][start:end],
                color=color,
                label=label,
                alpha=alpha,
                linestyle=linestyle,
                linewidth=2,
            )
            first_line = False  # Update the flag as the first line is plotted
    # Vline on trial mark transition
    if plot_trial_markers:
        diffs = plot_dict['pseudo_trial'][:-1] - plot_dict['pseudo_trial'][1:]
        diffs = torch.cat([diffs, torch.tensor([0])])
        for trial_changept in torch.where(diffs != 0)[0]:
            ax.axvline(trial_changept, linestyle="--", color="k", alpha=0.5)


def plot_target_pred_overlay_dict(
    plot_dict: TensorDict,
    label,
    palette,
    sources={'behavior': 'True', 'behavior_pred': 'NDT'},
    ax=None,
    plot_xlabel=False,
    xlim=None,
    bin_size_ms=20,
    linestyle=None,
    label_dict={},
    alpha_true=0.5,
    plot_trial_markers=True,
):
    r"""
        Plot a single covariate.
    """
    ax = prep_plt(ax, big=True)
    palette = ['k', *palette]
    linestyle = ['-', *linestyle]
    true_key = ''
    for k in sources:
        if sources[k] == 'True':
            true_key = k
            break
    if not true_key:
        raise ValueError("True key not found in sources")
    for i, k in enumerate(sources):
        plot_kin_dict_trace(
            plot_dict,
            ax,
            xlim,
            model_key=k,
            truth_key=true_key,
            model_label=sources[k],
            color=palette[i],
            linestyle='-' if linestyle is None else linestyle[i],
            alpha=alpha_true if k == true_key else 0.9,
            plot_trial_markers=plot_trial_markers,
        )
    if xlim is not None:
        ax.set_xlim(0, xlim[1] - xlim[0])
    return annotate_normalized_covariate_plot(
        ax,
        palette,
        label,
        plot_xlabel=plot_xlabel,
        bin_size_ms=bin_size_ms,
        label_dict=label_dict)

def annotate_normalized_covariate_plot(ax, palette, cov_label, plot_xlabel=False, bin_size_ms=20, label_dict={}):
    xticks = ax.get_xticks()
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticks * bin_size_ms / 1000)
    if plot_xlabel:
        ax.set_xlabel("Time (s)")

    ax.set_yticks([-1, 0, 1])
    # Set minor y-ticks
    ax.set_yticks(np.arange(-1, 1.1, 0.25), minor=True)
    # Enable minor grid lines
    ax.grid(which="minor", linestyle=":", linewidth="0.5", color="gray", alpha=0.3)
    ax.set_ylabel(f"{label_dict.get(cov_label, cov_label)} (au)")

    legend = ax.legend(
        loc="lower center",  # Positions the legend at the top center
        # bbox_to_anchor=(0.8, 1.1),  # Adjusts the box anchor to the top center
        ncol=3,  # Sets the number of columns equal to the length of the palette to display horizontally
        frameon=False,
        fontsize=20,
    )
    # Make text in legend colored accordingly
    for color, text in zip(palette, legend.get_texts()):
        text.set_color(color)
    return ax, legend


def plot_kin_dict_trace(
    plot_dict: TensorDict,
    ax,
    xlim,
    model_key='behavior_pred',
    model_label="NDT",
    mask_key='behavior_mask',
    truth_key='behavior',
    color="k",
    alpha=0.5,
    linestyle="-",
    plot_trial_markers=True,
):
    r"""
        Paint a single model's predictions (for model comparisons) for a single covariate
    """
    if model_key != truth_key:
        truth = plot_dict['kin'][truth_key]
        pred = plot_dict['kin'][model_key]
        if mask_key in plot_dict['kin'].keys():
            mask = plot_dict['kin'][mask_key]
            truth = truth[mask]
            pred = pred[mask]
        r2_subset = r2_score(truth.cpu().numpy(), pred.cpu().numpy())
        model_label = f"{model_label} ({r2_subset:.2f})"
    if xlim:
        plot_dict = plot_dict[xlim[0] : xlim[1]]
    plot_prediction_spans_dict(ax, plot_dict, color, model_label, model_key=model_key, alpha=alpha, plot_trial_markers=plot_trial_markers, linestyle=linestyle)