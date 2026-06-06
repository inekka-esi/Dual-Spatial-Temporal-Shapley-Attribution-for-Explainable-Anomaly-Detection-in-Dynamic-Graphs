"""
X-TADDY: Visualisation Utilities
=================================
Authors: Iyad Assaad Nekka, Hamida Seba,
         Khaled-Walid Hidouci, Karima Amrouche
ESI Algiers / University Claude Bernard Lyon 1

Functions for plotting spatial and temporal Shapley attribution bar charts.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import Optional, List
from .explainer import EdgeExplanation


CORAL = '#D85A30'
BLUE  = '#185FA5'
LIGHT = '#f5f5f3'


def plot_explanation(
    expl:       EdgeExplanation,
    k:          int           = 5,
    W:          int           = 2,
    edge_label: str           = '(u, v)',
    snap_label: str           = 't*',
    save_path:  Optional[str] = None,
    figsize:    tuple         = (10, 4),
):
    """
    Plot spatial and temporal Shapley attribution bar charts for one edge.

    Parameters
    ----------
    expl : EdgeExplanation
    k : int — number of neighbors in TADDY (default 5)
    W : int — window size (default 2)
    edge_label : str — displayed in title
    snap_label : str — displayed in title
    save_path : str or None — save figure if provided
    figsize : tuple
    """
    fig, (ax_sp, ax_tp) = plt.subplots(1, 2, figsize=figsize)
    fig.patch.set_facecolor('white')

    # ── Spatial ──────────────────────────────────────────────────────────
    phi_sp  = expl.phi_spatial
    n_nodes = k + 2
    labels  = ['u', 'v'] + [f'n{i}' for i in range(1, k + 1)]
    colors  = [CORAL if v >= 0 else BLUE for v in phi_sp]

    bars = ax_sp.barh(
        labels[:n_nodes], phi_sp[:n_nodes],
        color=colors[:n_nodes], alpha=0.85, height=0.6,
    )
    ax_sp.axvline(0, color='#888780', linewidth=0.6, linestyle='--')
    ax_sp.set_xlabel('Shapley value φ', fontsize=11, color='#3d3d3a')
    ax_sp.set_title('Spatial attribution — neighbor tokens',
                    fontsize=12, fontweight='bold', color='#1a1a18', pad=8)
    ax_sp.tick_params(axis='both', labelsize=10, colors='#3d3d3a')
    ax_sp.spines['top'].set_visible(False)
    ax_sp.spines['right'].set_visible(False)
    ax_sp.spines['left'].set_color('#d3d1c7')
    ax_sp.spines['bottom'].set_color('#d3d1c7')
    ax_sp.set_facecolor(LIGHT)

    for bar, val in zip(bars, phi_sp[:n_nodes]):
        xpos = val + 0.002 if val >= 0 else val - 0.002
        ax_sp.text(xpos, bar.get_y() + bar.get_height() / 2,
                   f'{val:+.3f}', va='center',
                   ha='left' if val >= 0 else 'right',
                   fontsize=9, color='#2c2c2a')

    # ── Temporal ─────────────────────────────────────────────────────────
    phi_tp      = expl.phi_temporal
    step_labels = ['t (current)'] + [f't-{l} (prev.)' for l in range(1, W)]
    tp_colors   = [CORAL if v >= 0 else BLUE for v in phi_tp]

    bars_tp = ax_tp.bar(
        step_labels, phi_tp,
        color=tp_colors, alpha=0.85, width=0.5,
    )
    ax_tp.axhline(0, color='#888780', linewidth=0.6, linestyle='--')
    ax_tp.set_ylabel('Shapley value φ', fontsize=11, color='#3d3d3a')
    ax_tp.set_title('Temporal attribution — lookback steps',
                    fontsize=12, fontweight='bold', color='#1a1a18', pad=8)
    ax_tp.tick_params(axis='both', labelsize=10, colors='#3d3d3a')
    ax_tp.spines['top'].set_visible(False)
    ax_tp.spines['right'].set_visible(False)
    ax_tp.spines['left'].set_color('#d3d1c7')
    ax_tp.spines['bottom'].set_color('#d3d1c7')
    ax_tp.set_facecolor(LIGHT)

    for bar, val in zip(bars_tp, phi_tp):
        ypos = val + 0.005 if val >= 0 else val - 0.005
        ax_tp.text(bar.get_x() + bar.get_width() / 2, ypos,
                   f'{val:+.3f}', va='bottom' if val >= 0 else 'top',
                   ha='center', fontsize=9, color='#2c2c2a')

    # ── Legend ────────────────────────────────────────────────────────────
    fig.legend(
        handles=[
            mpatches.Patch(color=CORAL, alpha=0.85,
                           label='positive (raises anomaly score)'),
            mpatches.Patch(color=BLUE,  alpha=0.85,
                           label='suppressive (lowers anomaly score)'),
        ],
        loc='lower center', ncol=2, fontsize=10, frameon=False,
        labelcolor='#2c2c2a', bbox_to_anchor=(0.5, -0.04),
    )

    fig.suptitle(
        f'X-TADDY explanation for edge {edge_label} at snapshot {snap_label}  '
        f'(score={expl.true_score:.3f}, fidelity={expl.fidelity:.4f})',
        fontsize=11, color='#1a1a18', y=1.02,
    )

    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f'Figure saved to {save_path}')

    return fig, (ax_sp, ax_tp)


def plot_mean_attribution(
    explanations: List[EdgeExplanation],
    k:            int           = 5,
    W:            int           = 2,
    title:        str           = 'Mean X-TADDY Attribution',
    save_path:    Optional[str] = None,
    figsize:      tuple         = (10, 4),
):
    """
    Plot mean spatial and temporal Shapley attributions across multiple edges.
    """
    from copy import deepcopy
    mock               = deepcopy(explanations[0])
    mock.phi_spatial   = np.stack([e.phi_spatial  for e in explanations]).mean(0)
    mock.phi_temporal  = np.stack([e.phi_temporal for e in explanations]).mean(0)
    mock.true_score    = float(np.mean([e.true_score for e in explanations]))
    mock.fidelity      = float(np.mean([e.fidelity   for e in explanations]))

    fig, axes = plot_explanation(
        mock, k=k, W=W,
        edge_label=f'(mean over {len(explanations)} edges)',
        save_path=save_path, figsize=figsize,
    )
    fig.suptitle(f'{title}  (n={len(explanations)} edges)',
                 fontsize=11, color='#1a1a18', y=1.02)
    return fig, axes
