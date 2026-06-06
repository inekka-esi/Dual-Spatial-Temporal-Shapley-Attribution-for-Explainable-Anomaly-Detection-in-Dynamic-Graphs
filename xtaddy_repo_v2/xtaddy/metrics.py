"""
X-TADDY: Evaluation Metrics and Reporting Utilities
=====================================================
Authors: Iyad Assaad Nekka, Hamida Seba,
         Khaled-Walid Hidouci, Karima Amrouche
ESI Algiers / University Claude Bernard Lyon 1

Functions for computing, aggregating, and printing results aligned to
Tables 2-5 of the paper.
"""

import numpy as np
from typing import List, Dict
from .explainer import EdgeExplanation


def summarise_explanations(results: List[EdgeExplanation]) -> Dict:
    """
    Aggregate Fidelity, Sufficiency, Comprehensiveness, and Sparsity
    across a list of EdgeExplanation objects.

    Parameters
    ----------
    results : list of EdgeExplanation

    Returns
    -------
    dict with keys:
        fidelity_mean, fidelity_std, fidelity_raw,
        sufficiency_mean, comprehensiveness_mean,
        sparsity_mean, n_edges,
        phi_spatial_mean, phi_temporal_mean
    """
    fids  = [r.fidelity          for r in results]
    suffs = [r.sufficiency       for r in results]
    comps = [r.comprehensiveness for r in results]
    spars = [r.sparsity          for r in results]

    sp_mat = np.stack([r.phi_spatial  for r in results], axis=0)
    tp_mat = np.stack([r.phi_temporal for r in results], axis=0)

    return {
        'fidelity_mean':          float(np.mean(fids)),
        'fidelity_std':           float(np.std(fids)),
        'fidelity_raw':           fids,
        'sufficiency_mean':       float(np.mean(suffs)),
        'comprehensiveness_mean': float(np.mean(comps)),
        'sparsity_mean':          float(np.mean(spars)),
        'n_edges':                len(results),
        'phi_spatial_mean':       sp_mat.mean(axis=0).tolist(),
        'phi_temporal_mean':      tp_mat.mean(axis=0).tolist(),
    }


def print_tables(session: dict):
    """
    Print Tables 2, 3, 4, and 5 from the paper using SESSION data.

    Parameters
    ----------
    session : dict
        Keys are dataset names ('uci', 'btc_alpha', 'btc_otc').
        Each value must contain:
            det_auc         — float
            top50_summary   — output of summarise_explanations()
        Optionally:
            expanded        — dict with keys 'all_tp', 'fp', 'low_tp',
                              each an output of summarise_explanations()
            sensitivity     — dict mapping k -> fidelity for top-50 TP
    """

    # ── TABLE 2: Detection AUC ────────────────────────────────────────────
    published = {'uci': 0.8370, 'btc_alpha': 0.9423, 'btc_otc': 0.9425}

    print(f"\n{'='*62}")
    print(f"  TABLE 2 — ANOMALY DETECTION AUC-ROC")
    print(f"  10% anomaly | 100 epochs | 50/50 split")
    print(f"{'='*62}")
    print(f"  {'Method':30s} {'UCI':>8} {'BTC-α':>8} {'BTC-OTC':>8}")
    print(f"  {'-'*55}")
    print(f"  {'TADDY (published)':30s} "
          f"{published['uci']:>8.4f} {published['btc_alpha']:>8.4f} "
          f"{published['btc_otc']:>8.4f}")

    our_aucs = {ds: session[ds]['det_auc'] for ds in session}
    print(f"  {'TADDY (our implementation)':30s} "
          f"{our_aucs.get('uci',0):>8.4f} "
          f"{our_aucs.get('btc_alpha',0):>8.4f} "
          f"{our_aucs.get('btc_otc',0):>8.4f}")
    print(f"  {'+ Explainability layer (ours)':30s} "
          f"{our_aucs.get('uci',0):>8.4f} "
          f"{our_aucs.get('btc_alpha',0):>8.4f} "
          f"{our_aucs.get('btc_otc',0):>8.4f}")
    print(f"  {'-'*55}")
    print(f"  Delta AUC = 0.0000 by post-hoc construction.")

    # ── TABLE 3: N/A vs Ours ─────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"  TABLE 3 — EXPLAINABILITY INTRODUCED BY THE FRAMEWORK")
    print(f"  Top-50 TP per dataset | top-k=5 | averaged across all datasets")
    print(f"{'='*65}")

    all_fids  = [session[ds]['top50_summary']['fidelity_mean']          for ds in session]
    all_suffs = [session[ds]['top50_summary']['sufficiency_mean']       for ds in session]
    all_comps = [session[ds]['top50_summary']['comprehensiveness_mean'] for ds in session]

    print(f"  {'Metric':25s}  {'TADDY':>8}  {'Ours':>10}")
    print(f"  {'-'*48}")
    print(f"  {'Fidelity':25s}  {'N/A':>8}  {np.mean(all_fids):>10.4f}")
    print(f"  {'Sufficiency':25s}  {'N/A':>8}  {np.mean(all_suffs):>10.4f}")
    print(f"  {'Comprehensiveness':25s}  {'N/A':>8}  {np.mean(all_comps):>10.4f}")
    print(f"  {'Spatial attribution':25s}  {'N/A':>8}  {'yes':>10}")
    print(f"  {'Temporal attribution':25s}  {'N/A':>8}  {'yes':>10}")
    print(f"  {'Explanations per edge':25s}  {'0':>8}  {'150':>10}")
    print(f"  {'-'*48}")
    print(f"  N/A = undefined (no attribution vector exists for TADDY).")

    # Per-dataset breakdown
    print(f"\n  Per-dataset (top-50 TP):")
    print(f"  {'Dataset':12s} {'Fidelity':>10} {'Sufficiency':>13} {'Comp.':>8}")
    print(f"  {'-'*48}")
    for ds in session:
        s = session[ds]['top50_summary']
        print(f"  {ds:12s} {s['fidelity_mean']:>10.4f} "
              f"{s['sufficiency_mean']:>13.4f} "
              f"{s['comprehensiveness_mean']:>8.4f}")

    # ── TABLE 4: Expanded evaluation ─────────────────────────────────────
    if any('expanded' in session[ds] for ds in session):
        print(f"\n{'='*65}")
        print(f"  TABLE 4 — FIDELITY ACROSS EDGE CATEGORIES")
        print(f"{'='*65}")
        print(f"  {'Dataset':12s} {'Category':22s} {'n':>6} "
              f"{'Fidelity':>10} {'Std':>8}")
        print(f"  {'-'*62}")

        cat_labels = {
            'top50':  'Top-50 TP',
            'all_tp': 'All TP',
            'fp':     'False Positives',
            'low_tp': 'Low-Conf TP',
        }

        for ds in session:
            if 'expanded' not in session[ds]:
                continue
            for cat, label in cat_labels.items():
                if cat == 'top50':
                    s = session[ds]['top50_summary']
                else:
                    exp = session[ds]['expanded']
                    if cat not in exp:
                        continue
                    s = exp[cat]
                print(f"  {ds:12s} {label:22s} {s['n_edges']:>6} "
                      f"{s['fidelity_mean']:>10.4f} "
                      f"{s['fidelity_std']:>8.4f}")
            print(f"  {'-'*62}")

    # ── TABLE 5: Sensitivity to top-k ────────────────────────────────────
    if any('sensitivity' in session[ds] for ds in session):
        print(f"\n{'='*62}")
        print(f"  TABLE 5 — FIDELITY SENSITIVITY TO TOP-K BUDGET")
        print(f"{'='*62}")
        print(f"  {'Dataset':14s} {'k=3':>10} {'k=5':>10} "
              f"{'k=7':>10} {'k=10':>10}")
        print(f"  {'-'*55}")
        for ds in session:
            if 'sensitivity' not in session[ds]:
                continue
            sens = session[ds]['sensitivity']
            row  = [f"{sens.get(k, 0.0):.4f}" for k in [3, 5, 7, 10]]
            print(f"  {ds:14s} " + " ".join(f"{v:>10}" for v in row))
        print(f"  {'-'*55}")
