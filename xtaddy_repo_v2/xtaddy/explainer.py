"""
Dual Spatial-Temporal Shapley Attribution for
Explainable Anomaly Detection in Dynamic Graphs
================================================
Authors: Iyad Assaad Nekka, Hamida Seba,
         Khaled-Walid Hidouci, Karima Amrouche
ESI Algiers / University Claude Bernard Lyon 1

Key design — embedding-space masking
-------------------------------------
Excluded tokens are replaced by the mean embedding of the active coalition,
computed AFTER EdgeEncoding (dense vector space). Index 0 in TADDY's
intimacy-rank table maps to the highest-ranked neighbor by PPR score, making
it maximally informative rather than neutral. Replacing with the coalition
mean keeps masked sequences within the convex hull of active embeddings.

Sigmoid clarification
---------------------
TADDY's cls_y layer returns raw logits. sigmoid() is applied exactly once
outside cls_y at every call site.

Note on probability vs. logit space
-------------------------------------
All metrics are computed in probability space (post-sigmoid). For TADDY's
datasets, scores are well-distributed away from the saturation region.

References
----------
Liu et al., TADDY, TKDE 2022.
Duval & Malliaros, GraphSVX, ECML 2021.
Shapley, 1953.
"""

import torch
import numpy as np
import scipy.special
from dataclasses import dataclass
from typing import List
from sklearn.linear_model import Ridge


@dataclass
class EdgeExplanation:
    """
    Full framework output for one explained edge.

    Attributes
    ----------
    true_score : float
        TADDY's anomaly score in [0,1]. sigmoid() applied once to raw logits.
    phi : np.ndarray [D]
        Per-token Shapley values (WLS surrogate coefficients).
    phi_spatial : np.ndarray [k+2]
        Aggregated by node position. [0]=u, [1]=v, [2..k+1]=neighbors.
        Endpoint tokens receive high attribution by design (TADDY pooling).
        Neighbor values measure additional causal contribution beyond baseline.
    phi_temporal : np.ndarray [W]
        Aggregated by lookback step. [0]=current snapshot, [1]=one step back.
    base_value : float
        Surrogate intercept (expected score with no tokens active).
    fidelity : float
        1 - |true_score - score_with_top_k_only|.
    sufficiency : float
        Score when only top-k tokens are active.
        High value confirms those tokens are sufficient for detection.
    comprehensiveness : float
        true_score - score_without_top_k_tokens.
        Complementary to Sufficiency (removes top-k instead of keeping only).
        Positive value confirms those tokens were necessary.
    sparsity : float
        1 - top_k / D. Fixed by construction.
    top_k_indices : List[int]
        Token positions with largest |phi_j|.
    """
    true_score:        float
    phi:               np.ndarray
    phi_spatial:       np.ndarray
    phi_temporal:      np.ndarray
    base_value:        float
    fidelity:          float
    sufficiency:       float
    comprehensiveness: float
    sparsity:          float
    top_k_indices:     List[int]


class XTADDYExplainer:
    """
    Post-hoc Shapley explainer for a trained TADDY (DynADModel).

    Parameters
    ----------
    model : DynADModel
        A fully trained TADDY model.
    num_samples : int
        Coalition samples per edge (default 150).
    top_k : int
        Tokens retained for metric measurement (default 5).
    ridge_alpha : float
        L2 regularisation for WLS Ridge surrogate (default 0.01).
    random_state : int
        Seed for coalition sampling (default 42).
    """

    def __init__(self, model, num_samples=150, top_k=5,
                 ridge_alpha=0.01, random_state=42):
        self.model       = model
        self.num_samples = num_samples
        self.top_k       = top_k
        self.ridge_alpha = ridge_alpha
        self.rng         = np.random.default_rng(random_state)
        self.model.eval()
        self.k = model.config.k
        self.W = model.config.window_size
        self.D = (self.k + 2) * self.W

    # ── Public API ────────────────────────────────────────────────────────

    def explain_edge(self, edge_idx, snap, int_emb, hop_emb, time_emb):
        """Explain one edge. Returns EdgeExplanation."""
        true_score, h_base = self._get_score_and_embedding(
            int_emb, hop_emb, time_emb, snap, edge_idx)
        D = self.D

        Z, weights = self._sample_coalitions(D)
        fz         = self._score_coalitions(h_base, Z, D)

        reg = Ridge(alpha=self.ridge_alpha, fit_intercept=True)
        reg.fit(Z.numpy(), fz.numpy(),
                sample_weight=np.clip(weights.numpy(), 0.0, 1000.0))
        phi        = reg.coef_
        base_value = float(reg.intercept_)

        phi_spatial, phi_temporal = self._aggregate(phi)
        top_k_indices     = list(np.argsort(np.abs(phi))[-self.top_k:])
        fidelity          = self._compute_fidelity(h_base, phi, true_score, D)
        sufficiency       = self._compute_sufficiency(h_base, phi, D)
        comprehensiveness = self._compute_comprehensiveness(
            h_base, phi, true_score, D)

        return EdgeExplanation(
            true_score        = true_score,
            phi               = phi,
            phi_spatial       = phi_spatial,
            phi_temporal      = phi_temporal,
            base_value        = base_value,
            fidelity          = fidelity,
            sufficiency       = sufficiency,
            comprehensiveness = comprehensiveness,
            sparsity          = 1.0 - self.top_k / D,
            top_k_indices     = top_k_indices,
        )

    def explain_top_edges(self, loaded_data, int_emb, hop_emb, time_emb,
                          preds, y_test, n=50):
        """Explain the top-n highest-scoring true-positive edges."""
        candidates = []
        for si, snap in enumerate(loaded_data['snap_test']):
            p = preds[si]; y = y_test[si]
            for ei in range(len(p)):
                if y[ei] == 1:
                    candidates.append((float(p[ei]), snap, ei))
        candidates.sort(key=lambda x: x[0], reverse=True)

        print(f"Explaining {min(n, len(candidates))} edges "
              f"(D={self.D}, top_k={self.top_k}) ...")
        results = []
        for rank, (_, snap, ei) in enumerate(candidates[:n]):
            results.append(self.explain_edge(ei, snap, int_emb, hop_emb, time_emb))
            if (rank + 1) % 10 == 0:
                print(f"  {rank+1}/{min(n,len(candidates))} | "
                      f"Mean Fidelity: {np.mean([r.fidelity for r in results]):.4f}")
        return results

    def explain_edges_list(self, edges, int_emb, hop_emb, time_emb):
        """
        Explain an arbitrary list of edges.
        Each element: tuple (score, snap, edge_idx).
        """
        results = []
        for rank, (_, snap, ei) in enumerate(edges):
            results.append(self.explain_edge(ei, snap, int_emb, hop_emb, time_emb))
            if (rank + 1) % 10 == 0:
                print(f"  {rank+1}/{len(edges)} | "
                      f"Mean Fidelity: {np.mean([r.fidelity for r in results]):.4f}")
        return results

    # ── Internal helpers ─────────────────────────────────────────────────

    def _get_score_and_embedding(self, int_emb, hop_emb, time_emb, snap, edge_idx):
        ii = int_emb[snap][edge_idx:edge_idx+1]
        hh = hop_emb[snap][edge_idx:edge_idx+1]
        tt = time_emb[snap][edge_idx:edge_idx+1]
        with torch.no_grad():
            out, _ = self.model.forward(ii, hh, tt)
            true_score = torch.sigmoid(out).item()   # sigmoid once
            h_base = self.model.transformer.embeddings(
                init_pos_ids=ii, hop_dis_ids=hh, time_dis_ids=tt)
        return true_score, h_base.detach()

    def _sample_coalitions(self, D):
        """
        GraphSVX Smarter sampler adapted to D temporal tokens.
        Anchors (full/empty coalitions) receive weight 1000 to enforce
        Shapley efficiency. Rankings stable to weight variations due to
        Ridge regularisation.
        """
        ns = self.num_samples
        Z  = torch.zeros(ns, D)
        Z[0] = torch.ones(D)    # full coalition
        Z[1] = torch.zeros(D)   # empty coalition
        i = 2
        for j in range(D):
            if i >= ns: break
            v = torch.ones(D);  v[j] = 0.0; Z[i] = v; i += 1
            if i >= ns: break
            v = torch.zeros(D); v[j] = 1.0; Z[i] = v; i += 1
        if i < ns:
            Z[i:] = torch.from_numpy(
                self.rng.integers(0, 2, size=(ns-i, D)).astype(np.float32))
        s = Z.sum(dim=1).long()
        w = []
        for si in s:
            si = si.item()
            if si == 0 or si == D:
                w.append(1000.0)
            else:
                denom = scipy.special.comb(D, si) * si * (D - si)
                w.append(float((D-1)/denom) if denom > 0 else 1.0)
        return Z, torch.tensor(w, dtype=torch.float32).clamp(max=1000.0)

    def _score_coalition(self, h_base, mask):
        """
        Replace inactive tokens with mean(active embeddings), run TADDY.
        sigmoid applied once — forward_from_embeddings returns logits.
        """
        h = h_base.clone()
        active = mask.bool()
        h_mean = h[0, active, :].mean(dim=0) if active.sum() > 0 \
                 else torch.zeros(h.shape[-1])
        h[0, ~active, :] = h_mean
        with torch.no_grad():
            out, _ = self.model.forward_from_embeddings(h)
            return torch.sigmoid(out).item()   # sigmoid once

    def _score_coalitions(self, h_base, Z, D):
        fz = torch.zeros(Z.shape[0])
        for s in range(Z.shape[0]):
            fz[s] = self._score_coalition(h_base, Z[s].bool())
        return fz

    def _aggregate(self, phi):
        tokens_per_step = self.k + 2
        phi_spatial  = np.zeros(tokens_per_step)
        phi_temporal = np.zeros(self.W)
        for j, p in enumerate(phi):
            phi_spatial[j % tokens_per_step]  += p
            if j // tokens_per_step < self.W:
                phi_temporal[j // tokens_per_step] += p
        return phi_spatial, phi_temporal

    def _top_k_mask(self, phi, D):
        mask = torch.zeros(D).bool()
        for idx in np.argsort(np.abs(phi))[-min(self.top_k, D):]:
            mask[idx] = True
        return mask

    def _compute_fidelity(self, h_base, phi, true_score, D):
        return float(1.0 - abs(true_score -
               self._score_coalition(h_base, self._top_k_mask(phi, D))))

    def _compute_sufficiency(self, h_base, phi, D):
        return self._score_coalition(h_base, self._top_k_mask(phi, D))

    def _compute_comprehensiveness(self, h_base, phi, true_score, D):
        mask = ~self._top_k_mask(phi, D)
        return float(true_score - self._score_coalition(h_base, mask))
