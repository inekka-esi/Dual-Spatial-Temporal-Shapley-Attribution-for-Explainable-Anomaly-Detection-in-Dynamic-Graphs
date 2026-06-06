Dual Spatial-Temporal Shapley Attribution for Explainable Anomaly Detection in Dynamic Graphs
Authors: Iyad Assaad Nekka · Hamida Seba · Khaled-Walid Hidouci · Karima Amrouche
Affiliations: Higher National School of Computer Science (ESI), Algiers, Algeria · University Claude Bernard Lyon 1, Lyon, France

What is X-TADDY?
TADDY is a strong Transformer-based dynamic graph anomaly detector — but like most deep learning methods, it is a black box: it flags an anomalous edge and tells you nothing about why.

X-TADDY wraps a trained TADDY model with a post-hoc Shapley explanation layer. For each flagged edge, it produces two attribution vectors:

Spatial attribution — which neighbor nodes drove the anomaly score
Temporal attribution — which historical snapshot carried the most signal
X-TADDY leaves TADDY's weights, training, and inference completely unchanged. Detection AUC is preserved exactly (ΔAUC = 0.0000).

Prior to this work, all explanation quality metrics are undefined for TADDY: there is no attribution vector to evaluate. After applying our framework, Fidelity reaches 0.9562, Sufficiency 0.9658, and positive Comprehensiveness is confirmed across all datasets.

Key Design: Embedding-Space Masking
The critical technical contribution is the graph-aware coalition masking. Naive index-space masking is wrong for TADDY: index 0 in its intimacy-rank table maps to the highest-ranked neighbor by PPR score — the opposite of neutral. X-TADDY operates in TADDY's dense embedding space. For a coalition S of active tokens, excluded tokens are replaced by the mean embedding of active ones, keeping masked sequences within the convex hull of active embeddings and in-distribution for the Transformer.

Results
Detection AUC (Table 2)
Method	UCI	BTC-Alpha	BTC-OTC
TADDY (published)	0.8370	0.9423	0.9425
TADDY (our implementation)	0.8284	0.9554	0.9596
+ Explainability layer (ours)	0.8284	0.9554	0.9596
ΔAUC = 0.0000 — detection performance preserved exactly by post-hoc construction.

Explainability Introduced (Table 3, averaged across all datasets)
Metric	TADDY	Ours
Fidelity	N/A	0.9562
Sufficiency	N/A	0.9658
Comprehensiveness	N/A	0.0576
Spatial attribution	N/A	yes
Temporal attribution	N/A	yes
Explanations per edge	0	150
N/A = undefined (no attribution vector exists for TADDY); not zero or unmeasured.

Fidelity Across Edge Categories (Table 4)
Dataset	Category	n	Fidelity
UCI	Top-50 TP	50	0.8743
UCI	All TP	691	0.9023
UCI	False Positives	50	0.8428
UCI	Low-Conf TP	50	0.8101
BTC-Alpha	Top-50 TP	50	0.9957
BTC-Alpha	All TP	706	0.9598
BTC-Alpha	False Positives	50	0.9957
BTC-Alpha	Low-Conf TP	50	0.9893
BTC-OTC	Top-50 TP	50	0.9982
BTC-OTC	All TP	1074	0.9735
BTC-OTC	False Positives	50	0.9977
BTC-OTC	Low-Conf TP	50	0.8158
Sensitivity to Top-k Budget (Table 5)
Dataset	k=3	k=5	k=7	k=10
UCI-Message	0.8578	0.8720	0.8976	0.9454
Bitcoin-Alpha	0.9951	0.9957	0.9961	0.9975
Bitcoin-OTC	0.9994	0.9982	0.9981	0.9988
Repository Structure
Explainable-TADDY-X-TADDY/
├── LICENSE
├── README.md
├── requirements.txt
├── xtaddy/
│   ├── __init__.py       # Public API
│   ├── explainer.py      # XTADDYExplainer — core Shapley framework
│   ├── metrics.py        # Fidelity, Sufficiency, Comprehensiveness, reporting
│   └── visualise.py      # Shapley bar chart plotting
└── examples/
    └── run_evaluation.py # Complete Colab evaluation pipeline
Quick Start (Google Colab, T4 GPU)
Upload the xtaddy/ folder to /content/xtaddy/, then:

python examples/run_evaluation.py
This handles everything automatically: clones TADDY, patches for Python 3.12 compatibility, prepares data, trains TADDY (100 epochs per dataset), runs X-TADDY, and prints Tables 2-5. Runtime ~3 hours on T4.

Standalone Usage
from xtaddy import XTADDYExplainer, plot_explanation

# Wrap a trained DynADModel
explainer = XTADDYExplainer(model, num_samples=150, top_k=5)

# Explain a single anomalous edge
expl = explainer.explain_edge(
    edge_idx=0, snap=7,
    int_emb=int_emb, hop_emb=hop_emb, time_emb=time_emb,
)

print(f"True score        : {expl.true_score:.4f}")
print(f"Fidelity          : {expl.fidelity:.4f}")
print(f"Sufficiency       : {expl.sufficiency:.4f}")
print(f"Comprehensiveness : {expl.comprehensiveness:.4f}")
print(f"Spatial phi       : {expl.phi_spatial}")    # shape [k+2]
print(f"Temporal phi      : {expl.phi_temporal}")   # shape [W]

# Plot attribution
plot_explanation(expl, k=5, W=2, save_path='explanation.pdf')

# Explain top-50 true-positive edges
results = explainer.explain_top_edges(
    loaded_data, int_emb, hop_emb, time_emb, preds, y_test, n=50
)
Design Notes
Post-hoc by design. The framework wraps a trained, frozen TADDY model. No retraining, no modification. ΔAUC = 0.0000 across all datasets.

Axiomatic guarantees. Shapley values satisfy efficiency, symmetry, linearity, and the null-player property. No scalar attribution method provides these guarantees.

Sigmoid clarification. TADDY's cls_y layer returns raw logits. sigmoid() is applied exactly once outside cls_y to obtain scores in [0,1]. This is explicit at every call site.

Metrics in probability space. All metrics are computed post-sigmoid, matching the operationally meaningful score analysts see.

Citation
@inproceedings{nekka2026dualshapley,
  title     = {Dual Spatial-Temporal Shapley Attribution for
               Explainable Anomaly Detection in Dynamic Graphs},
  author    = {Nekka, Iyad Assaad and Seba, Hamida and
               Hidouci, Khaled-Walid and Amrouche, Karima},
  booktitle = {Proceedings of CoopIS 2026},
  year      = {2026}
}
References
Liu et al., "Anomaly Detection in Dynamic Graphs via Transformer", IEEE TKDE, 2022.
Duval & Malliaros, "GraphSVX: Shapley Value Explanations for GNNs", ECML-PKDD, 2021.
Shapley, "A value for n-person games", 1953.
