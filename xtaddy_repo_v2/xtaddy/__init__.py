"""
Dual Spatial-Temporal Shapley Attribution for
Explainable Anomaly Detection in Dynamic Graphs
================================================
Authors: Iyad Assaad Nekka, Hamida Seba,
         Khaled-Walid Hidouci, Karima Amrouche
ESI Algiers / University Claude Bernard Lyon 1
"""

from .explainer import XTADDYExplainer, EdgeExplanation
from .metrics   import summarise_explanations, print_tables
from .visualise import plot_explanation, plot_mean_attribution

__version__ = '2.0.0'
__all__ = [
    'XTADDYExplainer',
    'EdgeExplanation',
    'summarise_explanations',
    'print_tables',
    'plot_explanation',
    'plot_mean_attribution',
]
