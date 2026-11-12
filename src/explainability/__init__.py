"""
Explainability package for ST-GRAD (Phase 6 & 7).
Provides:
- Integrated Gradients feature/temporal/spatial attribution.
- MVCA view-level counterfactual interventions.
- Static prior vs adaptive graph interventions.
- Quantitative faithfulness (deletion/insertion, parameter randomization).
- Cross-seed and input perturbation stability.
"""

from src.explainability.attribution import IntegratedGradients
from src.explainability.interventions import ViewInterventions, GraphInterventions
from src.explainability.faithfulness import deletion_insertion_test, model_randomization_test
from src.explainability.stability import evaluate_seed_stability, evaluate_perturbation_stability

__all__ = [
    'IntegratedGradients',
    'ViewInterventions',
    'GraphInterventions',
    'deletion_insertion_test',
    'model_randomization_test',
    'evaluate_seed_stability',
    'evaluate_perturbation_stability'
]
