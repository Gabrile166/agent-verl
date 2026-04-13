"""Evaluation helpers for pairwise phi ranking benchmark."""

from .metrics import (
    compute_accuracy_by_pair_type,
    compute_invalid_response_rate,
    compute_pairwise_choice_acc,
    compute_subset_accuracies,
)
from .model_interface import ManualInputModel, ModelInterface, OpenAICompatibleModel
from .prompt_builder import PairwiseComparisonPromptBuilder

__all__ = [
    "PairwiseComparisonPromptBuilder",
    "ModelInterface",
    "OpenAICompatibleModel",
    "ManualInputModel",
    "compute_pairwise_choice_acc",
    "compute_subset_accuracies",
    "compute_invalid_response_rate",
    "compute_accuracy_by_pair_type",
]
