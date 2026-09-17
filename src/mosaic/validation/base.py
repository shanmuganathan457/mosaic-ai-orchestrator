"""MOSAIC Abstract Validation Engine Interface."""

from abc import ABC, abstractmethod
from typing import List

from mosaic.domain.models.schemas import Action, CaseState, PolicyRule, ValidationResult


class BaseValidationEngine(ABC):
    """Abstract Base Class for state and policy validation engines."""

    @abstractmethod
    def validate(
        self,
        case_state: CaseState,
        proposed_actions: List[Action],
        policies: List[PolicyRule],
    ) -> ValidationResult:
        """Evaluates proposed actions against current case state and active policies."""
        pass
