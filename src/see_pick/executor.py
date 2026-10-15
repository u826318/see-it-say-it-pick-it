"""Phase-two seam; competition code never instantiates this protocol."""

from __future__ import annotations

from typing import Protocol

from .domain import PlanDecision, Scene


class RobotExecutor(Protocol):
    def execute(
        self,
        decision: PlanDecision,
        target_pixel: tuple[int, int],
        scene: Scene,
    ) -> None: ...
