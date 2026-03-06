"""Registry for federated-stage model factories."""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from typing import Any

from domain.models.basic_model import LogisticRegressionModel
from domain.models.device_selector import DeviceSelector

ModelFactory = Callable[[int, Mapping[str, Any]], Any]
LOGGER = logging.getLogger(__name__)


class ModelRegistry:
    """Simple in-process model registry keyed by model type."""

    def __init__(self) -> None:
        self._factories: dict[str, ModelFactory] = {}

    def register(self, model_type: str, factory: ModelFactory) -> None:
        self._factories[model_type] = factory

    def has(self, model_type: str) -> bool:
        return model_type in self._factories

    def list_model_types(self) -> list[str]:
        return sorted(self._factories.keys())

    def get_factory(self, model_type: str, config: Mapping[str, Any]) -> Callable[[int], Any]:
        if model_type not in self._factories:
            raise ValueError(
                f"Unknown model_type '{model_type}'. Registered model types: {', '.join(self.list_model_types())}"
            )

        factory = self._factories[model_type]
        return lambda n_features: factory(n_features, config)


MODEL_REGISTRY = ModelRegistry()


def _build_logistic_regression_model(n_features: int, config: Mapping[str, Any]) -> Any:
    del config
    return LogisticRegressionModel.initialize(n_features)


def _build_tabnet_model(n_features: int, config: Mapping[str, Any]) -> Any:
    from domain.models.tabnet_model import TabNetModel

    selector = DeviceSelector()
    available_devices = selector.available_devices()
    configured_device = config.get("tabnet_device")
    device = str(configured_device) if configured_device else selector.select_best_device()
    LOGGER.info(
        "tabnet_device_selection selected=%s available=%s",
        device,
        ",".join(available_devices),
    )
    return TabNetModel.initialize(
        n_features=n_features,
        decision_dim=int(config.get("tabnet_decision_dim", 16)),
        attention_dim=int(config.get("tabnet_attention_dim", 16)),
        n_steps=int(config.get("tabnet_steps", 3)),
        relaxation_factor=float(config.get("tabnet_relaxation_factor", 1.5)),
        sparsity_weight=float(config.get("tabnet_sparsity_weight", 1e-4)),
        device=device,
    )


MODEL_REGISTRY.register("logistic_regression", _build_logistic_regression_model)
MODEL_REGISTRY.register("tabnet", _build_tabnet_model)
