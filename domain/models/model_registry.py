"""Registry for federated-stage model factories."""

from __future__ import annotations

import logging
from typing import Protocol, TypeAlias, TypedDict

from domain.models.device_selector import DeviceSelector
from domain.models.federated_model_protocol import FederatedModelProtocol


class LogisticRegressionModelOptions(TypedDict):
    model_type: str


class TabNetModelOptions(TypedDict):
    model_type: str
    decision_dim: int
    attention_dim: int
    steps: int
    relaxation_factor: float
    sparsity_weight: float


ModelOptions: TypeAlias = LogisticRegressionModelOptions | TabNetModelOptions


class ModelBuilderProtocol(Protocol):
    def __call__(
        self,
        n_features: int,
        config: ModelOptions,
    ) -> FederatedModelProtocol: ...


class ModelFactoryProtocol(Protocol):
    def __call__(self, n_features: int) -> FederatedModelProtocol: ...


ModelFactory = ModelBuilderProtocol
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

    def get_factory(self, model_type: str, config: ModelOptions) -> ModelFactoryProtocol:
        if model_type not in self._factories:
            raise ValueError(
                f"Unknown model_type '{model_type}'. Registered model types: {', '.join(self.list_model_types())}"
            )

        factory = self._factories[model_type]
        return lambda n_features: factory(n_features, config)


MODEL_REGISTRY = ModelRegistry()

def _build_tabnet_model(n_features: int, config: ModelOptions) -> FederatedModelProtocol:
    from domain.models.tabnet_model import TabNetModel

    selector = DeviceSelector()
    available_devices = selector.available_devices()
    device = selector.select_best_device()
    LOGGER.info(
        "tabnet_device_selection selected=%s available=%s",
        device,
        ",".join(available_devices),
    )
    return TabNetModel.initialize(
        n_features=n_features,
        decision_dim=int(config.get("decision_dim", 16)),
        attention_dim=int(config.get("attention_dim", 16)),
        n_steps=int(config.get("steps", 3)),
        relaxation_factor=float(config.get("relaxation_factor", 1.5)),
        sparsity_weight=float(config.get("sparsity_weight", 1e-4)),
        device=device,
    )


MODEL_REGISTRY.register("tabnet", _build_tabnet_model)
