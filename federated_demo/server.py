from __future__ import annotations

import argparse
from pathlib import Path

import flwr as fl
import numpy as np
from flwr.common import Parameters, ndarrays_to_parameters, parameters_to_ndarrays

from federated_demo.model import SimpleClassifier, get_parameters


class SaveModelStrategy(fl.server.strategy.FedAvg):
    """FedAvg strategy that stores the latest global parameters on disk."""

    def __init__(self, output_path: Path, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.output_path = output_path

    def aggregate_fit(self, server_round, results, failures):
        aggregated = super().aggregate_fit(server_round, results, failures)
        if aggregated is None:
            return None

        parameters, metrics = aggregated
        self._save_global_parameters(parameters)
        return parameters, metrics

    def _save_global_parameters(self, parameters: Parameters) -> None:
        ndarrays = parameters_to_ndarrays(parameters)
        np.savez(self.output_path, *ndarrays)


def build_initial_parameters() -> Parameters:
    model = SimpleClassifier()
    return ndarrays_to_parameters(get_parameters(model))


def main() -> None:
    parser = argparse.ArgumentParser(description="Start the Flower federated learning server.")
    parser.add_argument("--address", type=str, default="0.0.0.0:8080")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--output", type=Path, default=Path("global_model.npz"))
    args = parser.parse_args()

    strategy = SaveModelStrategy(
        output_path=args.output,
        min_fit_clients=3,
        min_evaluate_clients=3,
        min_available_clients=3,
        initial_parameters=build_initial_parameters(),
        on_fit_config_fn=lambda _: {"local_epochs": 1},
    )

    fl.server.start_server(
        server_address=args.address,
        config=fl.server.ServerConfig(num_rounds=args.rounds),
        strategy=strategy,
    )


if __name__ == "__main__":
    main()
