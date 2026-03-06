from __future__ import annotations

import argparse

import flwr as fl
from flwr.common import parameters_to_ndarrays

from federated_demo.client import InstitutionClient
from federated_demo.data import load_institution_data
from federated_demo.model import SimpleClassifier, set_parameters
from federated_demo.train_eval import evaluate


class CaptureFinalStrategy(fl.server.strategy.FedAvg):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.final_parameters = None

    def aggregate_fit(self, server_round, results, failures):
        aggregated = super().aggregate_fit(server_round, results, failures)
        if aggregated is None:
            return None

        parameters, metrics = aggregated
        self.final_parameters = parameters
        return parameters, metrics


def run_simulation(rounds: int, local_epochs: int) -> CaptureFinalStrategy:
    strategy = CaptureFinalStrategy(
        min_fit_clients=3,
        min_evaluate_clients=3,
        min_available_clients=3,
        on_fit_config_fn=lambda _: {"local_epochs": local_epochs},
    )

    fl.simulation.start_simulation(
        client_fn=lambda context: InstitutionClient(int(context.node_config["partition-id"])),
        num_clients=3,
        config=fl.server.ServerConfig(num_rounds=rounds),
        strategy=strategy,
        client_resources={"num_cpus": 1},
    )
    return strategy


def evaluate_global_model(strategy: CaptureFinalStrategy) -> None:
    if strategy.final_parameters is None:
        raise RuntimeError("No global parameters captured. Did training run successfully?")

    params_ndarrays = parameters_to_ndarrays(strategy.final_parameters)

    print("\nGlobal model evaluation on each institution:")
    for institution_id in [0, 1, 2]:
        model = SimpleClassifier()
        set_parameters(model, params_ndarrays)
        data = load_institution_data(institution_id)
        loss, acc = evaluate(model, data.test_loader)
        print(f"  Institution {institution_id}: loss={loss:.4f} accuracy={acc:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a 3-institution federated learning demo with Flower + PyTorch.")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--local-epochs", type=int, default=1)
    args = parser.parse_args()

    strategy = run_simulation(rounds=args.rounds, local_epochs=args.local_epochs)
    evaluate_global_model(strategy)


if __name__ == "__main__":
    main()
