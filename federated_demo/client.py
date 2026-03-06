from __future__ import annotations

import argparse

import flwr as fl

from federated_demo.data import load_institution_data
from federated_demo.model import SimpleClassifier, get_parameters, set_parameters
from federated_demo.train_eval import evaluate, train_one_epoch


class InstitutionClient(fl.client.NumPyClient):
    def __init__(self, institution_id: int) -> None:
        self.institution_id = institution_id
        self.model = SimpleClassifier()
        self.data = load_institution_data(institution_id)

    def get_parameters(self, config):  # type: ignore[override]
        return get_parameters(self.model)

    def fit(self, parameters, config):  # type: ignore[override]
        set_parameters(self.model, parameters)
        local_epochs = int(config.get("local_epochs", 1))
        for _ in range(local_epochs):
            train_one_epoch(self.model, self.data.train_loader)
        return get_parameters(self.model), len(self.data.train_loader.dataset), {"institution_id": self.institution_id}

    def evaluate(self, parameters, config):  # type: ignore[override]
        set_parameters(self.model, parameters)
        loss, accuracy = evaluate(self.model, self.data.test_loader)
        return float(loss), len(self.data.test_loader.dataset), {"accuracy": float(accuracy)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Start a Flower client for a specific institution.")
    parser.add_argument("--institution-id", type=int, required=True, choices=[0, 1, 2])
    parser.add_argument("--server-address", type=str, default="127.0.0.1:8080")
    args = parser.parse_args()

    fl.client.start_numpy_client(
        server_address=args.server_address,
        client=InstitutionClient(args.institution_id),
    )


if __name__ == "__main__":
    main()
