from __future__ import annotations

import argparse

import numpy as np

from federated_demo.data import load_institution_data
from federated_demo.model import SimpleClassifier, set_parameters
from federated_demo.train_eval import evaluate


def _load_npz_parameters(path: str) -> list[np.ndarray]:
    data = np.load(path)
    return [data[key] for key in data.files]


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the saved global model on an institution's test data.")
    parser.add_argument("--model-path", type=str, default="global_model.npz")
    parser.add_argument("--institution-id", type=int, required=True, choices=[0, 1, 2])
    args = parser.parse_args()

    model = SimpleClassifier()
    params = _load_npz_parameters(args.model_path)
    set_parameters(model, params)

    data = load_institution_data(args.institution_id)
    loss, accuracy = evaluate(model, data.test_loader)
    print(f"institution={args.institution_id} loss={loss:.4f} accuracy={accuracy:.4f}")


if __name__ == "__main__":
    main()
