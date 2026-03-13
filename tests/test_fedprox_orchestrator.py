import unittest

from domain.federated.fedprox_orchestrator import FedProxOrchestrator, InstitutionUpdate


class _StubModel:
    def __init__(self):
        self._params = {"weights": [0.0, 0.0], "bias": [0.0]}

    def parameters(self):
        return self._params

    def load_parameters(self, params):
        self._params = {name: list(values) for name, values in params.items()}


class FedProxOrchestratorTests(unittest.TestCase):
    def test_aggregate_weighted_parameters_uses_sample_sizes(self):
        aggregated = FedProxOrchestrator._aggregate_weighted_parameters(
            updates=[
                InstitutionUpdate(
                    institution_id="bank_1",
                    num_samples=1,
                    parameters={"weights": [1.0, 3.0], "bias": [2.0]},
                    local_loss=0.5,
                    parameter_delta_l2=0.0,
                ),
                InstitutionUpdate(
                    institution_id="bank_2",
                    num_samples=3,
                    parameters={"weights": [5.0, 7.0], "bias": [6.0]},
                    local_loss=0.4,
                    parameter_delta_l2=0.0,
                ),
            ],
            parameter_names=["weights", "bias"],
        )

        self.assertEqual(aggregated["weights"].tolist(), [4.0, 6.0])
        self.assertEqual(aggregated["bias"].tolist(), [5.0])

    def test_run_round_rejects_zero_total_samples(self):
        model = _StubModel()
        orchestrator = FedProxOrchestrator(institutions=[], initial_model=model, proximal_mu=0.1)

        with self.assertRaises(RuntimeError):
            orchestrator.run_round(round_index=1)


if __name__ == "__main__":
    unittest.main()
