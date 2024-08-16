from everyvoice.evaluation import (
    calculate_objective_metrics_from_single_path,
    load_squim_objective_model,
)
from everyvoice.tests.basic_test_case import BasicTestCase


class EvaluationTest(BasicTestCase):
    def test_squim_evaluation(self):
        model, sr = load_squim_objective_model()
        stoi, pesq, si_sdr = calculate_objective_metrics_from_single_path(
            self.data_dir / "LJ010-0008.wav", model, sr
        )
        self.assertLess(stoi, 1)
        self.assertEqual(round(pesq, 2), 3.88)
        self.assertEqual(round(si_sdr, 2), 28.64)
