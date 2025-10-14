#!/usr/bin/env python

from unittest import TestCase, main

from everyvoice.evaluation import (
    calculate_objective_metrics_from_single_path,
    calculate_subjective_metrics_from_single_path,
    load_squim_objective_model,
    load_squim_subjective_model,
)
from everyvoice.tests.stubs import TEST_DATA_DIR


class EvaluationTest(TestCase):
    def test_squim_evaluation(self):
        model, sr = load_squim_objective_model()
        subj_model, subj_sr = load_squim_subjective_model()
        stoi, pesq, si_sdr = calculate_objective_metrics_from_single_path(
            TEST_DATA_DIR / "LJ010-0008.wav", model, sr
        )
        mos = calculate_subjective_metrics_from_single_path(
            TEST_DATA_DIR / "LJ010-0008.wav",
            TEST_DATA_DIR / "lj" / "wavs" / "LJ050-0269.wav",
            subj_model,
            subj_sr,
        )
        self.assertEqual(round(mos, 2), 4.47)
        self.assertLess(stoi, 1)
        self.assertEqual(round(pesq, 2), 3.88)
        self.assertEqual(round(si_sdr, 2), 28.64)


if __name__ == "__main__":
    main()
