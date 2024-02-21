#!/usr/bin/env python

from tqdm import tqdm

from everyvoice.dataloader import BaseDataModule
from everyvoice.dataloader.imbalanced_sampler import ImbalancedDatasetSampler
from everyvoice.model.aligner.config import AlignerConfig
from everyvoice.model.e2e.config import EveryVoiceConfig
from everyvoice.model.feature_prediction.config import FeaturePredictionConfig
from everyvoice.model.vocoder.config import VocoderConfig
from everyvoice.model.vocoder.HiFiGAN_iSTFT_lightning.hfgl.config import (
    HiFiGANTrainingConfig,
)
from everyvoice.model.vocoder.HiFiGAN_iSTFT_lightning.hfgl.dataset import (
    HiFiGANDataModule,
    SpecDataset,
)
from everyvoice.tests.basic_test_case import BasicTestCase
from everyvoice.tests.test_preprocessing import PreprocessingTest


class DataLoaderTest(BasicTestCase):
    """Basic test for dataloaders"""

    lj_preprocessed = PreprocessingTest.lj_preprocessed

    def setUp(self) -> None:
        super().setUp()
        PreprocessingTest.preprocess()  # Generate some preprocessed test data
        self.config = EveryVoiceConfig(
            contact=self.contact,
            aligner=AlignerConfig(contact=self.contact),
            feature_prediction=FeaturePredictionConfig(contact=self.contact),
            vocoder=VocoderConfig(
                contact=self.contact,
                training=HiFiGANTrainingConfig(
                    training_filelist=self.lj_preprocessed
                    / "training_preprocessed_filelist.psv",
                    validation_filelist=self.lj_preprocessed
                    / "validation_preprocessed_filelist.psv",
                ),
            ),
        )
        self.config.vocoder.preprocessing.save_dir = self.lj_preprocessed
        self.config.vocoder.training.training_filelist = (
            self.lj_preprocessed / "preprocessed_filelist.psv"
        )

    def test_base_data_loader(self):
        bdm = BaseDataModule(self.config.aligner)
        with self.assertRaises(NotImplementedError):
            bdm.load_dataset()

    def test_spec_dataset(self):
        dataset = SpecDataset(
            self.config.vocoder.training.filelist_loader(
                self.config.vocoder.training.training_filelist
            ),
            self.config.vocoder,
            use_segments=True,
        )
        for sample in tqdm(dataset):
            spec, audio, basename, spec_from_audio = sample
            self.assertTrue(isinstance(basename, str))
            self.assertEqual(spec.size(), spec_from_audio.size())
            self.assertEqual(
                spec.size(0), self.config.vocoder.preprocessing.audio.n_mels
            )
            self.assertEqual(
                spec.size(1),
                self.config.vocoder.preprocessing.audio.vocoder_segment_size
                / (
                    self.config.vocoder.preprocessing.audio.fft_hop_size
                    * (
                        self.config.vocoder.preprocessing.audio.output_sampling_rate
                        // self.config.vocoder.preprocessing.audio.input_sampling_rate
                    )
                ),
            )

    def test_hifi_data_loader(self):
        hfgdm = HiFiGANDataModule(self.config.vocoder)
        hfgdm.load_dataset()
        self.assertEqual(len(hfgdm.train_dataset), 5)

    def test_hifi_ft_data_loader(self):
        """TODO: can't make this test until I generate some synthesized samples"""
        pass

    def test_feature_prediction_data_loader(self):
        # TODO: once feature prediction is done
        pass

    def test_e2e_data_module(self):
        # TODO: once e2e is done
        pass

    def test_imbalanced_sampler(self):
        dataset = SpecDataset(
            self.config.vocoder.training.filelist_loader(
                self.config.vocoder.training.training_filelist
            ),
            self.config.vocoder,
            use_segments=True,
        )
        sampler = ImbalancedDatasetSampler(dataset)
        sample = list(sampler)
        self.assertEqual(len(sample), 5)
