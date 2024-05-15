from tqdm import tqdm

from everyvoice.config.type_definitions import TargetTrainingTextRepresentationLevel
from everyvoice.dataloader import BaseDataModule
from everyvoice.dataloader.imbalanced_sampler import ImbalancedDatasetSampler
from everyvoice.model.aligner.config import AlignerConfig
from everyvoice.model.e2e.config import EveryVoiceConfig
from everyvoice.model.feature_prediction.config import FeaturePredictionConfig
from everyvoice.model.vocoder.config import VocoderConfig
from everyvoice.model.vocoder.HiFiGAN_iSTFT_lightning.hfgl.config import (
    HiFiGANTrainingConfig,
    PreprocessingConfig,
)
from everyvoice.model.vocoder.HiFiGAN_iSTFT_lightning.hfgl.dataset import (
    HiFiGANDataModule,
    SpecDataset,
)
from everyvoice.tests.basic_test_case import BasicTestCase
from everyvoice.tests.preprocessed_audio_fixture import PreprocessedAudioFixture
from everyvoice.utils import filter_dataset_based_on_target_text_representation_level


class DataLoaderTest(PreprocessedAudioFixture, BasicTestCase):
    """Basic test for dataloaders"""

    def setUp(self) -> None:
        super().setUp()

        self.config = EveryVoiceConfig(
            contact=BasicTestCase.contact,
            aligner=AlignerConfig(contact=BasicTestCase.contact),
            feature_prediction=FeaturePredictionConfig(contact=BasicTestCase.contact),
            vocoder=VocoderConfig(
                contact=BasicTestCase.contact,
                training=HiFiGANTrainingConfig(
                    training_filelist=PreprocessedAudioFixture.lj_preprocessed
                    / "preprocessed_filelist.psv",
                    validation_filelist=PreprocessedAudioFixture.lj_preprocessed
                    / "validation_preprocessed_filelist.psv",
                ),
                preprocessing=PreprocessingConfig(
                    save_dir=PreprocessedAudioFixture.lj_preprocessed,
                ),
            ),
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

    def test_filter_dataset(self):
        train_dataset = [{"character_tokens": "b", "phone_tokens": ""}] * 4
        with self.assertRaises(SystemExit) as cm:
            filter_dataset_based_on_target_text_representation_level(
                TargetTrainingTextRepresentationLevel.characters,
                train_dataset,
                "training",
                6,
            )
        self.assertEqual(cm.exception.code, 1)
        with self.assertRaises(SystemExit) as cm:
            filter_dataset_based_on_target_text_representation_level(
                TargetTrainingTextRepresentationLevel.ipa_phones,
                train_dataset,
                "training",
                4,
            )
        self.assertEqual(cm.exception.code, 1)
        train_ds = filter_dataset_based_on_target_text_representation_level(
            TargetTrainingTextRepresentationLevel.characters,
            train_dataset,
            "training",
            4,
        )
        val_ds = filter_dataset_based_on_target_text_representation_level(
            TargetTrainingTextRepresentationLevel.characters,
            train_dataset,
            "validation",
            4,
        )
        self.assertEqual(len(train_ds), 4)
        self.assertEqual(len(val_ds), 4)

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
