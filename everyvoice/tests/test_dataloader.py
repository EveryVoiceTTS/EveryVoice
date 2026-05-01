#!/usr/bin/env python

import sys

from pytest import fixture, main, raises

from everyvoice.config.type_definitions import TargetTrainingTextRepresentationLevel
from everyvoice.dataloader import BaseDataModule
from everyvoice.dataloader.imbalanced_sampler import ImbalancedDatasetSampler
from everyvoice.model.vocoder.config import VocoderConfig
from everyvoice.model.vocoder.HiFiGAN_iSTFT_lightning.hfgl.config import (
    HiFiGANTrainingConfig,
    PreprocessingConfig,
)
from everyvoice.model.vocoder.HiFiGAN_iSTFT_lightning.hfgl.dataset import (
    HiFiGANDataModule,
    SpecDataset,
)
from everyvoice.tests.preprocessed_audio_fixture import PreprocessedAudioFixture
from everyvoice.tests.stubs import TEST_CONTACT, mute_logger
from everyvoice.utils import filter_dataset_based_on_target_text_representation_level


@fixture
def config() -> VocoderConfig:
    return VocoderConfig(
        contact=TEST_CONTACT,
        training=HiFiGANTrainingConfig(
            training_filelist=PreprocessedAudioFixture.lj_preprocessed
            / "preprocessed_filelist.psv",
            validation_filelist=PreprocessedAudioFixture.lj_preprocessed
            / "validation_preprocessed_filelist.psv",
        ),
        preprocessing=PreprocessingConfig(
            save_dir=PreprocessedAudioFixture.lj_preprocessed,
        ),
    )


class TestDataLoader(PreprocessedAudioFixture):
    """Basic test for dataloaders"""

    def test_base_data_loader(self, config):
        bdm = BaseDataModule(config)
        with raises(NotImplementedError):
            bdm.load_dataset()

    def test_spec_dataset(self, config):
        dataset = SpecDataset(
            config.training.filelist_loader(config.training.training_filelist),
            config,
            use_segments=True,
        )
        for sample in dataset:
            spec, audio, basename, spec_from_audio = sample
            assert isinstance(basename, str)
            assert spec.size() == spec_from_audio.size()
            assert spec.size(0) == config.preprocessing.audio.n_mels
            assert spec.size(1) == config.preprocessing.audio.vocoder_segment_size / (
                config.preprocessing.audio.fft_hop_size
                * (
                    config.preprocessing.audio.output_sampling_rate
                    // config.preprocessing.audio.input_sampling_rate
                )
            )

    def test_hifi_data_loader(self, config):
        hfgdm = HiFiGANDataModule(config)
        hfgdm.load_dataset()
        assert len(hfgdm.train_dataset) == 5

    def test_filter_dataset(self):
        train_dataset = [{"character_tokens": "b", "phone_tokens": ""}] * 4
        with raises(SystemExit) as cm:
            with mute_logger("everyvoice.utils"):
                filter_dataset_based_on_target_text_representation_level(
                    TargetTrainingTextRepresentationLevel.characters,
                    train_dataset,
                    "training",
                    6,
                )
        assert cm.value.code == 1
        with raises(SystemExit) as cm:
            with mute_logger("everyvoice.utils"):
                filter_dataset_based_on_target_text_representation_level(
                    TargetTrainingTextRepresentationLevel.ipa_phones,
                    train_dataset,
                    "training",
                    4,
                )
        assert cm.value.code == 1
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
        assert len(train_ds) == 4
        assert len(val_ds) == 4

    def test_hifi_ft_data_loader(self):
        """TODO: can't make this test until I generate some synthesized samples"""
        pass

    def test_feature_prediction_data_loader(self):
        # TODO: once feature prediction is done
        pass

    def test_e2e_data_module(self):
        # TODO: once e2e is done
        pass

    def test_imbalanced_sampler(self, config):
        dataset = SpecDataset(
            config.training.filelist_loader(config.training.training_filelist),
            config,
            use_segments=True,
        )
        sampler = ImbalancedDatasetSampler(dataset)
        print(sampler.weights)
        sample = list(sampler)
        assert len(sample) == 5


if __name__ == "__main__":
    main(sys.argv)
