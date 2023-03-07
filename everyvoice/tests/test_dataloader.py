from pathlib import Path
from unittest import TestCase, main

from tqdm import tqdm

from everyvoice.dataloader import BaseDataModule
from everyvoice.model.e2e.config import EveryVoiceConfig
from everyvoice.model.vocoder.HiFiGAN_iSTFT_lightning.hfgl.dataset import (
    HiFiGANDataModule,
    SpecDataset,
)


class DataLoaderTest(TestCase):
    """Basic test for dataloaders"""

    def setUp(self) -> None:
        self.config = EveryVoiceConfig.load_config_from_path()
        self.lj_preprocessed = Path(__file__).parent / "data" / "lj" / "preprocessed"
        self.config.vocoder.preprocessing.save_dir = self.lj_preprocessed
        self.config.vocoder.training.filelist = (
            self.lj_preprocessed / "preprocessed_filelist.psv"
        )

    def test_base_data_loader(self):
        bdm = BaseDataModule(self.config.aligner)
        with self.assertRaises(NotImplementedError):
            bdm.load_dataset()

    def test_spec_dataset(self):
        dataset = SpecDataset(
            self.config.vocoder.training.filelist_loader(
                self.config.vocoder.training.filelist
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
                    self.config.vocoder.preprocessing.audio.fft_hop_frames
                    * (
                        self.config.vocoder.preprocessing.audio.output_sampling_rate
                        // self.config.vocoder.preprocessing.audio.input_sampling_rate
                    )
                ),
            )

    def test_hifi_data_loader(self):
        hfgdm = HiFiGANDataModule(self.config.vocoder)
        hfgdm.load_dataset()
        self.assertEqual(len(hfgdm.dataset), 5)

    def test_hifi_ft_data_loader(self):
        """TODO: can't make this test until I generate some synthesized samples"""
        pass

    def test_feature_prediction_data_loader(self):
        # TODO: once feature prediction is done
        pass

    def test_e2e_data_module(self):
        # TODO: once e2e is done
        pass


if __name__ == "__main__":
    main()
