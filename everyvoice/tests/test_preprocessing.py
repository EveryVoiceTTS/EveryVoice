import doctest
import shutil
import tempfile
from pathlib import Path

import torch
import torchaudio
from pydantic_core._pydantic_core import ValidationError
from torch import float32

import everyvoice.preprocessor
import everyvoice.text.text_processor
from everyvoice.config.preprocessing_config import (
    AudioConfig,
    AudioSpecTypeEnum,
    PreprocessingConfig,
)
from everyvoice.config.shared_types import init_context
from everyvoice.model.aligner.config import AlignerConfig
from everyvoice.model.e2e.config import FeaturePredictionConfig
from everyvoice.model.vocoder.config import VocoderConfig
from everyvoice.preprocessor import Preprocessor
from everyvoice.tests.basic_test_case import BasicTestCase
from everyvoice.tests.preprocessed_audio_fixture import PreprocessedAudioFixture
from everyvoice.tests.stubs import (
    capture_stdout,
    monkeypatch,
    mute_logger,
    patch_logger,
)
from everyvoice.utils import generic_psv_filelist_reader


class PreprocessingTest(PreprocessedAudioFixture, BasicTestCase):
    """Unit tests for preprocessing steps"""

    filelist = generic_psv_filelist_reader(BasicTestCase.data_dir / "metadata.psv")

    def test_run_doctest(self):
        """Run doctests in everyvoice.text.text_processing"""
        results = doctest.testmod(everyvoice.text.text_processor)
        self.assertFalse(results.failed, results)

    # def test_compute_stats(self):
    #     feat_prediction_config = EveryVoiceConfig.load_config_from_path().feature_prediction
    #     preprocessor = Preprocessor(feat_prediction_config)
    #     preprocessor.compute_stats()
    # self.assertEqual(
    #     self.preprocessor.config["preprocessing"]["audio"]["mel_mean"],
    #     -4.018,
    #     places=3,
    # )
    # self.assertEqual(
    #     self.preprocessor.config["preprocessing"]["audio"]["mel_std"],
    #     4.017,
    #     places=3,
    # )

    def test_read_filelist(self):
        self.assertEqual(self.filelist[0]["basename"], "LJ050-0269")

    def test_no_permissions(self):
        no_permissions_args = self.fp_config.model_dump()
        no_permissions_args["preprocessing"]["source_data"][0][
            "permissions_obtained"
        ] = False
        with tempfile.TemporaryDirectory() as tmpdir:
            with init_context({"writing_config": Path(tmpdir)}):
                with self.assertRaises(ValueError):
                    FeaturePredictionConfig(**no_permissions_args)

    def test_process_audio_for_alignment(self):
        config = AlignerConfig(contact=self.contact)
        for entry in self.filelist[1:]:
            # This just applies the SOX effects
            audio, sr = self.preprocessor.process_audio(
                self.wavs_dir / (entry["basename"] + ".wav"),
                sox_effects=config.preprocessing.source_data[0].sox_effects,
                hop_size=config.preprocessing.audio.fft_hop_size,
            )
            self.assertEqual(sr, 22050)
            self.assertEqual(audio.dtype, float32)

    def test_remove_silence(self):
        audio_path_with_silence = str(
            self.data_dir / ("440tone-with-leading-trailing-silence.wav")
        )
        config = FeaturePredictionConfig(contact=self.contact)
        sox_effects = config.preprocessing.source_data[0].sox_effects + [
            [
                "silence",
                "1",
                "0.1",
                "0.1%",
            ],
            ["reverse"],  # reverse the clip to trim silence from end
            ["silence", "1", "0.1", "0.1%"],
            ["reverse"],  # reverse the clip again to revert to the right direction :)
        ]
        raw_audio, raw_sr = torchaudio.load(audio_path_with_silence)
        processed_audio, processed_sr = self.preprocessor.process_audio(
            audio_path_with_silence,
            sox_effects=sox_effects,
            hop_size=config.preprocessing.audio.fft_hop_size,
        )

        self.assertEqual(
            raw_sr, processed_sr, "Sampling Rate should not be changed by default"
        )
        self.assertEqual(
            raw_audio.size()[1] / raw_sr,
            3.5,
            "Should be exactly 3.5 seconds of audio at 44100 Hz sampling rate",
        )
        self.assertEqual(
            round(processed_audio.size()[0] / processed_sr, 2),
            2.5,
            msg="Should be about half a second of silence removed from the beginning and end",
        )
        # should work with resampling too
        rs_processed_audio, rs_processed_sr = self.preprocessor.process_audio(
            audio_path_with_silence,
            resample_rate=22050,
            sox_effects=sox_effects,
            hop_size=config.preprocessing.audio.fft_hop_size,
        )
        self.assertEqual(
            round(rs_processed_audio.size()[0] / rs_processed_sr, 2),
            2.5,
            msg="Should be about half a second of silence removed from the beginning and end when resampled too",
        )

    def test_process_empty_audio(self):
        for fn in ["empty.wav", "zeros.wav"]:
            audio, sr = self.preprocessor.process_audio(self.data_dir / fn)
            self.assertEqual(audio, None)
            self.assertEqual(sr, None)

    def test_process_audio(self):
        import torchaudio

        for entry in self.filelist[1:]:
            audio, sr = self.preprocessor.process_audio(
                self.wavs_dir / (entry["basename"] + ".wav"), hop_size=256
            )
            self.assertEqual(sr, 22050)
            self.assertEqual(audio.dtype, float32)
        # test that truncating according to hop size actually happened
        raw_audio, raw_sr = torchaudio.load(
            str(self.wavs_dir / (entry["basename"] + ".wav"))
        )
        # remove channel info
        raw_audio = raw_audio.squeeze()
        self.assertNotEqual(raw_audio.size(0), audio.size(0))
        self.assertLess(raw_audio.size(0) - audio.size(0), 256)
        # changing the hop size changes how much is removed
        diff_hop_audio, _ = self.preprocessor.process_audio(
            self.wavs_dir / (entry["basename"] + ".wav"), hop_size=35
        )
        self.assertNotEqual(audio.size(0), diff_hop_audio.size(0))
        # we should never truncate more than a portion of a single frame
        self.assertLess(raw_audio.size(0) - diff_hop_audio.size(0), 35)
        with self.assertRaises(ValueError):  # missing hop_size
            self.preprocessor.process_audio(
                self.wavs_dir / (entry["basename"] + ".wav")
            )

    def test_spectral_feats(self):
        linear_vocoder_config = VocoderConfig(
            contact=self.contact,
            preprocessing=PreprocessingConfig(
                audio=AudioConfig(spec_type=AudioSpecTypeEnum.linear)
            ),
        )
        complex_vocoder_config = VocoderConfig(
            contact=self.contact,
            preprocessing=PreprocessingConfig(
                audio=AudioConfig(spec_type=AudioSpecTypeEnum.raw)
            ),
        )
        linear_preprocessor = Preprocessor(linear_vocoder_config)
        complex_preprocessor = Preprocessor(complex_vocoder_config)

        for entry in self.filelist[1:]:
            audio, _ = self.preprocessor.process_audio(
                self.wavs_dir / (entry["basename"] + ".wav"),
                hop_size=linear_vocoder_config.preprocessing.audio.fft_hop_size,
            )

            # ming024_feats = np.load(
            #     self.data_dir
            #     / "ming024"
            #     / ("eng-LJSpeech-mel-" + entry["basename"] + ".npy")
            # )
            feats = self.preprocessor.extract_spectral_features(
                audio, self.preprocessor.input_spectral_transform
            )
            linear_feats = linear_preprocessor.extract_spectral_features(
                audio, linear_preprocessor.input_spectral_transform
            )
            complex_feats = complex_preprocessor.extract_spectral_features(
                audio, complex_preprocessor.input_spectral_transform, normalize=False
            )
            # check data is same number of mels
            self.assertEqual(
                feats.size(0),
                self.preprocessor.config.preprocessing.audio.n_mels,
            )
            # Check linear spec has right number of fft bins
            self.assertEqual(
                linear_feats.size(0),
                linear_preprocessor.config.preprocessing.audio.n_fft // 2 + 1,
            )
            # check all same length
            self.assertEqual(feats.size(1), linear_feats.size(1))
            # check all same length
            self.assertEqual(complex_feats.size(1), linear_feats.size(1))

    def test_bad_pitch(self):
        """Some files don't have any pitch values so we should make sure we handle these properly"""
        pyworld_config = FeaturePredictionConfig(
            contact=self.contact, preprocessing=PreprocessingConfig()
        )
        preprocessor_pyworld = Preprocessor(pyworld_config)
        audio = torch.zeros(22050)
        feats = self.preprocessor.extract_spectral_features(
            audio, self.preprocessor.input_spectral_transform
        )
        frame_pitch_pyworld = preprocessor_pyworld.extract_pitch(audio)
        self.assertEqual(frame_pitch_pyworld.max(), 0)
        self.assertEqual(frame_pitch_pyworld.min(), 0)
        self.assertEqual(frame_pitch_pyworld.size(0), feats.size(1))

    def test_pitch(self):
        pyworld_config = VocoderConfig(
            contact=self.contact, preprocessing=PreprocessingConfig()
        )
        preprocessor_pyworld = Preprocessor(pyworld_config)

        for entry in self.filelist[1:]:
            audio, _ = self.preprocessor.process_audio(
                self.wavs_dir / (entry["basename"] + ".wav"),
                hop_size=pyworld_config.preprocessing.audio.fft_hop_size,
            )
            dur_path = (
                self.lj_preprocessed
                / "duration"
                / self.preprocessor.sep.join(
                    [
                        entry["basename"],
                        entry.get("speaker", "default"),
                        entry.get("language", "default"),
                        "duration.pt",
                    ]
                )
            )
            durs = torch.load(dur_path)
            feats = self.preprocessor.extract_spectral_features(
                audio, self.preprocessor.input_spectral_transform
            )
            # ming024_pitch = np.load(
            #     self.data_dir
            #     / "ming024"
            #     / ("eng-LJSpeech-pitch-" + entry["basename"] + ".npy")
            # )
            # Ensure avg pitch for each phone
            frame_pitch_pyworld = preprocessor_pyworld.extract_pitch(audio)
            pyworld_phone_avg_energy = preprocessor_pyworld.average_data_by_durations(
                frame_pitch_pyworld, durs
            )
            # Ensure avg pitch for each phone
            self.assertEqual(len(durs), pyworld_phone_avg_energy.size(0))
            # Ensure same number of frames
            self.assertEqual(frame_pitch_pyworld.size(0), feats.size(1))

    # TODO: test nans: torch.any(torch.Tensor([[torch.nan, 2]]).isnan())

    def test_duration(self):
        for entry in self.filelist[1:]:
            audio, _ = self.preprocessor.process_audio(
                self.wavs_dir / (entry["basename"] + ".wav"), hop_size=256
            )
            dur_path = (
                self.lj_preprocessed
                / "duration"
                / self.preprocessor.sep.join(
                    [
                        entry["basename"],
                        entry.get("speaker", "default"),
                        entry.get("language", "default"),
                        "duration.pt",
                    ]
                )
            )
            durs = torch.load(dur_path)
            feats = self.preprocessor.extract_spectral_features(
                audio, self.preprocessor.input_spectral_transform
            )
            # ming024_durs = np.load(
            #     self.data_dir
            #     / "ming024"
            #     / ("eng-LJSpeech-duration-" + entry["basename"] + ".npy")
            # )
            # Ensure durations same number of frames as spectral features
            # note: this is off by a few frames due to mismatches in hop size between the aligner the test data
            # was trained with and the settings defined by the spectral transform function here.
            # It would be a problem if it weren't  but it's not really relevant since we're using jointly learned alignments now.
            self.assertTrue(feats.size(1) - int(sum(durs)) <= 10)

    def test_energy(self):
        frame_energy_config = VocoderConfig(
            contact=self.contact, preprocessing=PreprocessingConfig()
        )
        preprocessor = Preprocessor(frame_energy_config)
        for entry in self.filelist[1:]:
            audio, _ = self.preprocessor.process_audio(
                self.wavs_dir / (entry["basename"] + ".wav"),
                hop_size=frame_energy_config.preprocessing.audio.fft_hop_size,
            )
            dur_path = (
                self.lj_preprocessed
                / "duration"
                / self.preprocessor.sep.join(
                    [
                        entry["basename"],
                        entry.get("speaker", "default"),
                        entry.get("language", "default"),
                        "duration.pt",
                    ]
                )
            )
            durs = torch.load(dur_path)
            # ming024_energy = np.load(
            #     self.data_dir
            #     / "ming024"
            #     / ("eng-LJSpeech-energy-" + entry["basename"] + ".npy")
            # )
            feats = self.preprocessor.extract_spectral_features(
                audio, self.preprocessor.input_spectral_transform
            )

            frame_energy = preprocessor.extract_energy(feats)
            phone_avg_energy = preprocessor.average_data_by_durations(
                frame_energy, durs
            )
            # Ensure avg energy for each phone
            self.assertEqual(phone_avg_energy.size(0), len(durs))
            # Ensure same number of frames
            self.assertEqual(frame_energy.size(0), feats.size(1))

    def test_sanity(self):
        """TODO: make sanity checking code for each type of data, maybe also data analysis tooling"""
        pass

    def test_text_processing(self):
        with tempfile.TemporaryDirectory(
            prefix="test_text_processing"
        ) as tempdir, init_context({"writing_config": Path(tempdir)}):
            characters_eng_filelist = (
                self.data_dir / "metadata_characters_supported_lang.psv"
            )
            characters_default_filelist = (
                self.data_dir / "metadata_characters_no_supported_lang.psv"
            )
            arpabet_filelist = self.data_dir / "metadata_arpabet.psv"
            phones_filelist = self.data_dir / "metadata_phones.psv"
            mixed_representation_filelist = (
                self.data_dir / "metadata_mixed_representation.psv"
            )
            fp_config = FeaturePredictionConfig(**self.fp_config.model_dump())
            filelists_to_test = [
                {
                    "path": characters_eng_filelist,
                    "contains_characters": True,
                    "contains_phones": True,
                },  # will take characters and apply eng g2p
                {
                    "path": characters_default_filelist,
                    "contains_characters": True,
                    "contains_phones": False,
                },  # will just tokenize characters
                {
                    "path": arpabet_filelist,
                    "contains_characters": False,
                    "contains_phones": True,
                },  # will convert arpabet to phones
                {
                    "path": phones_filelist,
                    "contains_characters": False,
                    "contains_phones": True,
                },  # will just tokenize phones
                {
                    "path": mixed_representation_filelist,
                    "contains_characters": True,
                    "contains_phones": True,
                },  # will tokenize characters and tokenize phones
            ]
            for filelist_test_info in filelists_to_test:
                with tempfile.TemporaryDirectory(prefix="inputs", dir=".") as tmpdir:
                    tmpdir = Path(tmpdir)
                    preprocessed_dir = tmpdir / "preprocessed"
                    preprocessed_dir.mkdir(parents=True, exist_ok=True)
                    output_filelist = preprocessed_dir / "preprocessed_filelist.psv"
                    shutil.copyfile(filelist_test_info["path"], output_filelist)
                    fp_config.preprocessing.source_data[0].filelist = (
                        filelist_test_info["path"]
                    )
                    fp_config.preprocessing.save_dir = preprocessed_dir
                    preprocessor = Preprocessor(fp_config)
                    with capture_stdout() as output, mute_logger(
                        "everyvoice.preprocessor"
                    ):
                        preprocessor.preprocess(
                            output_path=str(output_filelist),
                            cpus=1,
                            to_process=["text", "pfs"],
                        )
                    self.assertIn(
                        "You've finished preprocessing: text", output.getvalue()
                    )
                    processed_filelist = preprocessor.load_filelist(output_filelist)
                    characters = [
                        x["character_tokens"]
                        for x in processed_filelist
                        if "character_tokens" in x
                    ]
                    phones = [
                        x["phone_tokens"]
                        for x in processed_filelist
                        if "phone_tokens" in x
                    ]
                    phonological_features = [
                        torch.load(f)
                        for f in sorted(
                            list((output_filelist.parent / "pfs").glob("*.pt"))
                        )
                    ]
                    for i, utt_phones in enumerate(phones):
                        # Phonlogical features are derived from phones so they should be of equal length
                        self.assertEqual(
                            len(utt_phones.split("/")),
                            phonological_features[i].size(0),
                            utt_phones.split("/"),
                        )

                    if filelist_test_info["contains_characters"]:
                        self.assertEqual(
                            len(characters),
                            5,
                            f'failed finding characters in {filelist_test_info["path"]}',
                        )
                        self.assertEqual(
                            characters[0],
                            "t/h/e/ /e/s/s/e/n/t/i/a/l/ /t/e/r/m/s/ /o/f/ /s/u/c/h/ /m/e/m/o/r/a/n/d/a/ /m/i/g/h/t/ /w/e/l/l/ /b/e/ /e/m/b/o/d/i/e/d/ /i/n/ /a/n/ /e/x/e/c/u/t/i/v/e/ /o/r/d/e/r/.",
                            f'failed in {filelist_test_info["path"]}',
                        )
                    if filelist_test_info["contains_phones"]:
                        self.assertEqual(
                            len(phones),
                            5,
                            f'failed finding phones in {filelist_test_info["path"]}',
                        )
                        if "arpabet" in filelist_test_info["path"].stem:
                            # arpabet uses space for phone boundaries
                            self.assertEqual(
                                phones[0],
                                "ð/ /ʌ/ /e/ /s/ /e/ /n/ /ʃ/ /ʌ/ /l/ /t/ /r/ /m/ /z/ /ʌ/ /v/ /s/ /ʌ/ /c/h/ /m/ /e/ /m/ /r/ /æ/ /n/ /d/ /ʌ/ /m/ /a/ɪ/ /t/ /w/ /e/ /l/ /b/ /i/ /ɪ/ /m/ /b/ /ɑ/ /d/ /i/ /d/ /ɪ/ /n/ /æ/ /n/ /ɪ/ /g/ /z/ /e/ /k/ /j/ /ʌ/ /t/ /ɪ/ /v/ /ɔ/ /r/ /d/ /r/ /.",
                            )
                        else:
                            self.assertEqual(
                                phones[0],
                                "ð/ʌ/ /ɛ/s/ɛ/n/ʃ/ʌ/l/ /t/ɜ˞/m/z/ /ʌ/v/ /s/ʌ/t/ʃ/ /m/ɛ/m/ɜ˞/æ/n/d/ʌ/ /m/a/ɪ/t/ /w/ɛ/l/ /b/i/ /ɪ/m/b/ɑ/d/i/d/ /ɪ/n/ /æ/n/ /ɪ/ɡ/z/ɛ/k/j/ʌ/t/ɪ/v/ /ɔ/ɹ/d/ɜ˞/.",
                                f'failed in {filelist_test_info["path"]}',
                            )

    def get_simple_config(self, tmpdir: str | Path):
        """Create a simple config for testing"""
        tmpdir = Path(tmpdir)
        lj_preprocessed = tmpdir / "preprocessed"
        lj_filelist = lj_preprocessed / "preprocessed_filelist.psv"

        fp_config = FeaturePredictionConfig(contact=self.contact)
        fp_config.preprocessing.source_data[0].data_dir = self.data_dir / "lj" / "wavs"

        full_filelist = self.data_dir / "metadata.psv"
        partial_filelist = tmpdir / "partial-metadata.psv"
        with open(partial_filelist, mode="w") as f_out:
            with open(full_filelist) as f_in:
                lines = list(f_in)
                for line in lines[:4]:
                    f_out.write(line)
        fp_config.preprocessing.source_data[0].filelist = full_filelist
        fp_config.preprocessing.save_dir = lj_preprocessed

        to_process = ("audio", "energy", "pitch", "attn", "text", "spec")
        return (fp_config, lj_filelist, full_filelist, partial_filelist, to_process)

    def test_incremental_preprocess(self):
        with tempfile.TemporaryDirectory(
            prefix="test_incremental_preprocess", dir="."
        ) as tmpdir:
            (
                fp_config,
                lj_filelist,
                full_filelist,
                partial_filelist,
                to_process,
            ) = self.get_simple_config(tmpdir)

            fp_config.preprocessing.source_data[0].filelist = partial_filelist
            with capture_stdout() as output, mute_logger("everyvoice.preprocessor"):
                Preprocessor(fp_config).preprocess(
                    output_path=lj_filelist, cpus=1, to_process=to_process
                )
            self.assertRegex(output.getvalue(), r"processed files *3")
            self.assertRegex(output.getvalue(), r"previously processed files *0")

            fp_config.preprocessing.source_data[0].filelist = full_filelist
            with capture_stdout() as output, mute_logger("everyvoice.preprocessor"):
                Preprocessor(fp_config).preprocess(
                    output_path=lj_filelist, cpus=1, to_process=to_process
                )
            self.assertRegex(output.getvalue(), r"processed files *2")
            self.assertRegex(output.getvalue(), r"previously processed files *3")
            with capture_stdout() as output, mute_logger("everyvoice.preprocessor"):
                Preprocessor(fp_config).preprocess(
                    output_path=lj_filelist,
                    cpus=1,
                    overwrite=True,
                    to_process=to_process,
                    debug=True,
                )
            self.assertRegex(output.getvalue(), r"processed files *5")
            self.assertRegex(output.getvalue(), r"previously processed files *0")

    def test_gotta_do_audio_first(self):
        with tempfile.TemporaryDirectory(
            prefix="test_gotta_do_audio_first", dir="."
        ) as tmpdir:
            fp_config, lj_filelist, _, _, _ = self.get_simple_config(tmpdir)

            to_process_no_audio = ("energy", "pitch", "attn", "text", "spec")
            with self.assertRaises(SystemExit), capture_stdout():
                Preprocessor(fp_config).preprocess(
                    output_path=lj_filelist, cpus=1, to_process=to_process_no_audio
                )

    def test_empty_preprocess(self):
        # Test case where the file list is not empty but after filtering
        # silence, the result is empty. The behaviour of the code base is not
        # super satisfying, we exit when we try to read
        # preprocessed/filelist.psv and it's not there, rather than catching the
        # fact that we're trying to write an empty list.
        with tempfile.TemporaryDirectory(
            prefix="test_empty_preprocess", dir="."
        ) as tmpdir:
            tmpdir = Path(tmpdir)
            fp_config, lj_filelist, _, _, to_process = self.get_simple_config(tmpdir)
            fp_config.preprocessing.source_data[0].data_dir = self.data_dir
            input_filelist = tmpdir / "empty-metadata.psv"
            with open(input_filelist, mode="w") as f:
                print("basename|raw_text|characters|speaker|language", file=f)
                print("empty|foo bar baz|foo bar baz|noone|und", file=f)
            fp_config.preprocessing.source_data[0].filelist = input_filelist

            with self.assertRaises(SystemExit), capture_stdout():
                Preprocessor(fp_config).preprocess(
                    output_path=lj_filelist, cpus=1, to_process=to_process
                )

    def test_config_lock(self):
        with tempfile.TemporaryDirectory(prefix="test_config_lock", dir=".") as tmpdir:
            tmpdir = Path(tmpdir)
            fp_config, lj_filelist, _, _, to_process = self.get_simple_config(tmpdir)

            Preprocessor(fp_config).preprocess(
                output_path=lj_filelist, cpus=1, to_process=to_process
            )

            def fail_config_lock(config_object, element, value, message):
                import everyvoice.preprocessor.preprocessor

                with monkeypatch(config_object, element, value):
                    with self.assertRaises(SystemExit):
                        with patch_logger(
                            everyvoice.preprocessor.preprocessor
                        ) as logger:
                            with self.assertLogs(logger) as logs:
                                Preprocessor(fp_config).preprocess(
                                    output_path=lj_filelist,
                                    cpus=1,
                                    to_process=to_process,
                                )
                log_output = "\n".join(logs.output)
                self.assertIn("Config lock mismatch:", log_output)
                self.assertIn(message, log_output)

            fail_config_lock(
                fp_config.preprocessing.audio,
                "alignment_sampling_rate",
                42,
                "differs from locked preprocessing.audio config",
            )

            fail_config_lock(
                fp_config.text, "cleaners", [], "differs from locked text config"
            )

            fail_config_lock(
                fp_config.preprocessing.source_data[0],
                "sox_effects",
                [],
                "differs from locked preprocessing.source_data",
            )

            Preprocessor(fp_config).save_config_lock(in_progress=True)
            fail_config_lock(
                fp_config.preprocessing,
                "_na",
                "_na",
                "the previous preprocessing run was interrupted",
            )

            lock_file = tmpdir / "preprocessed" / ".config-lock"
            lock_file.chmod(0o666)
            with open(tmpdir / "preprocessed" / ".config-lock", "w") as f:
                f.write("This is not valid JSON")
            fail_config_lock(
                fp_config.preprocessing,
                "_na",
                "_na",
                "Error loading existing config lock",
            )

    def test_train_split(self):
        """
        PreprocessingConfig's train_split should be [0., 1.].
        """
        config = PreprocessingConfig(train_split=0.5)
        self.assertEqual(config.train_split, 0.5)

        config = PreprocessingConfig(train_split=0.0)
        self.assertEqual(config.train_split, 0.0)

        config = PreprocessingConfig(train_split=1.0)
        self.assertEqual(config.train_split, 1.0)

        with self.assertRaises(ValidationError), capture_stdout() as cout:
            config = PreprocessingConfig(train_split=-0.1)
            self.assertIn("Input should be greater than or equal to 0", cout.getvalue())
        with self.assertRaises(ValidationError), capture_stdout() as cout:
            config = PreprocessingConfig(train_split=1.1)
            self.assertIn("Input should be less than or equal to 1", cout.getvalue())


class PreprocessingHierarchyTest(BasicTestCase):
    def test_hierarchy(self):
        """Unit tests for preprocessing steps"""

        with tempfile.TemporaryDirectory(prefix="test_hierarchy", dir=".") as tmpdir:
            tmpdir = Path(tmpdir)
            data_dir = Path(__file__).parent / "data"
            wavs_dir = data_dir / "hierarchy" / "wavs"
            preprocessed_dir = tmpdir / "hierarchy" / "preprocessed"
            filelist = preprocessed_dir / "preprocessed_filelist.psv"

            fp_config = FeaturePredictionConfig(contact=self.contact)
            fp_config.preprocessing.source_data[0].data_dir = wavs_dir
            fp_config.preprocessing.source_data[0].filelist = (
                data_dir / "hierarchy" / "metadata.psv"
            )
            fp_config.preprocessing.save_dir = preprocessed_dir
            preprocessor = Preprocessor(fp_config)

            with mute_logger("everyvoice.preprocessor"), capture_stdout():
                preprocessor.preprocess(
                    output_path=filelist,
                    cpus=2,
                    overwrite=True,
                    # to_process=("audio", "energy", "pitch", "text", "spec"),
                    # to_process=("audio", "text", "pfs", "spec", "attn", "energy", "pitch"),
                    to_process=("audio", "text", "spec", "attn", "energy", "pitch"),
                )
            for t in ("audio", "spec", "attn", "energy", "pitch"):
                # There are two speakers
                sources = [d.name for d in tmpdir.glob(f"**/{t}/*")]
                self.assertSetEqual(
                    set(sources), set(("LJ010", "LJ050")), f"failed for {t}"
                )
                # First speaker has one recording
                files = (
                    list(tmpdir.glob(f"**/{t}/LJ010/*.wav"))
                    if t == "audio"
                    else list(tmpdir.glob(f"**/{t}/LJ010/*.pt"))
                )
                self.assertEqual(len(files), 1)
                # Second speaker has 5 recordings
                files = (
                    list(tmpdir.glob(f"**/{t}/LJ050/*.wav"))
                    if t == "audio"
                    else list(tmpdir.glob(f"**/{t}/LJ050/*.pt"))
                )
                self.assertEqual(len(files), 5)
