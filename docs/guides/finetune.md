# How to fine-tune the existing checkpoints

There are two ways to train from an existing checkpoint:

- **Start a new experiment from a checkpoint** by setting `training.finetune_checkpoint` in your configuration. `training.finetune_mode` sets how the checkpoint is loaded:
    - `weights-only` (the default): [fine-tuning](#fine-tuning-weights-only).
    - `vocoder-matching` (spec-to-wav models only): [vocoder matching](#vocoder-matching).
    - `full-state`: a [full-state start](#full-state-start-full-state).
- **Continue an interrupted run** in its own experiment folder with [`--auto-resume`](#resuming-an-interrupted-run).

An experiment folder is set by the `training.logger` settings: `<save_dir>/<name>/<version>`, e.g. `logs_and_checkpoints/VocoderExperiment/base`. Its checkpoints are in its `checkpoints` folder.

### Fine-tuning (`weights-only`)

Set `training.finetune_checkpoint` to the checkpoint you want to start from, in a new experiment: an experiment folder different from the one the checkpoint was saved in, e.g. with a different `training.logger.version`. Training loads the checkpoint's weights, and starts the optimizer, learning rate schedule and step counter fresh, from your configuration.

This is the right choice to train on new data, a new speaker or a new language, or with different optimizer settings, e.g. a lower learning rate. The model architecture has to match the checkpoint's.

### Resuming an interrupted run

To continue a run that was interrupted, run the same `everyvoice train` command, with the same configuration, and add `--auto-resume`:

```bash
everyvoice train spec-to-wav config/everyvoice-spec-to-wav.yaml --auto-resume
```

This continues the experiment from its last checkpoint, `last.ckpt` in its `checkpoints` folder (for StyleTTS2, `stage-1-last.ckpt`, `stage-2-last.ckpt` or `finetune-last.ckpt`, depending on `--mode`), with its weights, optimizer, learning rate schedule, step and epoch. It does this even when `training.finetune_checkpoint` is set. If the experiment has no last checkpoint yet, `--auto-resume` starts a new run, so you can always pass it. When Slurm requeues a job, it resumes automatically: see [Requeueing on Slurm](#requeueing-on-slurm).

When resuming, you can change settings like the batch size, the training data, `max_steps` and `max_epochs`. You can't change the optimizer settings, since the checkpoint's optimizer and learning rate schedule are restored as they are, or the model architecture or `training.finetune_mode: vocoder-matching`: training stops with an error that lists the changed settings. To change them, start a new experiment with `training.finetune_checkpoint` set to the last checkpoint instead.

If you start training without `--auto-resume` in an experiment folder that already has a last checkpoint, training stops with an error instead of overwriting it. Pass `--auto-resume` to continue that run, or set a different `training.logger.version` to start a new one.

#### Requeueing on Slurm

On a Slurm cluster, `everyvoice train` can save a checkpoint and requeue itself before the job reaches its time limit. To enable it, add to your `sbatch` script:

```sh
#SBATCH --requeue
#SBATCH --signal=B:USR1@300
```

`--signal=B:USR1@300` asks Slurm to send `SIGUSR1` to the batch shell (`B:`), so that it reaches the training process, 300 seconds before the time limit. Training then saves its last checkpoint and requeues the job, which `--requeue` allows. Make sure the grace period is long enough to write a full checkpoint (model and optimizer state) on your cluster's filesystem: start conservative, and shorten it once you've seen how long your model's checkpoints take to write. You can also trigger this by hand with `scancel --signal=USR1 --batch <jobid>`.

The checkpoint is saved as the experiment's last checkpoint, replacing the previous one only once it is completely written, so an interrupted save never destroys the last good checkpoint. The requeued job resumes from it automatically (Slurm sets `SLURM_RESTART_COUNT`, which turns on `--auto-resume`), and continues its own progress even when `training.finetune_checkpoint` is set.

A new job submitted into the same experiment needs `--auto-resume` to continue from the last checkpoint. Since `--auto-resume` starts a new run when there is no checkpoint yet, your `sbatch` script can always pass it.

### Vocoder matching

Vocoder (i.e. your spec-to-wav model) matching is an important part of the TTS pipeline. Because your spec-to-wav model is trained with the ground-truth Mel spectrograms from your audio, there is a mismatch between the Mel spectrograms created by your text-to-spec model and the ones that the pre-trained vocoders have seen during training. For that reason, it can be helpful to fine-tune your spec-to-wav model with the _generated_ Mel spectrograms from your text-to-spec model.

You can use a pre-trained EveryVoice spec-to-wav model for this process and do not have to train your own spec-to-wav model. You can find EveryVoice pre-trained checkpoints [here](https://drive.google.com/drive/folders/1ya0U4K2d26DoJamg96cEynMJ1w1Tm8nU?usp=sharing)

!!! note
    Vocoder matching will only help with the metallic artefacts that sometimes occur when synthesizing speech. If your model is not intelligible, has other types of errors like mispronunciations - vocoder matching will not solve it. In these cases, the problem is likely with your text-to-spec model, and probably due to either noisy data (noisy recordings, mistranscriptions etc), too little data, or data that is too varied (many different speakers). Please refer to *TODO: troubleshooting* for more information.

<TODO: insert graphic showing text-to-spec and spec-to-wav models>

To finetune your spec-to-wav model (a.k.a 'Vocoder') with Mel spectrograms from your text-to-spec model, you need to have a pre-trained text-to-spec and spec-to-wav model ready. This process is also sometimes called 'vocoder matching'. You also need to have access to some parallel text/audio data (the same or similar data that you used to train your text-to-spec model).

Then you:

1. Generate a folder full of Mel spectrograms from your text-to-spec model (repeat this process for both your training and validation filelists):

    Training filelist:

    ```bash
    everyvoice synthesize from-text <path-to-your-text-to-spec.ckpt> -O spec --filelist <path-to-your-training-filelist.psv> --teacher-forcing-directory <path-to-your-preprocessed-directory> --output-dir <path-to-your-preprocessed-directory> -c text.split_text=false
    ```

    Validation filelist:

    ```bash
    everyvoice synthesize from-text <path-to-your-text-to-spec.ckpt> -O spec --filelist <path-to-your-validation-filelist.psv> --teacher-forcing-directory <path-to-your-preprocessed-directory> --output-dir <path-to-your-preprocessed-directory> -c text.split_text=false
    ```

    !!! note
        For vocoder matching to work, the size of the generated Mel spectrogram has to be the same as the ground truth Mel spectrogram calculated from the audio, so you have to use 'teacher-forcing' to force the text-to-spec model to output spectrograms of a specific size. To do this, we add the --teacher-forcing-directory and point it to the project `preprocessed` directory with the processed files from our filelist. This will write a `synthesized_spec` folder within your `preprocessed` directory, that you can use instead of the groundtruth `spec` data by setting `finetune_mode` to `vocoder-matching` as described in the next step.

2. Set the `finetune_checkpoint` value in the `everyvoice-spec-to-wav.yaml` configuration file to the vocoder checkpoint that you want to fine-tune. This can be either a model you have trained yourself or a pre-trained checkpoint.

3. Lower the learning rate (we suggest starting at 0.00001)

4. Train the vocoder again with `finetune_mode` set to `vocoder-matching` (train for at least 25000 steps, but preferrably 100000 steps). Train it in a new experiment folder by setting a different `training.logger.version`, so the fine-tuned vocoder doesn't overwrite the checkpoints of the vocoder you are fine-tuning:

    ```bash
    everyvoice train spec-to-wav config/everyvoice-spec-to-wav.yaml -c training.finetune_mode=vocoder-matching -c training.logger.version=matched
    ```

    If training is interrupted, run the same command with `--auto-resume` to continue it.

    `vocoder-matching` loads only the weights of `finetune_checkpoint`, so the optimizer, learning rate schedule and step counter start fresh, like in [fine-tuning](#fine-tuning-weights-only).

    !!! note
        Older configurations used `training.finetune: true` for vocoder matching. It still works: it is converted to `training.finetune_mode: vocoder-matching`, with a warning.

### Full-state start (`full-state`)

With `training.finetune_mode: full-state`, a new experiment continues the full training state of `training.finetune_checkpoint`: its weights, optimizer, learning rate schedule, step and epoch. Use it to continue another experiment's run in a new experiment folder, e.g. with new training data.

!!! warning
    `max_steps` and `max_epochs` count from the checkpoint's step and epoch, not from 0: if the checkpoint has already reached them, training stops with an error. The model architecture and the optimizer settings (and, for StyleTTS2, `--mode`) have to match the checkpoint's, as when [resuming](#resuming-an-interrupted-run).
