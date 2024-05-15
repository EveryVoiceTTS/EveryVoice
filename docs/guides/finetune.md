# How to fine-tune the existing checkpoints

...more guides coming soon...

### Vocoder matching

Vocoder (i.e. your spec-to-wav model) matching is an important part of the TTS pipeline. Because your spec-to-wav model is trained with the ground-truth Mel spectrograms from your audio, there is a mismatch between the Mel spectrograms created by your text-to-spec model and the ones that the pre-trained vocoders have seen during training. For that reason, it can be helpful to fine-tune your spec-to-wav model with the _generated_ Mel spectrograms from your text-to-spec model.

!!! note
    Vocoder matching will only help with the metallic artefacts that sometimes occur when synthesizing speech. If your model is not intelligible, has other types of errors like mispronunciations - vocoder matching will not solve it. In these cases, the problem is likely with your text-to-spec model, and probably due to either noisy data (noisy recordings, mistranscriptions etc), too little data, or data that is too varied (many different speakers). Please refer to *TODO: troubleshooting* for more information.

<TODO: insert graphic showing text-to-spec and spec-to-wav models>

To finetune your spec-to-wav model with Mel spectrograms from your text-to-spec model (also called 'vocoder matching'), you need to have a pre-trained text-to-spec and spec-to-wav model ready. You also need to have access to some parallel text/audio data (the same or similar data that you used to train your text-to-spec model).

Then you:

1. Generate a folder full of Mel spectrograms from your text-to-spec model:

```bash
everyvoice synthesize from-text <path-to-your-text-to-spec.ckpt> -O spec --filelist <path-to-your-training-filelist.psv> --teacher-forcing-folder <path-to-your-preprocessed-folder>
```

!!! note
    For vocoder matching to work, the size of the generated Mel spectrogram has to be the same as the ground truth Mel spectrogram calculated from the audio, so you have to use 'teacher-forcing' to force the text-to-spec model to output spectrograms of a specific size. To do this, we add the --teacher-forcing-folder and point it to the project `preprocessed` folder with the processed files from our filelist.


2. Move the `synthesized_spec` folder from the generated `synthesis_output` folder to your project `preprocessed` folder.

3. Change the `training.finetune` configuration in your {{ config_filename('spec-to-wav') }} file to True.

4. Set the finetune_ckpt value point to the vocoder that you want to fine-tune.

5. Lower the learning rate (we suggest ****)

6. Train the vocoder again:

```bash
everyvoice train spec-to-wav config/everyvoice-spec-to-wav.yaml
```
