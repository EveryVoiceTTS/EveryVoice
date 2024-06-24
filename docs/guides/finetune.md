# How to fine-tune the existing checkpoints

### Vocoder matching

Vocoder (i.e. your spec-to-wav model) matching is an important part of the TTS pipeline. Because your spec-to-wav model is trained with the ground-truth Mel spectrograms from your audio, there is a mismatch between the Mel spectrograms created by your text-to-spec model and the ones that the pre-trained vocoders have seen during training. For that reason, it can be helpful to fine-tune your spec-to-wav model with the _generated_ Mel spectrograms from your text-to-spec model.

!!! note
    Vocoder matching will only help with the metallic artefacts that sometimes occur when synthesizing speech. If your model is not intelligible, has other types of errors like mispronunciations - vocoder matching will not solve it. In these cases, the problem is likely with your text-to-spec model, and probably due to either noisy data (noisy recordings, mistranscriptions etc), too little data, or data that is too varied (many different speakers). Please refer to *TODO: troubleshooting* for more information.

<TODO: insert graphic showing text-to-spec and spec-to-wav models>

To finetune your spec-to-wav model with Mel spectrograms from your text-to-spec model (also called 'vocoder matching'), you need to have a pre-trained text-to-spec and spec-to-wav model ready. You also need to have access to some parallel text/audio data (the same or similar data that you used to train your text-to-spec model).

Then you:

1. Generate a folder full of Mel spectrograms from your text-to-spec model (repeat this process for both your training and validation filelists):

    Training filelist:

    ```bash
    everyvoice synthesize from-text <path-to-your-text-to-spec.ckpt> -O spec --filelist <path-to-your-training-filelist.psv> --teacher-forcing-directory <path-to-your-preprocessed-directory> --output-dir <path-to-your-preprocessed-directory>
    ```

    Validation filelist:

    ```bash
    everyvoice synthesize from-text <path-to-your-text-to-spec.ckpt> -O spec --filelist <path-to-your-validation-filelist.psv> --teacher-forcing-directory <path-to-your-preprocessed-directory> --output-dir <path-to-your-preprocessed-directory>
    ```

    !!! note
        For vocoder matching to work, the size of the generated Mel spectrogram has to be the same as the ground truth Mel spectrogram calculated from the audio, so you have to use 'teacher-forcing' to force the text-to-spec model to output spectrograms of a specific size. To do this, we add the --teacher-forcing-directory and point it to the project `preprocessed` directory with the processed files from our filelist. This will write a `synthesized_spec` folder within your `preprocessed` directory, that you can use instead of the groundtruth `spec` data by setting `finetune` to True as described in the next step.

2. Set the finetune_ckpt value point to the vocoder checkpoint that you want to fine-tune.

3. Lower the learning rate (we suggest starting at 0.00001)

4. Train the vocoder again with finetuning set to True (train for at least 25000 steps):

    ```bash
    everyvoice train spec-to-wav config/everyvoice-spec-to-wav.yaml -c training.finetune=True
    ```
