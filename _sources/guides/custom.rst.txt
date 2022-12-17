.. _custom:

Customize to your language
==========================

Step 1: Make sure you have Permission!
--------------------------------------

So, you want to build a text-to-speech system for a new language or dataset - cool! But, just because you **can** build a text-to-speech system, doesn't mean you **should**. There are a lot of tricky ethical
questions around text-to-speech. It's not ethical to just use audio you find somewhere if it doesn't have explicit permission to use it for the purposes of text-to-speech. The first step is to make sure you have
permission to use the data in question and that whoever contributed their voice to the data you want to use is aware and supportive of your goal.

Step 2: Gather Your Data
------------------------

The first thing to do is to get all the data you have (in this case audio with text transcripts) together in one place. Your audio should be in 'wav' format. Ideally it would be 16bit, mono (one channel) audio sampled somewhere between 22.05kHz and 48kHz. If that doesn't mean anything to you, don't worry, we can ensure the right format in later steps.
It's best if your audio clips are somewhere between half a second and 10 seconds long. Any longer and it could be difficult to train. If your audio is longer than this, we suggest processing it into smaller chunks first.

Your text should be consistently written and should be in a pipe-separated values spreadsheet, similar to this file: https://github.com/roedoejet/SmallTeamSpeech/blob/main/smts/filelists/lj_full.psv
It should have a column that contains text and a column that contains the "basename" of your associated audio file. So if you have a recording of somebody saying "hello how are you?" and the corresponding audio is called mydata0001.wav
then you should have a psv file that looks like this:


.. code-block:: csv
    :emphasize-lines: 2

    basename|text
    mydata0001|hello how are you?
    mydata0002|some other sentence.
    ...

We also support comma and tab separated files, but find that using pipes (|) causes the least problems.

You can also use the "festival" format which is like this (example from `Sinhala TTS <https://openslr.org/30/>`_):

.. code-block:: text

    ( sin_2241_0329430812 " කෝකටත් මං වෙනදා තරම් කාලෙ ගන්නැතිව ඇඳ ගත්තා " )
    ( sin_2241_0598895166 " ඇන්ජලීනා ජොලී කියන්නේ පසුගිය දිනවල බොහෝ සෙයින් කතා බහට ලක්වූ චරිතයක් " )
    ( sin_2241_0701577369 " ආර්ථික චින්තනය හා සාමාජීය දියුණුව ඇති කළ හැකිවනුයේ පුද්ගල ආර්ථික දියුණුව සලසා දීමෙන්ය " )
    ( sin_2241_0715400935 " ඉන් අදහස් වන්නේ විචාරාත්මක විනිවිද දැකීමෙන් තොර බැල්මයි " )
    ( sin_2241_0817100025 " අප යුද්ධයේ පළමු පියවරේදීම පරාද වී අවසානය " )

In this format, there are corresponding wav files labelled sin_2241_0329430812.wav etc..

Step 3: Install |project|
-------------------------

Head over to the :ref:`install` documentation and install |project|


Step 4: Run the Configuration Wizard 🧙
---------------------------------------

Once you have your data, the best thing to do is to run the Configuration Wizard 🧙. To do that run:

.. code-block:: bash

    smts config-wizard

After running the config-wizard, cd into your newly created directory. Let's call it `test` for now.

.. code-block:: bash

    cd test

Step 5: Run the Preprocessor
---------------------------

Your models need to do a number of preprocessing steps in order to prepare for training. To preprocess everything you need, run the following:

.. code-block:: bash

    smts dfa preprocess -p config/aligner.yaml


Step 6: Train Aligner
----------------------

.. code-block:: bash

    smts dfa train -p config/aligner.yaml

By default, we run our training with PyTorch Lightning\'s "auto" strategy. But, if you are on a machine where you know the hardware, you can specify it like:

.. code-block:: bash

    smts dfa train -p config/aligner.yaml -d 1 -a gpu

Which would use the GPU accelerator and specify 1 device/chip.

Step 6: Export Alignments
-------------------------

.. code-block:: bash

    smts dfa extract-alignments -p config/aligner.yaml -m logs/Aligner\ Experiment/base/checkpoints/last.ckpt

This will extract the alignments from your best model and put them in your preprocessing folder.


Step 7: Preprocess the pitch and energy data for your Feature Prediction Network
---------------------------------------------------------------------------------

For the phone-averaged pre-processing to work, you must have the durations from your aligner which is why we preprocess the data for the feature prediction network after training the aligner.

.. code-block:: bash

    smts fs2 preprocess -p config/feature_prediction.yaml -d pitch -d energy -O


Step 8: Train your Vocoder
--------------------------

.. code-block:: bash

    smts hifigan train -p config/vocoder.yaml


Step 9: Train your Feature Prediction Network
---------------------------------------------------------------

To generate audio when you train your feature prediction network, you need to add your vocoder checkpoint to the config/feature_prediction.yaml

At the bottom of that file you'll find a key called vocoder_path. Add the absolute path to your trained vocder (here it would be /path/to/test/logs/Vocoder Experiment/base/checkpoints/last.ckpt where /path/to would be the actual path to it on your computer.)

Once you've replaced the vocoder_path key, you can train your feature prediction network:

.. code-block:: bash

    smts fs2 train -p config/feature_prediction.yaml


Step 10: Synthesize Speech in Your Language!
---------------------------------------------

You can synthesize by pointing the CLI to your trained feature prediction network and passing in the text. You can export to wav, npy, or pt files.

.. code-block:: bash

    smts fs2 synthesize logs/Feature\ Prediction\ Experiment/base/checkpoints/last.ckpt -t "මෙදා සැරේ සාකච්ඡාවක් විදියට නෙවෙයි නේද පල කරල තියෙන්නෙ" -a gpu -d 1 -O wav



.. Step 10 (optional): Finetune your vocoder
.. ----------------------------------------

.. .. code-block:: bash

..     smts e2e train -p config/e2e.yaml


.. Step 11: Synthesize Speech
.. --------------------------

.. .. code-block:: bash

..     smts e2e synthesize -t "hello world" -c config/e2e.yaml

.. .. warning::

..     TODO: this doesn't exist yet
..     TODO: e2e needs checkpoint paths
