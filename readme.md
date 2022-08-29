# Small Team Speech Model

This is an implementation of the Text-to-Speech (TTS) model used by the Small Teams "Speech Generation for Indigenous Language Education" project.

It is largely based off the FastSpeech2/Fastpitch models.

## Quickstart

### Dependencies

I recommend using Conda and Python 3.7. To do that, create a new environment:

```
conda create --name SmallTeamsSpeech python=3.7
conda activate SmallTeamsSpeech
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
```

You can then install the Python dependencies with pip or conda
```
pip3 install -r requirements.txt
```

