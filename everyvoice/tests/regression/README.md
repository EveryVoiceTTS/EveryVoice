# EveryVoice regression test suite

## Preparing the regression training data:

- Download LJ 1.1 from https://keithito.com/LJ-Speech-Dataset/ into $HOME/sgile/data/LJSpeech-1.1
- Download Sinhala TTS from https://openslr.org/30/ into $HOME/sgile/data/SinhalaTTS
- Download High quality TTS data for four South African languages (af, st, tn,
  xh) from https://openslr.org/32 into $HOME/sgile/data/OpenSLR32-four-South-Afican-languages
- See [`prep-datasets.sh`](prep-datasets.sh) to see exactly where these datasets
  are expected to be found.
- The suite also trains a StyleTTS2 (text-to-wav) model, which needs pretrained
  F0/ASR/PLBERT/WavLM weights from HuggingFace. `regression-test.sh` fetches
  these itself via `everyvoice fetch-pretrained text-to-wav`, but if your GPU
  nodes don't have internet access, run that command yourself from a node that
  does (once per `config/everyvoice-text-to-wav.yaml`) before submitting jobs,
  so the weights are already cached.
- Run this to create the regression testing directory structure:

```sh
export ACTIVATE_SCRIPT=$HOME/start-ev.sh
export SGILE_DATASET_ROOT=$HOME/sgile/data

mkdir regress-1  # or any suffix you want
cd regress-1
../prep-datasets.sh
```

## Running the regression tests

On a Slurm cluster:

```sh
for dir in regress-*; do
    pushd $dir
    sbatch ../../regression-test.sh
    popd
done
```

Or just use `../../regression-test.sh` directly in the loop if you're not on a cluster.

## One script to run them all

All the above can be accomplished by running `go.sh`.

## Cluster parameters

For NRC clusters, use `sbatch go-<clustername>.sh` to get the appropriate Slurm
parameters. To run this on a different cluster, copy one of the go-*.sh scripts
and customize it with the right partition and account for you.

## Surviving requeue and preemption

`everyvoice train` relies on PyTorch Lightning's built-in Slurm support to save
a checkpoint and requeue itself before the job is killed, whether that's from
hitting the time limit, preemption, or `scancel`. To enable it, add to your
`sbatch` script (or `go-<clustername>.sh`):

```sh
#SBATCH --requeue
#SBATCH --signal=B:USR1@300
```

`B:` sends the signal to the batch shell so it reaches the training process,
and `@300` asks Slurm to send `SIGUSR1` 300 seconds before it sends `SIGTERM`
(and, if `--requeue` is set, calls `scontrol requeue` for you). Make sure the
grace period is long enough for a full checkpoint (model + optimizer state) to
write on your cluster's filesystem — start conservative and tune down once
you've observed the actual write time for your model. Resubmitting from the
same working directory (which `scontrol requeue` does automatically) picks the
checkpoint back up with no extra flags needed.
