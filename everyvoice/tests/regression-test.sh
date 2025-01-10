#!/bin/bash

#SBATCH --job-name=EV-regress
#SBATCH --partition=gpu_a100
#SBATCH --account=nrc_ict__gpu_a100
#SBATCH --qos=low
#SBATCH --time=180
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16000M
#SBATCH --gres=gpu:1
#SBATCH --output=./%x.o%j
#SBATCH --error=./%x.e%j

set -o errexit

# Run a command, logging it first
r() {
    cmd="$*"
    printf "\n\n======================================================================\n"
    printf 'Running "%s"\n' "$cmd"
    printf "\n\n======================================================================\n"
    eval "$cmd" 2>&1
    rc=$?
    if [[ $rc != 0 ]]; then
        echo "Command \"$cmd\" exited with non-zero return code $rc"
    fi
    return $rc
}

# User env config -- adjust this as necessary before running:
ACTIVATE_SCRIPT=$HOME/start_ev.sh
LJ_SPEECH_DATASET=$HOME/tts/corpora/Speech/LJ.Speech.Dataset/LJSpeech-1.1

# Regression config
[[ -e "$ACTIVATE_SCRIPT" ]] && source "$ACTIVATE_SCRIPT"
export TQDM_MININTERVAL=5
LINES_TO_USE=2000
EVERYVOICE_ROOT=$(python -c 'import everyvoice; print(everyvoice.__path__[0])')
if [[ $SLURM_JOBID ]]; then
    WORKDIR_SUFFIX="$SLURM_JOBID"
else
    WORKDIR_SUFFIX="$(date +'%Y%m%d')"
fi
WORKDIR=regress-"$WORKDIR_SUFFIX"
mkdir "$WORKDIR"
cd "$WORKDIR"

# 1: create a small dataset with *no header line*
head -$(("$LINES_TO_USE" + 1)) "$LJ_SPEECH_DATASET/metadata.csv" | tail -"$LINES_TO_USE" > metadata.csv
ln -s "$LJ_SPEECH_DATASET/wavs" .

# 2: run the new-project wizard
r "coverage run -p -m everyvoice new-project --resume-from '$EVERYVOICE_ROOT/tests/regress-lj2k-resume'"

# 3: preprocess

cd regress
r "coverage run -p -m everyvoice preprocess config/everyvoice-text-to-spec.yaml"

# 4: train the fs2 model
r "coverage run -p -m everyvoice train text-to-spec config/everyvoice-text-to-spec.yaml --config-args training.max_epochs=2"
FS2=logs_and_checkpoints/FeaturePredictionExperiment/base/checkpoints/last.ckpt
ls $FS2

# 5: train the vocoder
r "coverage run -p -m everyvoice train spec-to-wav config/everyvoice-spec-to-wav.yaml --config-args training.max_epochs=2"
VOCODER=logs_and_checkpoints/VocoderExperiment/base/checkpoints/last.ckpt
ls $VOCODER

# 6: synthesize some text
r "coverage run -p -m everyvoice synthesize from-text \
    --output-type wav --output-type spec --output-type textgrid --output-type readalong-xml --output-type readalong-html \
    --filelist '$EVERYVOICE_ROOT/tests/regress-text.txt' \
    --vocoder-path '$VOCODER' \
    '$FS2'"
# TODO: check the synthesized files, somehow

# 7: spin up the demo
# everyvoice demo $FS2 $VOCODER &


# 8: use playwright to synthesize something using the demo
# TODO...


# Run the regular everyvoice test suite to complete coverage
cd ..
r "coverage run -p -m everyvoice test"

# Collect coverage data
coverage combine . regress
coverage html --include='*/everyvoice/*'
coverage report --include='*/everyvoice/*'
