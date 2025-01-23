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

# User env config -- set ACTIVATE_SCRIPT to point to something that will activate the
# right Python environment, or leave it empty if you don't need it.
ACTIVATE_SCRIPT=$HOME/start_ev.sh

# Run a command, logging it first
r() {
    cmd="$*"
    printf "\n\n======================================================================\n"
    printf 'Running "%s"\n' "$cmd"
    date
    printf "======================================================================\n"
    eval "$cmd" 2>&1
    rc=$?
    if [[ $rc != 0 ]]; then
        printf "\n\nXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX\n"
        echo "Command \"$cmd\" exited with non-zero return code $rc."
        date
        printf "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX\n"
    fi
    return $rc
}

echo "Start at $(date)"
date > START

trap 'echo "Failed or killed at $(date)"; date > FAILED' 0

# Regression config
[[ -e "$ACTIVATE_SCRIPT" ]] && source "$ACTIVATE_SCRIPT"
export TQDM_MININTERVAL=5
MAX_STEPS=1000
# For a production config, use MAX_STEPS=100000 and increase the SBATCH --time above

# Run the new-project wizard
r "coverage run -p -m everyvoice new-project --resume-from wizard-resume"

# Enter the directory created by the wizard
cd regress || { echo "Cannot cd into regress directory, aborting."; date > DONE ; exit 1; }
trap 'echo "Failed or killed at $(date)"; date > ../FAILED' 0

# Preprocess
r "coverage run -p -m everyvoice preprocess config/everyvoice-text-to-spec.yaml"

# Train the fs2 model
r "coverage run -p -m everyvoice train text-to-spec config/everyvoice-text-to-spec.yaml --config-args training.max_steps=$MAX_STEPS"
FS2=logs_and_checkpoints/FeaturePredictionExperiment/base/checkpoints/last.ckpt
ls $FS2

# Train the vocoder
r "coverage run -p -m everyvoice train spec-to-wav config/everyvoice-spec-to-wav.yaml --config-args training.max_steps=$MAX_STEPS"
VOCODER=logs_and_checkpoints/VocoderExperiment/base/checkpoints/last.ckpt
ls $VOCODER

# Synthesize some text
r "coverage run -p -m everyvoice synthesize from-text \
    --output-type wav --output-type spec --output-type textgrid --output-type readalong-xml --output-type readalong-html \
    --filelist ../test.txt \
    --vocoder-path '$VOCODER' \
    '$FS2'"
# TODO: check the synthesized files, somehow

# Exercise two-step synthesis
ONE_WORD=$(cat ../test2.txt)
r "coverage run -p -m everyvoice synthesize from-text --output-type spec --text '$ONE_WORD' '$FS2'"
r "coverage run -p -m everyvoice synthesize from-spec \
    --input synthesis_output/synthesized_spec/'$ONE_WORD'-*.pt \
    --model '$VOCODER'"

# Exercise DeepForceAligner
# Meh, this appears to be broken...
#r "coverage run -p -m dfaligner train config/everyvoice-aligner.yaml --config-args training.max_steps=$MAX_STEPS"
#r "coverage run -p -m dfaligner extract-alignments"


# 7: spin up the demo
# everyvoice demo $FS2 $VOCODER &


# 8: use playwright to synthesize something using the demo
# TODO...

echo "Done at $(date)"
date > ../DONE
trap - 0
