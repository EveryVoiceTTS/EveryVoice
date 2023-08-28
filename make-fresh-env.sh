# Create a fresh conda environment for EveryVoice development, following all the
# instructions in readme.md

if [[ "$1" == "-h" || "$1" == "--help" ]]; then
    echo "Usage: bash make-fresh-env.sh [ENV_NAME]"
    exit 0
fi

if (( $# >= 1 )); then
    ENV_NAME="$1"
else
    ENV_NAME=EveryVoice
fi

# Don't overwrite an existing env
if conda env list | grep -q "^$ENV_NAME "; then
    echo "Environment $ENV_NAME already exists, please use a different name."
    exit 1
fi

# This can only be run from the root of an EveryVoice sandbox
if [[ "$0" != make-fresh-env.sh ]]; then
    echo "make-fresh-env.sh only works from the root of an EveryVoice sandbox."
    exit 1
fi

# submodules have to have already been initialized
if git submodule status | grep -q "^-"; then
    echo "Please init the submodules with \"git submodule update --init\"."
    exit 1
fi

echo "Creating EveryVoice conda environment called \"$ENV_NAME\"."
echo -n "Proceed (y/[n])? "
read proceed
if [[ "$proceed" =~ ^[y|Y] ]]; then
    echo Proceeding
else
    echo Quitting
    exit 1
fi

set -o errexit

conda create -y --name "$ENV_NAME" python=3.9
eval "$(conda shell.bash hook)"
conda activate "$ENV_NAME"
pip install -r requirements.torch.txt --extra-index-url https://download.pytorch.org/whl/cu117
pip install cython
pip install -e .
pip install -r requirements.dev.txt
pre-commit install || true
gitlint install-hook || true

echo "EveryVoice environment \"$ENV_NAME\" successfully created."
echo "Run \"conda activate $ENV_NAME\" to activate it."
echo "Run \"cd everyvoice; ./run_tests.py all\" to validate it."
