# Create a fresh conda environment for EveryVoice development, following all the
# instructions in readme.md

# Edit this line to match your CUDA version or set CUDA_VERSION in the calling
# environment.
CUDA_VERSION=${CUDA_VERSION:=11.7}

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

if which nvidia-smi >& /dev/null && nvidia-smi | grep -q CUDA; then
    if nvidia-smi | grep -q "CUDA Version: $CUDA_VERSION "; then
        : # CUDA version OK
    else
        echo "Mismatched CUDA version. Please set CUDA_VERSION to what is installed on your system."
        echo -n "Found: "
        nvidia-smi | grep CUDA
        echo "Specified: CUDA_VERSION=$CUDA_VERSION"
        exit 1
    fi
else
    echo "Please make sure the CUDA version installed on your system matches CUDA_VERSION=$CUDA_VERSION."
fi
CUDA_TAG=cu$(echo $CUDA_VERSION | sed 's/\.//g')

echo "Creating EveryVoice conda environment called \"$ENV_NAME\" for CUDA $CUDA_VERSION."
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
CUDA_TAG=$CUDA_TAG pip install -r requirements.torch.txt --find-links https://download.pytorch.org/whl/torch_stable.html
pip install cython
pip install -e .
pip install -r requirements.dev.txt
echo "Main environment creation completed with success"

if ! pre-commit install; then
    echo "Error running \"pre-commit install\". Your \"$ENV_NAME\" environment is good, but if you want to submit contributions to the project, please troubleshoot and rerun \"pre-commit install\" in your sandbox."
fi
if ! gitlint install-hook; then
    echo "Error running \"gitlint install-hook\". Your \"$ENV_NAME\" environment is good, but if you want to submit contributions to the project, please troubleshoot and rerun \"gitlint install-hook\" in your sandbox."
fi

echo "EveryVoice environment \"$ENV_NAME\" successfully created."
echo "Run \"conda activate $ENV_NAME\" to activate it."
echo "Run \"cd everyvoice; ./run_tests.py all\" to validate it."
