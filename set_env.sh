export METALHOME="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
echo "MeTaL home directory: $METALHOME"
export PYTHONPATH="$PYTHONPATH:$METALHOME"
echo "Using PYTHONPATH=${PYTHONPATH}"
export PATH="$PATH:$METALHOME"
echo "Environment variables set!"

# export CUDA_VISIBLE_DEVICES=0
