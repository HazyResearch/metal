echo "USE THIS WITH CAUTION!"
echo "set_env.sh will not be a part of the final repo"

export METALHOME="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
echo "MeTaL home directory: $METALHOME"

export PYTHONPATH="$PYTHONPATH:$METALHOME"
echo "Using PYTHONPATH=${PYTHONPATH}"
export PATH="$PATH:$METALHOME"
echo "Environment variables set!"