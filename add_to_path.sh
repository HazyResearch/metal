export METALHOME="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
export PYTHONPATH="$PYTHONPATH:$METALHOME"
echo "Added Snorkel MeTaL repository ($METALHOME) to \$PYTHONPATH."