export METALHOME="$( cd "$( dirname "../../../${BASH_SOURCE[0]}" )" && pwd )"
export PYTHONPATH="$PYTHONPATH:$METALHOME"
echo "Added Snorkel MeTaL repository ($METALHOME) to \$PYTHONPATH."

export GLUEDATA="/dfs/scratch0/bradenjh/glue"
echo "Added data path ($GLUEDATA) to environment variables."

VENVPATH="/dfs/scratch0/vschen/venv-mmtl"
source "$VENVPATH/bin/activate"
echo "Activated virtual environment ($VENVPATH)."
