# Overview
This directory is the home of tools specific to the MMTL SOTA Quest.

# Setup 

## DFS
If you're working on dfs, run the following command from `metal/mmtl` to ensure that 
we're working in the same environment:
`source activate_mmtl.sh`

 - Sets `METALHOME` directory as environment variable and adds to $PYTHONPATH.
 - Sets `GLUEDATA` directory as environment variable.
 - Activates shared virtual environment on DFS.
 - (To deactivate, type `deactivate`)

## Virtual Environment
To initialize a virtual environment from scratch: 
`pip install -r requirements-mmtl.txt`

Please update `requirements-mmtl.txt` if you add a package.
