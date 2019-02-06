# Overview
This directory is the home of tools specific to the MMTL SOTA Quest.

* Make all paths relative to $METALHOME (likely using add_to_path.sh)
* Store data in $METALHOME/data/
* As much as possible, subclass in this dir rather than rewriting outside this dir

# Setup 

## DFS
If you're working on dfs, use the following command ensure that we're working on the same environment:
`source activate_mmtl.sh`

 - Sets `METALHOME` directory as environment variable and adds to $PYTHONPATH.
 - Sets `GLUEDATA` directory as environment variable.
 - Activates shared virtual environment on DFS.

## Virtuale Environment
To initialize a virtual environment from scratch: 
`pip install -r requirements-mmtl.txt`

Please update `requirements-mmtl.txt` if you add a package.
