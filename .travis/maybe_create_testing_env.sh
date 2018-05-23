#!/usr/bin/env bash

test -e ~/miniconda/envs/test/bin/activate || ( rm -rf ~/miniconda/envs/test; conda create --yes -n test python=$TRAVIS_PYTHON_VERSION )

# Create conda environment "testing" if it doesn't exist
if test -e $HOME/miniconda/envs/testing/bin/activate ; then
    echo 'Conda environment "testing" already exists.'
else
    echo 'Creating conda environment "testing".'
    rm -rf ~/miniconda/envs/testing; 
    conda create -n testing python=$TRAVIS_PYTHON_VERSION
fi