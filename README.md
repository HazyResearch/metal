# Snorkel MeTaL (previously known as _MuTS_)

<img src="assets/logo_01.png" width="150"/>

**_v0.1.0-alpha_**
[![Build Status](https://travis-ci.com/HazyResearch/metal.svg?branch=master)](https://travis-ci.com/HazyResearch/metal)

Snorkel MeTaL: A framework for using multi-task weak supervision (MTS), provided by users in the form of _labeling functions_ applied over unlabeled data, to train multi-task models.
Snorkel MeTaL can use the output of labeling functions developed and executed in [Snorkel](snorkel.stanford.edu), or take in arbitrary _label matrices_ representing weak supervision from multiple sources of unknown quality, and then use this to train auto-compiled MTL networks.

**Check out the tutorial on basic usage: https://github.com/HazyResearch/metal/blob/master/Tutorial.ipynb**

Snorkel MeTaL uses a new matrix approximation approach to learn the accuracies of diverse sources with unknown accuracies, arbitrary dependency structures, and structured multi-task outputs.
For more detail, see the **working draft of our technical report on MeTaL: [_Training Complex Models with Multi-Task Weak Supervision_](https://ajratner.github.io/assets/papers/mts-draft.pdf)**

v0.2.0 will be coming soon (mid-August) with additional documentation and tutorials.

## Setup
[1] Install anaconda3 (https://www.anaconda.com/download/#macos)

[2] Create conda environment:
```
conda create -n metal python=3.6
source activate metal
```

[3] Download dependencies:
```
conda install -q jupyter matplotlib networkx nose numpy pandas pytorch scipy torchtext -c pytorch
conda install --channel=conda-forge nb_conda_kernels
```

[4] Set environment:
```
git clone https://github.com/HazyResearch/metal.git
cd metal
source set_env.sh
```

[5] Test functionality:
```
nosetests
```

[6] View (bare bones) tutorial:
[launch jupyter notebook] (selecting your metal conda environment as the kernel)

```
jupyter notebook
```

Navigate to ```Tutorial.ipynb``` and run all.
