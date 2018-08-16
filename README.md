# Snorkel MeTaL (previously known as _MuTS_)

<img src="assets/logo_01.png" width="150"/>

[![Build Status](https://travis-ci.com/HazyResearch/metal.svg?branch=master)](https://travis-ci.com/HazyResearch/metal)

**_v0.1.0-alpha_** 

**NOTE:**
_[8/5/18] Expect frequent changes, not all backwards-compatible, through mid-August when v0.1.0 is solidified and bundled as a pip/conda package. It will be released with additional documentation and tutorials._

**This project builds on [Snorkel](snorkel.stanford.edu) in an attempt to understand how _massively multi-task supervision and learning_ changes the way people program.
_Multitask learning (MTL)_ is an established technique that effectively pools samples by sharing representations across related _tasks_, leading to better performance with less training data (for a great primer of recent advances, see [this survey](https://arxiv.org/abs/1706.05098)).
However, most existing multi-task systems rely on two or three fixed, hand-labeled training sets.
Instead, weak supervision opens the floodgates, allowing users to add arbitrarily many _weakly-supervised_ tasks.
We call this setting _massively multitask learning_, and envision models with tens or hundreds of tasks with supervision of widely varying quality.
Our goal with the Snorkel MeTaL project is to understand this new regime, and the new way of interacting with models it entails.**

More concretely, Snorkel MeTaL is a framework for using multi-task weak supervision (MTS), provided by users in the form of _labeling functions_ applied over unlabeled data, to train multi-task models.
Snorkel MeTaL can use the output of labeling functions developed and executed in [Snorkel](snorkel.stanford.edu), or take in arbitrary _label matrices_ representing weak supervision from multiple sources of unknown quality, and then use this to train auto-compiled MTL networks.

**Check out the basics tutorial: https://github.com/HazyResearch/metal/blob/master/tutorials/Basics.ipynb**

Snorkel MeTaL uses a new matrix approximation approach to learn the accuracies of diverse sources with unknown accuracies, arbitrary dependency structures, and structured multi-task outputs.
This makes it significantly more scalable than our previous approaches.
For more detail, see the **working draft of our technical report on MeTaL: [_Training Complex Models with Multi-Task Weak Supervision_](https://ajratner.github.io/assets/papers/mts-draft.pdf)**

## Sample Usage
This sample is for a single-task problem. 
For a multi-task example, see tutorials/Multi-task.ipynb.

```
"""
Load for each split: 
L: an [n,m] scipy.sparse label matrix of noisy labels
Y: an n-dim numpy.ndarray of target labels
X: an n-dim iterable (e.g., a list) of end model inputs
"""

from metal.label_model import LabelModel, EndModel

# Train a label model and generate training labels
label_model = LabelModel(k)
label_model.train(L_train)
Y_train_pred = label_model.predict(L_train)

# Train a discriminative end model with the generated labels
end_model = EndModel(k, layer_out_dims=[1000,10])
end_model.train(X_train, Y_train_pred, X_dev, Y_dev)

# Evaluate performance
score = end_model.score(X_test, Y_test)
```

## Setup
[1] Install anaconda3 (https://www.anaconda.com/download/#macos)

[2] Clone repository:
```
git clone https://github.com/HazyResearch/metal.git
cd metal
```

[3] Create environment:
```
source set_env.sh
conda env create -f environment.yml
source activate metal
```

[4] Test functionality:
```
nosetests
```

[5] Run the Basics Tutorial:
```
jupyter notebook 
```
Open ```tutorials/Basics.ipynb```  
Select Kernel > Change kernel > Python [conda env:metal]  
Restart and run all
