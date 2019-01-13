# Snorkel MeTaL

<img src="assets/logo_01.png" width="150"/>

[![Build Status](https://travis-ci.com/HazyResearch/metal.svg?branch=master)](https://travis-ci.com/HazyResearch/metal)

**_v0.3.3_**

## Getting Started
* Quickly [set up](#setup) your environment
* Try out the [tutorials](tutorials/)
* View the [developer guide](#developer-guidelines)

## Motivation
This project builds on [Snorkel](snorkel.stanford.edu) in an attempt to understand how _massively multi-task supervision and learning_ changes the way people program.
_Multitask learning (MTL)_ is an established technique that effectively pools samples by sharing representations across related _tasks_, leading to better performance with less training data (for a great primer of recent advances, see [this survey](https://arxiv.org/abs/1706.05098)).
However, most existing multi-task systems rely on two or three fixed, hand-labeled training sets.
Instead, weak supervision opens the floodgates, allowing users to add arbitrarily many _weakly-supervised_ tasks.
We call this setting _massively multitask learning_, and envision models with tens or hundreds of tasks with supervision of widely varying quality.
Our goal with the Snorkel MeTaL project is to understand this new regime, and the programming model it entails.

More concretely, Snorkel MeTaL is a framework for using multi-task weak supervision (MTS), provided by users in the form of _labeling functions_ applied over unlabeled data, to train multi-task models.
Snorkel MeTaL can use the output of labeling functions developed and executed in [Snorkel](snorkel.stanford.edu), or take in arbitrary _label matrices_ representing weak supervision from multiple sources of unknown quality, and then use this to train auto-compiled MTL networks.

Snorkel MeTaL uses a new matrix approximation approach to learn the accuracies of diverse sources with unknown accuracies, arbitrary dependency structures, and structured multi-task outputs.
This makes it significantly more scalable than our previous approaches.

## References
* **Best Reference: [Training Complex Models with Multi-Task Weak Supervision](https://arxiv.org/abs/1810.02840) [AAAI 2019]**
* [Snorkel MeTaL: Weak Supervision for Multi-Task Learning](https://ajratner.github.io/assets/papers/deem-metal-prototype.pdf) [SIGMOD DEEM 2018]
* _[Snorkel: Rapid Training Data Creation with Weak Supervision](https://arxiv.org/abs/1711.10160) [VLDB 2018]_
* _[Data Programming: Creating Large Training Sets, Quickly](https://arxiv.org/abs/1605.07723) [NeurIPS 2016]_

## Q & A
One of the best places to look for Q&A (besides the doc strings) is the Issues page.
We tag issues that might be aprticularly helpful with the "reference question" label; see them [here](https://github.com/HazyResearch/metal/issues?utf8=%E2%9C%93&q=is%3Aissue+label%3A%22reference+question%22).

## Sample Usage
This sample is for a single-task problem. 
For a multi-task example, see tutorials/Multitask.ipynb.

```
"""
n = # data points
m = # labeling functions
k = cardinality of the classification task

Load for each split: 
L: an [n,m] scipy.sparse label matrix of noisy labels
Y: an n-dim numpy.ndarray of target labels
X: an n-dim iterable (e.g., a list) of end model inputs
"""

from metal.label_model import LabelModel, EndModel

# Train a label model and generate training labels
label_model = LabelModel(k)
label_model.train_model(L_train)
Y_train_pred = label_model.predict(L_train)

# Train a discriminative end model with the generated labels
end_model = EndModel([1000,10,2])
end_model.train_model(train_data=(X_train, Y_train_pred), dev_data=(X_dev, Y_dev))

# Evaluate performance
score = end_model.score(X_test, Y_test)
```

**_Note for Snorkel users: Snorkel MeTaL, even in the single-task case, learns a slightly different label model than Snorkel does (e.g. here we learn class-conditional accuracies for each LF, etc.)---so expect slightly different (hopefully better!) results._**

## Setup
[1] Install anaconda:  
Instructions here: https://www.anaconda.com/download/

[2] Clone the repository:
```
git clone https://github.com/HazyResearch/metal.git
cd metal
```

[3] Create virtual environment:
```
conda env create -f environment.yml
source activate metal
```

[4] Run unit tests:
```
nosetests
```
If the tests run successfully, you should see 50+ dots followed by "OK".  
Check out the [tutorials](tutorials/) to get familiar with the Snorkel MeTaL codebase!

Or, to use Snorkel Metal in another project, install it with pip (conda coming soon):
```
pip install snorkel-metal
```

## Developer Guidelines
First, read the [MeTaL Commandments](https://github.com/HazyResearch/metal/blob/master/docs/metal_design.md), which describe the major design principles, terminology, and style guidelines for Snorkel MeTaL.

If you are interested in contributing to Snorkel MeTaL (and we welcome whole-heartedly contributions via pull requests!), follow the [setup](#setup) guidelines above, then run the following additional command:
```
make dev
```
This will install a few additional tools that help to ensure that any commits or pull requests you submit conform with our established standards. We use the following packages:
* [isort](https://github.com/timothycrosley/isort): import standardization
* [black](https://github.com/ambv/black): automatic code formatting
* [flake8](http://flake8.pycqa.org/en/latest/): PEP8 linting

After running `make dev` to install the necessary tools, you can run `make check` to see if any changes you've made violate the repo standards and `make fix` to fix any related to isort/black. Fixes for flake8 violations will need to be made manually.

### GPU Usage
MeTaL supports GPU usage, but does not include this in automatically-run tests; to run these tests, first install the requirements in `tests/gpu/requirements.txt`, then run:
```
nosetests tests/gpu
```
