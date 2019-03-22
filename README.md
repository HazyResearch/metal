# Snorkel MeTaL

<img src="assets/logo_01.png" width="150"/>

[![Build Status](https://travis-ci.com/HazyResearch/metal.svg?branch=master)](https://travis-ci.com/HazyResearch/metal)

**_v0.4.1_**

**ANNOUNCEMENT** (3/20): We are thrilled to have achieved a new state-of-the-art score on the [GLUE Benchmark](https://gluebenchmark.com/leaderboard) and four of its component tasks using Snorkel MeTaL. 
Check out the corresponding [blog post](https://dawn.cs.stanford.edu/2019/03/22/glue/) for an overview of how we did it.
The code we used to accomplish this is part of a significant restructuring of multi-task end models in Snorkel MeTaL to make it as easy as possible to perform Massive Multi-Task Learning (MMTL) with supervision at varying levels of granularity and over an arbitrarily large number of tasks.
That code is being prepared for release and will be a part of Snorkel MeTaL v0.5, coming out in April!
Stay tuned on this and other Snorkel developments at our project landing page: [snorkel.stanford.edu](snorkel.stanford.edu).

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

## Blog Posts
* [3/22/2019] [Massive Multi-Task Learning with Snorkel MeTaL: Bringing More Supervision to Bear](https://dawn.cs.stanford.edu/2019/03/22/glue/)
* [2/4/2019] [Emerging Topics in Multi-Task Learning Systems](https://hazyresearch.github.io/snorkel/blog/mtl_systems.html)

## Q&A
If you are looking for help regarding how to use a particular class or method, the best references are (in order):

*  The docstrings for that class
*  The [MeTaL Commandments](https://github.com/HazyResearch/metal/blob/master/docs/metal_commandments.md)
*  The corresponding unit tests in `tests/`
*  The Issues page (We tag issues that might be particularly helpful with the "[reference question](https://github.com/HazyResearch/metal/issues?utf8=%E2%9C%93&q=is%3Aissue+label%3A%22reference+question%22)" label)

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
Y_train_probs = label_model.predict_proba(L_train)

# Train a discriminative end model with the generated labels
end_model = EndModel([1000,10,2])
end_model.train_model(train_data=(X_train, Y_train_probs), valid_data=(X_dev, Y_dev))

# Evaluate performance
score = end_model.score(X_test, Y_test)
```

**_Note for Snorkel users: Snorkel MeTaL, even in the single-task case, learns a slightly different label model than Snorkel does (e.g. here we learn class-conditional accuracies for each LF, etc.)---so expect slightly different (hopefully better!) results._**

## Release Notes
### Major changes in v0.4:
* Upgrade to pytorch v1.0
* Improved control over logging/checkpointing/validation
    * More modular code, separate Logger, Checkpointer, LogWriter classes
    * Support for user-defined metrics for validation/checkpointing
    * Logging frequency can now be based on seconds, examples, batches, or epochs
* Naming convention change: hard (int) labels -> preds, soft (float) labels -> probs


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

Or, to use Snorkel Metal in another project, install it with pip:
```
pip install snorkel-metal
```

## Developer Guidelines
First, read the [MeTaL Commandments](https://github.com/HazyResearch/metal/blob/master/docs/metal_commandments.md), which describe the major design principles, terminology, and style guidelines for Snorkel MeTaL.

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
