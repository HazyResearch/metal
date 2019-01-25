Author: Braden Hancock (github: bhancock8)  
Last Updated: 11/20/18  
MeTaL Version: 0.4.0 

# The MeTaL Commandments
- [The MeTaL Commandments](#the-metal-commandments)
  - [MeTaL Design Principles](#metal-design-principles)
    - [Boundaries](#boundaries)
    - [Single-task vs. Multi-task](#single-task-vs-multi-task)
    - [Contrib](#contrib)
    - [Dependencies](#dependencies)
    - [Tests](#tests)
    - [Config files](#config-files)
    - [Classifiers](#classifiers)
    - [Model Tuning](#model-tuning)
  - [MeTaL Types and Terminology](#metal-types-and-terminology)
  - [MeTaL Style](#metal-style)
    - [Style guide](#style-guide)
    - [Versioning:](#versioning)

---
## MeTaL Design Principles

### Boundaries
**Rule:** MeTaL only does learning and analysis (not preprocessing, featurization, or labeling)

There are too many data types and formats out there to create a one-size-fits-all solution for all machine learning problems. MeTaL is intended to be a set of learning and analysis tools inserted into a pipeline, not the entire pipeline itself. Consequently, preprocessing, featurization, and label matrix generation (i.e., applying labeling sources to the data) are all performed by the user before plugging into MeTaL.

### Single-task vs. Multi-task
**Rule:** All multi-task models, tools, utilities must be in metal/multitask/
MeTaL supports multi-task workflows, but we recognize that the majority of people using it will be in single-task mode. A single-task user should not need to wade through multi-task code, so we keep it in a separate directory that single-task users can be blissfully unaware of. No code in metal/ outside of metal/multitask/ should import from it.

### Contrib
**Rule:** Nothing in the main repo can rely on code in contrib/
metal/contrib/ is a place for putting code that may be helpful for specific use cases of MeTaL. It is made available as a convenience, but comes with no guarantees of correctness/support. Consequently, no core code should depend on it. If you think it needs to, then you should consider if what you’re doing belongs in contrib or if that code in contrib should be elevated to core status (in which case, open an issue to discuss it). If a file in contrib has additional dependencies beyond those required by the MetaL core, those should be stored in a requirements.txt file within that directory of contrib/.

### Dependencies
**Rule:** Keep dependencies light and simple
We want MeTaL to be easily importable to other projects. There’s no quicker way to find yourself in dependency hell than to make your project rely on a bunch of relatively uncommon libraries (which you rely on only for one or two methods anyway) with varying dependencies of their own. If you really need that one function, see if you can find it in a more standard library or implement it simply yourself. (See the [contrib](contrib) section for an additional note on contrib-specific dependencies.)

### Tests
**Rule:** All new code gets a test, and no new code goes in until it passes all tests
The tests/ directory contains unit tests mirroring the structure of the metal/ repo. Whenever a new class, method, or feature is added to the repo, it should come with a corresponding unit tests. Before merging in any PR, it must pass all unit tests. All tests can be run easily by executing the command ‘nosetests’ from the directory home.

### Config files
**Rule:** Prefer config dicts to kwargs, and config dicts contain settings only, no data
In a complex codebase with lots of inheritance, passing kwargs all the way down to the functions where they are used can result in the same kwarg appearing in 5 or 6 different method signatures (possibly with different default values) despite only being used in one. This approach is hard to maintain and error prone. Instead, we define a config dict (configuration dictionary) for each object, pulling values from it where necessary and allowing for easy updates to it. This also makes logging simple, as it writing to file a single python dict stores the complete settings for your model.

Config dicts should contain only settings. All problem-specific data will be passed directly to the individual methods that use them, so we never need to store it, and the model and data remain modular.

Each model (LabelModel and EndModel) has its own config dict with default values for all necessary settings for running a model. If you pass in no additional arguments other than your data, the models will run. If you would like to update the parameters, however, there are three ways to do so: __init__(), train_model(), and update_config(). For the first two methods, all unused kwargs will be converted into a dict to merge into config. The update_config() method accepts a dict directly. Merging is performed recursively, so regardless of how nested a particular setting is inside of the config, you need only specify it by its name, not its full nested path in the dict.

### Classifiers
**Rule:** Use the right Classifier method for predictions/scoring
All models in MeTaL (both LabelModels and EndModels) are descended from the Classifier class, which implements a number of important in-common methods, including all evaluation methods. There are few things quite so pernicious as evaluation bugs; do yourself a favor and use the corrected provided method rather than rounding probabilistic predictions or calculating metrics on your own! We follow the convention of scikit-learn classifiers (for familiarity as well as  cross-compatibility for analysis tools):

predict_proba() - returns soft (probabilistic) predictions
predict() - returns hard (integer) predictions
score() - calculates and scores predictions

Most of the magic happens in predict_proba(). The predict() method calls predict_proba() and then intelligently rounds them to hard predictions. The score() method calls predict and then evaluates the desired metrics (and reduces across tasks in the multi-task setting). Children classes should never overwrite predict() or score().

In the multi-task setting, additional task-specific versions of these are also implemented. If there is a significant efficiency gain to be had by predicting a single task in isolation, a model may also implement the predict_task_proba() method.

```
	(all models) 	        (multi-task only)
	score 		    	score_task
	    |			        |
	predict 	     	predict_task
	    |	          		|
	*predict_proba    <- 	predict_task_proba
```

### Model Tuning
**Rule:** The ModelTuner searches over config dicts, so all config settings are searchable
MeTaL comes with a ModelTuner that converts a user-provided search space into separate complete config dicts, which are then used to instantiate and train a model. Consequently, any setting in the config dict (i.e., all the model settings) can be searched over.


---
## MeTaL Types and Terminology

Basic terms:
items: the individual candidates/examples/elements being classified
hard labels: standard (integer) labels
soft labels: probabilistic (float) labels
arraylike: a list, tuple, 1D np.ndarray, or 1D torch.Tensor

In general, we recommend self-explanatory variable names. There are, however, a number of unique or frequently used constants for a each problem that we give shortened names and use consistently in the code as described below. All terms with a ‘_t’ suffix only apply to the multi-task setting. As is common, lowercase variables refer to scalars and uppercase refer to tensors.

n: (int) the total number of candidates
n: (int) the number of candidates in some local context (e.g., a mini-batch)

m: (int) the total number of labeling sources
m: (int) the number of labeling sources in some local context (e.g., a mini-batch)

t: (int) the number of a tasks

k: (int) the cardinality of the single task
K: (list) the cardinalities of the T tasks
k_t: (int) the cardinality of task t (e.g., for k_t in K_t: …)

L: (scipy.sparse) an [n, m] label matrix
L_t: (scipy.sparse) an [n, m] label matrix for task t

These are the matrices of labels applied by labeling sources to items. MeTaL never handles the user’s labeling sources.

Y: an n-length arraylike of target labels (Y \in [1,k]^n)
Y_t: an n-length arraylike of target labels for task t
Y_p: predicted labels (as opposed to target labels)
Y_s: an [n, k] np.ndarray of soft (float) labels, one per class

These subscripts may be combined as necessary, and should be combined in this order. 
(e.g., Y_tps is soft predicted labels for task t)

X: an n-length iterable of inputs to the EndModel (inputs are often features) 
OR a t-length list of such n-length iterables (if each task requires a different input type)
x: an element of X (e.g., for x in X: …)

A few common featurizers are provided in contrib for convenience, but featurization happens before using MeTaL. Rather than requiring all features to be of a certain type or shape, we only require that if the features are not torch.Tensors, the user provides an input module (inheriting from our base input module) that accepts their feature type as input and outputs a torch.Tensor to the rest of the network.

D: an n-length iterable of items
	d: an element of d (e.g., for d in D: …)

D differs from X in that it may be a more user-friendly representation of your data. For example, in a text task, an x may be the list of encoded indices of the tokens in a sentence, whereas d may be the unencoded sentence as a single string for convenient viewing and debugging. Note that we will never do anything with the elements of D other than print them or run user-defined functions on them.

Notes:
The LabelModel requires only Ls and Ys.
The EndModel requires only Xs and Ys.
The analysis tools may use Xs, Ys, Ls, or Ds.

---
## MeTaL Style

### Style guide
We use the following packages:
* [isort](https://isort.readthedocs.io/en/stable/): import standardization
* [black](https://github.com/ambv/black): automatic code formatting
* [flake8](http://flake8.pycqa.org/en/latest/): PEP8 linting

No commits violating these style protocols will be accepted by the repository. See the developer guidelines on the main repository README for instructions on how to set up an environment for development that will inform you of violations (and autocorrect most of them) before you try to commit.

### Versioning: 
We attempt to follow [semantic versioning](https://semver.org/).

