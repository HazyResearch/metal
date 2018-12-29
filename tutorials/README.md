# Snorkel MeTaL Tutorials

We provide a few tutorials to get you started with Snorkel MetaL.
We recommend starting with the Basics Tutorial. 
The others can be completed in any order.

### Basics
This tutorial walks through the basic training and evaluation process for the two primary classes of a Snorkel MeTaL pipeline: a label model for combining the votes from multiple weak supervision sources, and a discriminative end model for improved generalization and/or transferability to new form factors.

### Multi-Task
Learn how to use the multi-task versions of our models to utilize supervision sources that (implicitly or explicitly) label multiple tasks at once and capitalize on the benefits of multi-task learning.

### Model Tuning
[Coming Soon]  
Try out our built-in model tuners (grid/random search, or Hyperband) to perform hyperparameter search over many potential model configurations.

### Synthetic
[Coming Soon]  
Use our synthetic data generators to create training sets with various properties (e.g., class balance, label density, source accuracies, correlated sources, related tasks, etc.) and answer questions about how you may be able to improve performance on your own applications.


## Advanced Tutorials

### Class Balance
This tutorial demonstrates estimation of the _class balance_ P(Y) using the `ClassBalanceModel` class.