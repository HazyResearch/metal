# Snorkel MeTaL Tutorials

We provide a few tutorials to get you started with Snorkel MeTaL.
We recommend starting with the Basics Tutorial. 
The others can be completed in any order.

### Basics
This tutorial walks through the basic training and evaluation process for the two primary classes of a Snorkel MeTaL pipeline: a label model for combining the votes from multiple weak supervision sources, and a discriminative end model for improved generalization and/or transferability to new form factors.

### Multi-Task
Learn how to use the multi-task versions of our models to utilize supervision sources that (implicitly or explicitly) label multiple tasks at once and capitalize on the benefits of multi-task learning.

### CIFAR 10 Example
See an example of how to easily run the standard machine learning task of classifying the CIFAR 10 image dataset using Snorkel MeTaL; building upon existing datasets like this can be a useful gateway to exploring multi-task learning in a well-known setting.


## Advanced Tutorials

### Class Balance
This tutorial demonstrates estimation of the _class balance_ P(Y) using the `ClassBalanceModel` class.
