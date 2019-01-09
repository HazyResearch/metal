import os
from random import shuffle

import matplotlib.pyplot as plt
import numpy as np


def shuffle_matrices(matrices):
    """Shuffle each member of a list of matrices having the same first dimension
    (along first dimension) according o the same shuffling order."""

    N = matrices[0].shape[0]
    idxs = list(range(N))
    shuffle(idxs)
    out = []
    for M in matrices:
        if M.shape[0] != N:
            raise ValueError("All matrices must have same first dimension.")
        out.append(M[idxs])
    return out


def generate_multi_mode_data(n, mus, props, labels, variances):
    """Generate multi-mode data

    Args:
        - n: [int] Number of data points to generate
        - mus: [list of d-dim np.arrays] centers of the modes
        - props: [list of floats] proportion of data in each mode
        - labels: [list of ints] class label of each mode
        - variances: [list of floats] variance for each mode

    Returns:
        - X: [n x d-dim array] Data points
        - Y: [n-dim array] Data labels
        - C: [n-dim array] Index of the mode each data point belongs to
    """

    assert sum(props) == 1.0
    ns = [int(n * prop) for prop in props]
    d = mus[0].shape[0]
    I_d = np.diag(np.ones(d))

    # Generate data
    Xu = [
        np.random.multivariate_normal(mu, I_d*var, size=ni)
        for mu, ni, var in zip(mus, ns, variances)
    ]
    Yu = [l * np.ones(ni) for ni, l in zip(ns, labels)]
    Cu = [i * np.ones(ni) for i, ni in enumerate(ns)]

    # Generate labels and shuffle
    return shuffle_matrices([np.vstack(Xu), np.hstack(Yu), np.hstack(Cu)])

def create_circular_slice(X, Y, C, h, k, r, slice_label):
    """ Given generated data, creates a slice (represented by 2D circle)
    by assigning all points within circle of specified location/size
    to this slice rom 1 --> 2.

    Args:
        - X: [2 x d-dim array] Data points
        - Y: [2-dim array] Data labels
        - C: [2-dim array] Index of the mode each data point belongs to
        - h: [float] horizontal shift of slice
        - k: [float] vertical shift of slice
        - r: [float] radius of slice
        - slice_label: [int] label to assign slice, in {1, -1}
    """

    circ_idx = np.sqrt((X[:,0] - h)**2 + (X[:,1] - k)**2) < r
    #circ_idx = np.logical_and(circ_idx, C==1)
    C[circ_idx] = 2 
    Y[circ_idx] = slice_label


def overlap_proportion_to_slice_radius(target_op, config, step_size=0.05):
    # naively estimate radius to achieve overlap proportion
    if target_op == 0:
        return 0
    
    import copy
    config_copy = copy.deepcopy(config)
    
    emp_op = -1
    r = 0
    while emp_op < target_op:
        config_copy['head_config']['r'] = r
        X, Y, C, L = generate_synthetic_data(config_copy)
        emp_op = np.sum(C==2) / np.sum(np.logical_or(C==1, C==2))
        r += step_size

    print (f"target op: {target_op}, found op: {emp_op}, found r: {r}")
    return r


def generate_label_matrix(n, accs, covs, Y, C):
    """Generate label matrix. We assume that the last LF is the head LF and the
    one before it is the torso LF it will interact with.

    Args:
        - n: [int] Number of data points
        - accs: [list of floats] accuracies of LFs
        #TODO: covs isn't the overall coverage, but coverage on the associated mode
        - covs: [list of floats] coverage for each LF for its mode
        - Y: [n-dim array] Data labels
        - C: [n-dim array] Index of the mode each data point belongs to

    Returns:
        - L: [n x d-dim array] Data points
    """
    m = np.shape(accs)[0]

    # Construct a label matrix with given accs and covs
    L = np.zeros((n, m))
    for i in range(n):
        j = int(C[i]) # slice
        if np.random.random() < covs[j]:
            if np.random.random() < accs[j]:
                L[i, j] = Y[i]
            else:
                L[i, j] = -Y[i]

    return L

def generate_synthetic_data(config, x_var=None, x_val=None):
    """ Generates synthetic data, overwriting default "x_var" 
    in config with "x_val" if they are specified.
    
    Args:
        config: with default data generation values
        x_var: variable to override, in {"op", "acc", "cov"}
        x_val value to override variable with
    
    Returns:
        X: data points in R^2
        Y: labels in {-1, 1}
        C: slice assignment in {0, 1, 2}
        L: generated label matrix (n x 2)
    """
    assert x_var in ["op", "acc", "cov", None]
    
    X, Y, C = generate_multi_mode_data(
        config['N'], config['mus'], config['props'], 
        config['labels'], config['variances']
    )

    # overwrite data points to create head slice 
    
    # find radius for specified overlap proportion
    slice_radius = overlap_proportion_to_slice_radius(x_val, config) \
        if x_var == 'op' else config['head_config']['r']    
    create_circular_slice(
        X, Y, C, 
        h=config['head_config']['h'], 
        k=config['head_config']['k'], 
        r=slice_radius,
        slice_label=-1
     )
    
    # labeling function generation
    
    accs = config['accs'] 
    if x_var == 'acc':
        accs[-1] = x_val # vary head lf (last index) accuracy
    
    covs = config['covs']
    if x_var == 'cov':
        covs[-1] = x_val # vary head lf (last index) coverage over slice
    
    L = generate_label_matrix(
        config['N'], 
        accs, 
        covs,
        Y, 
        C
    )

    return X, Y, C, L

def plot_slice_scores(
    results, slice_name='S2', xlabel='Overlap Proportion', save_dir=None
):
    baseline_scores = results['baseline']
    manual_scores = results['manual']
    attention_scores = results['attention']
    x_range = baseline_scores.keys()
    
    # take average value across trials
    baseline_collected = [np.mean(np.array([s[slice_name] for s in baseline_scores[x]]))
                          for x in x_range]
    manual_collected = [np.mean(np.array([s[slice_name] for s in manual_scores[x]]))
                          for x in x_range]
    attention_collected = [np.mean(np.array([s[slice_name] for s in attention_scores[x]]))
                          for x in x_range]

    # print x-axis in precision 2
    x_range = ["%.2f" % float(x) for x in x_range]
    
    plt.title(f'Accuracy on {slice_name} vs. {xlabel}')
    plt.plot(x_range, baseline_collected, label='baseline')
    plt.plot(x_range, manual_collected, label='manual')
    plt.plot(x_range, attention_collected, label='attention')
    plt.xlabel(xlabel)
    plt.ylabel(f"Accuracy on {slice_name}")
    plt.ylim(bottom=0, top=1)
    plt.legend()
    plt.show()
    
    if save_dir is not None:
        save_path = os.path.join(save_dir, f"{slice_name}-{xlabel}.png")
        plt.savefig(save_path)
        plt.clf()

