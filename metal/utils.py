import numpy as np
from scipy.sparse import issparse, csr_matrix, hstack
import torch
from torch.utils.data import Dataset
import random

class MetalDataset(Dataset):
    """A dataset that group each item in X with it label from Y
    
    Args:
        X: an N-dim iterable of items
        Y: a torch.Tensor of labels
            This may be hard labels [N] or soft labels [N, k]
    """
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
        assert(len(X) == len(Y))

    def __getitem__(self, index):
        return tuple([self.X[index], self.Y[index]])

    def __len__(self):
        return len(self.X)


class Checkpointer(object):
    def __init__(self, model_class, checkpoint_min=0, checkpoint_runway=0,
        verbose=True):
        """Saves checkpoints as applicable based on a reported metric.

        Args:
            checkpoint_min (float): the initial "best" score to beat
            checkpoint_runway (int): don't save any checkpoints for the first
                this many iterations
        """
        self.model_class = model_class
        self.best_model = None
        self.best_iteration = None
        self.best_score = checkpoint_min
        self.checkpoint_runway = checkpoint_runway
        self.verbose = verbose
        if checkpoint_runway and verbose:
            print(f"No checkpoints will be saved in the first "
                f"checkpoint_runway={checkpoint_runway} iterations.")
  
    def checkpoint(self, model, iteration, score):
        if iteration >= self.checkpoint_runway:
            is_best = score > self.best_score
            if is_best:
                if self.verbose:
                    print(f"Saving model at iteration {iteration} with best "
                        f"score {score}")
                self.best_model = model.state_dict()
                self.best_iteration = iteration
                self.best_score = score

    def restore(self, model):
        if self.best_model is None:
            raise Exception(f"Best model was never found. Best score = "
                f"{self.best_score}")
        if self.verbose:
            print(f"Restoring best model from iteration {self.best_iteration} "
                f"with score {self.best_score}")
            model.load_state_dict(self.best_model)
            return model
        

def rargmax(x, eps=1e-8):
    """Argmax with random tie-breaking
    
    Args:
        x: a 1-dim numpy array
    Returns:
        the argmax index
    """
    idxs = np.where(abs(x - np.max(x, axis=0)) < eps)[0]
    return np.random.choice(idxs)

def hard_to_soft(Y_h, k):
    """Converts a 1D tensor of hard labels into a 2D tensor of soft labels

    Args:
        Y_h: an [N], or [N,1] tensor of hard (int) labels between 0 and k 
            (inclusive), where 0 = abstain.
        k: the largest possible label in Y_h
    Returns:
        Y_s: a torch.FloatTensor of shape [N, k + 1] where Y_s[i,j] is the soft
            label for item i and class j.
    """
    Y_h = Y_h.clone()
    Y_h = Y_h.squeeze()
    assert(Y_h.dim() == 1)
    assert((Y_h >= 0).all())
    assert((Y_h <= k).all())
    N = Y_h.shape[0]
    Y_s = torch.zeros((N, k+1))
    for i, j in enumerate(Y_h):
        Y_s[i, j] = 1.0
    return Y_s

def arraylike_to_numpy(array_like):
    """Convert a 1d array-like (e.g,. list, tensor, etc.) to an np.ndarray"""

    orig_type = type(array_like)
    
    # Convert to np.ndarray
    if isinstance(array_like, np.ndarray):
        pass
    elif isinstance(array_like, list):
        array_like = np.array(array_like)
    elif issparse(array_like):
        array_like = array_like.toarray()
    elif isinstance(array_like, torch.Tensor):
        array_like = array_like.numpy()
    elif not isinstance(array_like, np.ndarray):
        array_like = np.array(array_like)
    else:
        msg = (f"Input of type {orig_type} could not be converted to 1d "
            "np.ndarray")
        raise ValueError(msg)
        
    # Correct shape
    if (array_like.ndim > 1) and (1 in array_like.shape):
        array_like = array_like.flatten()
    if array_like.ndim != 1:
        raise ValueError("Input could not be converted to 1d np.array")

    # Convert to ints
    if any(array_like % 1):
        raise ValueError("Input contains at least one non-integer value.")
    array_like = array_like.astype(np.dtype(int))

    return array_like

def convert_labels(Y, source, dest):
    """Convert a matrix from one label type to another
    
    Args:
        X: A np.ndarray or torch.Tensor of labels (ints)
        source: The convention the labels are currently expressed in
        dest: The convention to convert the labels to

    Conventions:
        'categorical': [0: abstain, 1: positive, 2: negative]
        'plusminus': [0: abstain, 1: positive, -1: negative]
        'onezero': [0: negative, 1: positive]

    Note that converting to 'onezero' will combine abstain and negative labels.
    """
    if Y is None: return Y
    Y = Y.copy()
    negative_map = {'categorical': 2, 'plusminus': -1, 'onezero': 0}
    Y[Y == negative_map[source]] = negative_map[dest]
    return Y

def plusminus_to_categorical(Y):
    return convert_labels(Y, 'plusminus', 'categorical')

def categorical_to_plusminus(Y):
    return convert_labels(Y, 'categorical', 'plusminus')

def recursive_merge_dicts(x, y, misses='report', verbose=None):
    """
    Merge dictionary y into a copy of x, overwriting elements of x when there 
    is a conflict, except if the element is a dictionary, in which case recurse.

    misses: what to do if a key in y is not in x
        'insert'    -> set x[key] = value
        'exception' -> raise an exception
        'report'    -> report the name of the missing key
        'ignore'    -> do nothing

    TODO: give example here (pull from tests)
    """
    def recurse(x, y, misses='report', verbose=1):
        found = True
        for k, v in y.items():
            found = False
            if k in x:
                found = True
                if isinstance(x[k], dict):
                    if not isinstance(v, dict):
                        msg = (f"Attempted to overwrite dict {k} with "
                            f"non-dict: {v}")
                        raise ValueError(msg)
                    recurse(x[k], v, misses, verbose)
                else:
                    if x[k] == v:
                        msg = f"Reaffirming {k}={x[k]}"
                    else:
                        msg = f"Overwriting {k}={x[k]} to {k}={v}"
                        x[k] = v
                    if verbose > 1 and k != 'verbose':
                        print(msg)
            else:
                for kx, vx in x.items():
                    if isinstance(vx, dict):
                        found = recurse(vx, {k: v}, 
                            misses='ignore', verbose=verbose)
                    if found:
                        break
            if not found:
                msg = f'Could not find kwarg "{k}" in default config.'
                if misses == 'insert':
                    x[k] = v
                    if verbose > 1: 
                        print(f"Added {k}={v} from second dict to first")
                elif misses == 'exception':
                    raise ValueError(msg)
                elif misses == 'report':
                    print(msg)
                else:
                    pass
        return found
    
    # If verbose is not provided, look for an value in y first, then x
    # (Do this because 'verbose' kwarg is often inside one or both of x and y)
    if verbose is None:
        verbose = y.get('verbose', x.get('verbose', 1))

    z = x.copy()
    recurse(z, y, misses, verbose)
    return z

def make_unipolar_matrix(L):
    """
    Creates a unipolar label matrix from non-unipolar label matrix,
    handles binary and categorical cases
    
    Args:
        csr_matrix L: sparse label matrix
    
    Outputs:
        csr_matrix L_up: equivalent unipolar matrix
    """
    
    # Creating list of columns for matrix
    col_list = []
    
    for col in range(L.shape[1]):
        # Getting unique values in column, ignoring 0
        col_unique_vals = list(set(L[:,col].data)-set([0]))
        if len(col_unique_vals) == 1:
            # If only one unique value in column, keep it
            col_list.append(L[:,col])
        else:
            # Otherwise, make a new column for each value taken by the LF 
            for val in col_unique_vals:
                # Efficiently creating and appending column for each LF value
                val_col = csr_matrix(np.zeros((L.shape[0],1)))
                val_col = val_col + val*(L[:,col] == val)
                col_list.append(val_col)
                
    # Stacking columns and converting to csr_matrix
    L_up = hstack(col_list)
    L_up = csr_matrix(L_up)
    return L_up

def split_data(input_data, splits, input_labels=None, shuffle=True, stratify=None, seed=None):
    """Splits inputs into multiple splits of defined sizes

    Args:
        data: iterable containing data points
        labels: iterable containing labels
        splits: list containing of split sizes (fraction or integer)
        shuffle: if True, shuffle the data before splitting
        stratify: if shuffle=True, uses these labels to stratify the splits
            (separating the data into groups by these labels and sampling from
            those, rather than from the population at large)

    Note: This is very similar to scikit-learn's train_test_split() method,
        but with support for more than two splits.
    """

    # Making copies
    data = input_data.copy()
    labels = input_labels.copy()

    # Setting random seed
    if seed is not None:
        random.seed(seed)

    # Checking size of data
    n_points = len(data)
    if n_points == 0:
        raise ValueError("At least one iterable required as input")
    
    # Ensuring we have either all ints or all floats
    valid_int_split = all(isinstance(x, int) for x in splits)
    valid_float_split = all(isinstance(x, float) for x in splits)
    if not (valid_int_split or valid_float_split):
        raise ValueError(
            'Argument split must contain all decimals or all integers!')

    # If we have a valid split, make sure it's mathematically consistent:
    consistent_int_split = np.sum(splits) == n_points
    consistent_float_split = np.sum(splits) == 1
    if not (consistent_int_split or consistent_float_split):
        raise ValueError(
            'Integer splits must add up to number of data points, fractions to one!')

    # Checking if we are using integer or fractional splits
    int_split = float(splits[0]).is_integer()

    # Converting to integer representation
    if int_split:
        split_nums = splits.copy()
        split_fracs = [float(x)/n_points for x in splits]
    else:
        split_nums = [int(x*n_points) for x in splits]
        split_fracs = splits.copy()
    
    # Adding in 0th element
    split_nums.insert(0,0)
    split_fracs.insert(0,0)

    # Creating index per split vector with 0 in first index
    split_sum = np.cumsum(split_nums).astype(int)
    split_frac_sum = np.cumsum(split_fracs)

    # Shuffling
    if shuffle:
        random.shuffle(data)
        
    # Defining outputs
    data_out = []
    labels_out = []
    split_list = []

    # Handling case without stratification
    if stratify:
        if labels is None:
            raise ValueError("Cannot stratify without labels!")
        if shuffle:
            inds = {}
            unique_vals = np.unique(labels)
            for val in unique_vals:
                inds[val] = np.where(labels == val)[0]
            for ii in range(len(splits)):
                spl_list = []
                for val in unique_vals:
                    start_ind = int(split_frac_sum[ii]*len(inds[val]))
                    end_ind = int(split_frac_sum[ii+1]*len(inds[val]))
                    val_list = [inds[val][x] for x in range(start_ind,end_ind)]
                    spl_list = spl_list + val_list   
                split_list.append(spl_list)
        else:
            raise ValueError(
                "Stratified train/test split is not implemented for "
                "shuffle=False")
    
    else:
        for ii in range(len(splits)):
            split_list.append([x for x in range(split_sum[ii],split_sum[ii+1])])
    
    for spl in split_list:
        data_out.append([data[x] for x in spl])
        if labels is not None:
            labels_out.append([labels[x] for x in spl])        

    return data_out, labels_out, split_list
        
