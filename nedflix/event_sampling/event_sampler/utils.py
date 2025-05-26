import numpy as np

def new_sample(bounds):
    return np.array([np.random.uniform(lb, ub) for lb, ub in bounds])
