import numpy as np

from tqdm import tqdm

from .utils import new_sample

def metropolis_hastings(target_density, x0, bounds, size=500000):
    xt = x0
    samples = []
    for i in tqdm(range(size)):
        xt_candidate = new_sample(bounds)
        try:
            accept_prob = (target_density(xt_candidate))/(target_density(xt))
        except ValueError as e:
            print(bounds)
            print(xt, xt_candidate)
            raise e
        if np.random.uniform(0, 1) < accept_prob:
            xt = xt_candidate
        samples.append(xt)
    samples = np.array(samples)
    samples = np.reshape(samples, [samples.shape[0], xt.shape[0]])
    return samples
