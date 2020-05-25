import numpy as np
import math
import pickle
import utils
from loader import Loader
from ase import Atoms
from dscribe.descriptors import MBTR


class Generator(Loader):
    def __init__(self):
        super().__init__()
        self.mbtr = MBTR(
            species=list(set(self.atoms)),
            k2={
                "geometry": {"function": "inverse_distance"},
                "grid": {"min": 0, "max": 1, "n": 100, "sigma": 0.1},
                "weighting": {"function": "exponential",
                              "scale": 0.5, "cutoff": 1e-3}, },
            k3={
                "geometry": {"function": "cosine"},
                "grid": {"min": -1, "max": 1, "n": 100, "sigma": 0.1},
                "weighting": {"function": "exponential",
                              "scale": 0.5, "cutoff": 1e-3}, },
            periodic=False,
            normalization="l2_each",
        )

    def mbtr_calc(self, coords):
        structure = Atoms(symbols=self.atoms, positions=coords)
        mbtr = self.mbtr.create(structure)
        return mbtr

    def generate(self, configs, picks_tot, sigma=[0.1]):  # sigma is a numpy array
        picks = np.array([x/sum(sigma) for x in sigma])*picks_tot
        new_config, new_desc = [], []
        for n, config in enumerate(configs):
            config = config.flatten()
            cov = (1/sigma[n])*np.identity(config.shape[0])
            normal = np.random.multivariate_normal(config, cov, picks[n])
            normal_desc = np.array([self.mbtr_calc(coord) for coord in normal])
            new_config. append(normal)
