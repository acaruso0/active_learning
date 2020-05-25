import numpy as np
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

    def generate(self, configs, sigma=[1]):  # sigma is a numpy array
        picks_tot = 2*self.first_batch
        picks = (np.array([x/sum(sigma) for x in sigma])*picks_tot).astype(int)
        print(picks)
        new_config, new_desc = [], []
        for n, config in enumerate(configs):
            config = config.flatten()
            cov = np.exp(sigma[n])*np.identity(config.shape[0])
            normal = np.random.multivariate_normal(config, cov, picks[n])
            normal = np.reshape(normal, (-1, len(self.atoms), 3))
            normal_desc = np.array([self.mbtr_calc(coord)[0] for coord in normal])
            if n < 1:
                new_config = normal
                new_desc = normal_desc
            else:
                new_config = np.append(new_config, normal, axis=0)
                new_desc = np.append(new_desc, normal_desc, axis=0)
        return new_config, new_desc
