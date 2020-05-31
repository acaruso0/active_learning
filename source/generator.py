import numpy as np
from loader import Loader
from ase import Atoms
from dscribe.descriptors import MBTR
import ps


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

    def generate(self, configs, p_list=[1]):
        tot_picks = 10*self.first_batch
        configs = configs.reshape((len(configs), -1))
        en_seeds = np.array([config[3:] for config in configs])
        en_seeds = ps.energy(en_seeds.flatten(), configs.shape[0])
        picks = [p_list[n]/sum(p_list) for n in range(configs.shape[0])]
        picks = (np.array(picks)*tot_picks).astype(int)
        diag = np.append(np.full(3, 1), np.full(9, .01))*np.identity(configs[0].shape[0])
        for n, config in enumerate(configs):
            cov = np.exp(-p_list[n]/sum(p_list))*diag
            normal = np.random.multivariate_normal(config, cov, picks[n])
            normal = np.reshape(normal, (-1, len(self.atoms), 3))
            energies = ps.energy(normal[:, 1:].flatten(), picks[n])
            energies = np.array(energies)
            normal = normal[energies < 60]
            dists = np.linalg.norm(normal[:, :1] - normal[:, 1:], axis=2)
            normal = normal[np.all((dists > np.full(3,1)) & (dists < np.full(3,9)), axis=1)]
            normal_desc = np.array([self.mbtr_calc(coord)[0] for coord in normal])
            if normal_desc.shape[0] == 0:
                continue
            elif "new_config" not in locals():
                new_config = normal
                new_desc = normal_desc
            else:
                new_config = np.append(new_config, normal, axis=0)
                new_desc = np.append(new_desc, normal_desc, axis=0)
        return new_config, new_desc

