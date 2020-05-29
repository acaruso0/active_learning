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

    def mass_centr(self, coords):
        mo = 15.999
        mh = 1.00784
        loc = (1/(mo + 2*mh))*(mo*coords[:, 0] + mh*coords[:, 1] + mh*coords[:, 2])
        return loc

    def generate(self, configs, p_list=[1]):
        tot_picks = 20#5*self.first_batch
        configs = configs.reshape((len(configs), -1))
        en_seeds = np.array([config[3:] for config in configs])
        en_seeds = ps.energy(en_seeds.flatten(), configs.shape[0])
        picks = [(p_list[n]*70)/(sum(p_list)*en_seeds[n]) for n in range(configs.shape[0])]
        picks = (np.array(picks)*tot_picks).astype(int)
        for n, config in enumerate(configs):
            cov = 0.1*np.exp(-p_list[n]/sum(p_list))*np.identity(config.shape[0])
            normal = np.random.multivariate_normal(config, cov, picks[n])
            normal = np.reshape(normal, (-1, len(self.atoms), 3))
            energies = ps.energy(normal[:, 1:].flatten(), picks[n])
            energies = np.array(energies)
            normal = normal[energies < 70]
            normal = normal[np.linalg.norm(normal[:, 0] - self.mass_centr(normal[:, 1:]), axis=1) > 1]
            normal = normal[np.linalg.norm(normal[:, 0] - self.mass_centr(normal[:, 1:]), axis=1) < 9]
            normal_desc = np.array([self.mbtr_calc(coord)[0] for coord in normal])
            if n < 1:
                new_config = normal
                new_desc = normal_desc
            else:
                new_config = np.append(new_config, normal, axis=0)
                new_desc = np.append(new_desc, normal_desc, axis=0)
        return new_config, new_desc

