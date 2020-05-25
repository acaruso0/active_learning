import os
import numpy as np
from loader import Loader
from ase import Atoms
from dscribe.descriptors import MBTR
import subprocess as sp


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

    def distcheck(self, coords):
        file_path = os.path.join(self.main, "temp.nrg")
        json_path = os.path.join(self.main, "mbx.json")
        idx_pick = []
        for idx, coord in coords:
            with open(file_path, 'w+') as outfile:
                outfile.write(F"SYSTEM h2o\nMOLECULE\nMONOMER h2o\n\n")
                for n in range(len(self.atoms[1:])):
                    outfile.write(F"{self.atoms[n]} {coord[n][0]}"
                                  + F"{coord[n][1]} {coord[n][2]}\n")
                outfile.write(F"ENDMON\nENDMOL\nENDSYS")
            single_point = os.path.join(self.mbx, "install", "bin", "main", "single_point")
            command = [single_point, file_path, json_path]
            call = sp.Popen(command, stdout=sp.PIPE)
            energy = float(call.communicate()[0].decode().split()[1])
            if energy < 60:
                idx_pick.append(idx)

        return np.array(idx_pick)

    def generate(self, configs, sigma=[1]):  # sigma is a numpy array
        picks_tot = 10*self.first_batch
        picks = (np.array([x/sum(sigma) for x in sigma])*picks_tot).astype(int)
        new_config, new_desc = [], []
        for n, config in enumerate(configs):
            config = config.flatten()
            cov = 0.3*np.exp(sigma[n]/sum(sigma))*np.identity(config.shape[0])
            normal = np.random.multivariate_normal(config, cov, picks[n])
            normal = np.reshape(normal, (-1, len(self.atoms), 3))

            idx_pick = self.distcheck(normal)
            while len(idx_pick) < picks[n]:
                new_normal = np.random.multivariate_normal(config, cov, picks[n] - len(idx_pick))
                new_normal = np.reshape(new_normal, (-1, len(self.atoms), 3))
                new_idx = idx_pick.shape[0] + self.distcheck(new_normal)
                idx_pick = np.append(idx_pick, new_idx)
                normal = np.append(normal, new_normal)

            normal = normal[idx_pick]
            normal_desc = np.array([self.mbtr_calc(coord)[0] for coord in normal])
            if n < 1:
                new_config = normal
                new_desc = normal_desc
            else:
                new_config = np.append(new_config, normal, axis=0)
                new_desc = np.append(new_desc, normal_desc, axis=0)
        return new_config, new_desc
