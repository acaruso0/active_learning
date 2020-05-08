import os
import pandas as pd
import numpy as np
import subprocess as sp
import configparser


class Loader():
    def __init__(self, settings_file="settings.ini"):
        self.settings = settings_file
        self.load()

    def load(self):
        config = configparser.ConfigParser()
        config.read(self.settings)
        self.main = os.getcwd()
        self.file_train = os.path.join([self.main, config.get("locations", "train_set")])
        self.file_train = os.path.join([self.main, config.get("locations", "test_set")])
        self.desc_file = os.path.join([self.main, config.get("locations", "desc_file")])
        self.fit_fold = [self.main, 'fitting']
        self.fit_code = config.get("locations", 'fitting_code')
        self.fit_exe = config.get("locations", "fit_exe")
        self.eval_exe = config.get("locations", "eval_exe")
        self.calculations = os.path.join([self.main, "calculations"])
        self.output = os.path.join([self.main, "output"])

        self.username = config.get("account", "username")
        self.email = config.get("account", "email")

        self.E_min = config.getfloat("fitting", "E_min")
        self.delta_E = config.getfloat("fitting", "delta_E")
        self.nfits = config.getint("fitting", "nfits")

        self.t = config.getint("iterations", "t")
        self.tmax = config.getint("iterations", "tmax")
        self.batch = config.getint("iterations", "batch")
        self.first_batch = config.getint("iterations", "first_batch")
        self.restart = config.getboolean("iterations", "restart")
        if self.restart:
            self.restart_file = config.get("iterations", "restart_file")

        self.cluster_sz = config.getint("active learning", "cluster_sz")
        self.STD_w = config.getfloat("active learning", "STD_w")
        self.TRAINERR_w = config.getfloat("active learning", "TRAINERR_w")

    def read_data(self, xyz_path, picked_idx=[], energies=False):
        command = F'head -n 2 {xyz_path} | tail -n 1'
        call = sp.Popen(command, shell=True, stdout=sp.PIPE)
        E_columns = len(call.communicate()[0].split())

        coords = pd.read_csv(xyz_path, sep='\s+', names=range(max(E_columns, 4)))
        natoms = int(coords.iloc[0][0])
        xyz = coords.iloc[coords.index % (natoms + 2) > 1, :].copy()

        at_list = xyz.loc[:natoms + 1, 0:0].values
        at_list = at_list.reshape(natoms)
        xyz = xyz.loc[:, 1:4].values
        xyz = xyz.reshape((-1, natoms, 3))

        if (len(picked_idx) > 0):
            xyz = xyz[picked_idx, :]

        if energies:
            E = coords.iloc[coords.index % (natoms+2) == 1, :].copy()
            E = E.iloc[:, :E_columns].values
            E = E.astype(np.float)
            if (len(picked_idx) > 0):
                E = E[picked_idx, :]
            return at_list, xyz, E
        else:
            return at_list, xyz

    def get_weights(self, E,  E_min=None):
        if E_min is None:
            E_min = np.min(E)
        w = np.square(self.delta_E/(E - E_min + self.delta_E))
        w_mean = np.mean(w)
        w /= w_mean

        return w, E_min
