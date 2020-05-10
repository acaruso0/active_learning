import os
import configparser
import numpy as np
from utils import atoms


class Loader():
    def __init__(self, settings_file=os.path.join('..', 'settings.ini')):
        self.settings = settings_file
        self.load()

    def load(self):
        config = configparser.ConfigParser()
        config.read(self.settings)
        self.main = os.path.join(os.getcwd(), '..')
        self.file_train = os.path.join(self.main,
                                       config.get("locations", "train_set"))
        self.file_test = os.path.join(self.main,
                                      config.get("locations", "test_set"))
        self.desc_file = os.path.join(self.main,
                                      config.get("locations", "desc_file"))
        self.fit_fold = os.path.join(self.main, "fitting")
        self.train_out = os.path.join(self.main, "train_out.xyz")
        self.fit_exe = os.path.join(self.fit_fold,
                                    config.get("locations", 'fitting_code'),
                                    config.get("locations", "fit_exe"))
        self.eval_exe = config.get("locations", "eval_exe")
        self.calculations = os.path.join(self.main, "calculations")
        self.output = os.path.join(self.main, "output")

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

        molecules = config.get("system", "molecules")
        self.molecules = np.array(molecules.strip('[]').split()).astype(int)
        self.atoms = atoms(self.file_train, np.sum(self.molecules))
        self.mon1_opt = config.getfloat("system", "mon1_opt")
        self.mon2_opt = config.getfloat("system", "mon2_opt")

        self.molpro = config.get("software", "molpro_path")
