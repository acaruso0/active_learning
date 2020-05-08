import os
import configparser


class Loader():
    def __init__(self, settings_file="settings.ini"):
        self.settings = settings_file
        self.load()

    def load(self):
        config = configparser.ConfigParser()
        config.read(self.settings)
        self.main = os.getcwd()
        self.file_train = os.path.join([self.main,
                                        config.get("locations", "train_set")])
        self.file_train = os.path.join([self.main,
                                        config.get("locations", "test_set")])
        self.desc_file = os.path.join([self.main,
                                       config.get("locations", "desc_file")])
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
