import os
import configparser
import numpy as np
from utils import atoms
from dataclasses import dataclass, field
from typing import Text


@dataclass
class Loader():
    settings: Text = "settings.ini"
    main: Text = field(default_factory=os.getcwd)

    def __post_init__(self):
        i = ['nfits', 't', 'tmax', 'batch', 'first_batch', 'cluster_sz']
        f = ['e_min', 'delta_e', 'std_w', 'trainerr_w', 'mon1_opt', 'mon2_opt']
        b = ['restart']
        config = configparser.ConfigParser()
        config.read(self.settings)
        for section in config.sections():
            for item in config.items(section):
                if item[0] in i:
                    setattr(self, item[0], int(item[1]))
                elif item[0] in f:
                    setattr(self, item[0], float(item[1]))
                elif item[0] in b:
                    setattr(self, item[0], eval(item[1]))
                else:
                    setattr(self, item[0], item[1])

        def join_main(file_dir):
            return os.path.join(self.main, file_dir)

        self.train_set = join_main(self.train_set)
        self.test_set = join_main(self.test_set)
        self.desc_file = join_main(self.desc_file)
        self.fit_fold = join_main("fitting")
        self.train_out = join_main("train_out.xyz")
        self.calculations = join_main("calculations")
        self.output = join_main("output")
        self.fit_exe = os.path.join(self.fit_fold,
                                    self.fitting_code, self.fit_exe)
        self.eval_exe = os.path.join(self.fit_fold,
                                     self.fitting_code, self.eval_exe)
        self.molecules = np.array(self.molecules.strip('[]').split()).astype(int)
        self.atoms = atoms(self.train_set, np.sum(self.molecules))
