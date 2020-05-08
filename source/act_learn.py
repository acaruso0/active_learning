import os, copy, pickle
import pandas as pd
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from fitting import FittingModel
from loader import Loader
from submit import SubmitFit


class Learner(Loader):
    def __init__(self, settings_file="settings.ini"):
        super().__init__(settings_file)

    def prepare(self):
        self.kernel = C(1.0, (1e-5, 1e5)) * RBF(15, (1e-5, 1e5))
        self.gp = GPR(kernel=self.kernel, n_restarts_optimizer=9, alpha=1e-6)
        self.model = FittingModel(self.main, self.fit_exe, self.eval_exe)

        _, self.coords = self.read_data(self.file_train)
        with open(self.desc_file, 'rb') as pickled:
            self.X_train = pickle.load(pickled)
        self.Y_train = np.zeros(self.X_train.shape[0])

        self.idx_all = np.arange(self.X_train.shape[0])
        self.idx_left = copy.deepcopy(self.idx_all)
        self.idx_now = self.idx_all[~np.in1d(self.idx_all, self.idx_left)]
        # idx_failed = np.ones(idx_all.shape[0], dtype=bool)
        self.err_train = np.zeros_like(self.idx_all, dtype=float)

        os.makedirs(self.output, exist_ok=True)

        self.write_energy_file(self.file_test, self.output + 'val_refer.dat', col_to_write=1)
        self.file_train_tmp = '_trainset_tmp.xyz'

        fitting_code = os.path.join([self.fit_code, self.fit_exe])
        SubmitFit(self.username, self.email, self.fit_fold, fitting_code, self.train_set, self.delta_E)

        # Logfile
        self.logfile = self.output + '_PickSet.log'
        to_log = ['Iteration' 'TrainSet_Size', 'Leftover_Size',
                  'Train_MSE', 'Train_wMSE', 'Test_MSE', 'Test_wMSE',
                  'Fitting[s]',
                  ]
        with open(self.logfile, 'w') as f:
            for key in to_log:
                print(key, end='\t', file=f)
            print('', end='\n', file=f)

        # saving settings
        to_record = {
            'Nr of samples in first iteration': self.first_batch,
            'Nr of samples picked in other iterations': self.batch,
            'cluster size': self.cluster_sz,
            'STD weight': self.STD_w,
            'TRAIN ERROR weight': self.TRAINERR_w,
            'Used Guassian Process model': self.gp,
        }

        with open(self.output + '_setting.ini', 'w') as f:
            for key, value in to_record.items():
                try:
                    print(key + '\t' + str(value), end='\n', file=f)
                except:
                    print(key, end='\n', file=f)
                    print(value, end='\n', file=f)

    def learn(self):
        pass
