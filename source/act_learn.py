import os, copy, pickle
import pandas as pd
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from fitting import FittingModel
from loader import Settings


class Learner(Settings):
    def __init__(self, settings_file="settings.ini"):
        super().__init__(settings_file)

    def write_energy_file(self, infile, outfile='tofitE.dat', picked_idx=[], col_to_write=0):
        max_col_idx = 0
        if type(col_to_write) is int:
            max_col_idx = col_to_write + 1
        elif type(col_to_write) is list:
            try:
                max_col_idx = max(col_to_write)
            except:
                print('Error !!!')
                return False

        _, e = self.read_data(infile, picked_idx=picked_idx, E_columns=max_col_idx)
        np.savetxt(outfile,  e[:, col_to_write], delimiter='    ')
        return True

    def get_weights(self, E,  E_min=None):
        if E_min is None:
            E_min = np.min(E)
        w = np.square(self.delta_E/(E - E_min + self.delta_E))
        w_mean = np.mean(w)
        w /= w_mean

        return w, E_min

    def generate_set(self, infile=None, outfile='set.xyz', picked_idx=[]):
        try:
            data = pd.read_csv(infile, sep='\s+', names=range(4))
            Natom = int(data.iloc[0][0])
            with open(outfile, 'w') as f:
                for i in picked_idx:
                    line_start = i * (Natom + 2)
                    try:
                        print('{0:s} '.format(data.iloc[line_start, 0]), end='\n', file=f)  # number of atoms in one configuration
                        for d in data.iloc[line_start+1, :]:
                            try:
                                if np.isnan(d):
                                    continue
                                else:
                                    print(' {0:s} '.format(str(d)), end='\t', file=f)  # energies
                            except:
                                print(' {0:s} '.format(str(d)), end='\t', file=f)  # energies
                        print(' ', end='\n', file=f)
                        for a in range(Natom):
                            for d in data.iloc[line_start+2+a, :]:
                                print(' {0:s} '.format(str(d)), end='\t', file=f)
                            print('', end='\n', file=f)
                    except:
                        continue
            return True
        except Exception as e:
            print('create new trainset file fails: ' + str(e))
        return None

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
        self.model.init(self.output, self.file_test, self.E_min)
        self.file_train_tmp = '_trainset_tmp.xyz'

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
