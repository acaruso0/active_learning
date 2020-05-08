import pandas as pd
import numpy as np
import subprocess as sp
import configparser
from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C


class Learner():
    def __init__(self, settings_file="settings.ini"):
        self.settings = settings_file
        self.load()
        self.prepare()

    def load(self):
        config = configparser.ConfigParser()
        config.read(self.settings)
        self.delta_E = config.getfloat("fitting", "delta_E")

    def read_data(self, xyz_path, picked_idx=[], energies=True):
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
        # Initialization
        # Gaussian Kernel
        kernel = C(1.0, (1e-5, 1e5)) * RBF(15, (1e-5, 1e5))   # a common Gaussian kernel
        gp = GPR(kernel=kernel, n_restarts_optimizer=9, alpha=1e-6)

        # the fitting code used in AL, including both fitting and evaluating parts
        model = fitting_model(fit_fold, fit_cdl, eval_exe)

        coords = readxyz(file_train_pool, atom_lbl)
        with open(desc_file, 'rb') as pickled:
            X_train = pickle.load(pickled)
        Y_train = np.zeros(X_train.shape[0])  # energy placeholder for the training set

        idx_all = np.arange(X_train.shape[0])     # INDEX of all samples in the pool
        idx_left = copy.deepcopy(idx_all)          # INDEX of samples left in the pool at current iteration
        idx_now = idx_all[~np.in1d(idx_all, idx_left)]  # INDEX of samples in the current training set
        #idx_failed = np.ones(idx_all.shape[0], dtype=bool)
        err_train = np.zeros_like(idx_all, dtype=float)  # placeholder for training error on each sample in the pool

        os.makedirs(output_folder, exist_ok=True)

        write_energy_file(file_test, output_folder + 'val_refer.dat', col_to_write=1) # CHECK COLUMN NUMBER
        model.init(output_folder, file_test, E_min)
        file_train_tmp = '_trainset_tmp.xyz'

        # Logfile
        logfile = output_folder + '_PickSet.log'
        to_log = ['Iteration' 'TrainSet_Size', 'Leftover_Size',
                  'Train_MSE', 'Train_wMSE', 'Test_MSE', 'Test_wMSE',
                  'Fitting[s]',
                 ]
        with open(logfile, 'w') as f:
            for key in to_log:
                print(key, end='\t', file=f)
            print('', end='\n', file=f)

        # saving settings
        to_record = {
            'Nr of samples in first iteration': sample_first_ite,
            'Nr of samples picked in other iterations': sample_chose_per_ite,
            'cluster size': cluster_size,
            'STD weight': STD_weight,
            'TRAIN ERROR weight': TRAINERR_weight,
            'Used Guassian Process model': gp,
        }

        with open(output_folder+'_setting.ini', 'w') as f:
            for key, value in to_record.items():
                try:
                    print(key + '\t' + str(value), end='\n', file = f )
                except:
                    print(key, end='\n', file = f)
                    print(value, end='\n', file = f)
