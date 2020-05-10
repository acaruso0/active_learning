import os
import copy
import pickle
import utils
import numpy as np
import pandas as pd
# from sklearn.metrics import mean_squared_error
from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.cluster import KMeans
from fitting import FittingModel
from loader import Loader
from submit import SubmitFit
from time import time
from calc_en import Energy


class Learner(Loader):
    def __init__(self):
        super().__init__()

    def prepare(self):
        self.kernel = C(1.0, (1e-5, 1e5)) * RBF(15, (1e-5, 1e5))
        self.gp = GPR(kernel=self.kernel, n_restarts_optimizer=9, alpha=1e-6)
        self.model = FittingModel(self.main, self.fit_exe, self.eval_exe)

        self.coords, _ = utils.read_data(self.file_train)
        with open(self.desc_file, 'rb') as pickled:
            self.X_train = pickle.load(pickled)
        self.Y_train = np.zeros(self.X_train.shape[0])

        self.idx_all = np.arange(self.X_train.shape[0])
        self.idx_left = copy.deepcopy(self.idx_all)
        self.idx_now = self.idx_all[~np.in1d(self.idx_all, self.idx_left)]
        # idx_failed = np.ones(idx_all.shape[0], dtype=bool)
        self.err_train = np.zeros_like(self.idx_all, dtype=float)

        os.makedirs(self.output, exist_ok=True)

        utils.write_energy_file(self.file_test, self.output + 'val_refer.dat',
                                col_to_write=1)
        self.file_train_tmp = '_trainset_tmp.xyz'

        fitting_code = os.path.join([self.fit_code, self.fit_exe])
        SubmitFit(self.username, self.email, self.fit_fold, fitting_code,
                  self.train_set, self.delta_E)

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
        if self.restart:
            restart = pd.read_csv(self.restart_file, sep='\t')
            self.idx_now = restart['idx'].to_numpy()
            energies = restart['energy'].to_numpy()
            error = restart['error'].to_numpy()
            self.Y_train[self.idx_now] = energies
            self.err_train[self.idx_now] = error
            self.idx_left = self.idx_left[~np.in1d(self.idx_left,
                                                   self.idx_now)]

        # while idx_left.shape[0] > 0:
        while self.t < self.tmax:
            print('Start iteration: ', self.t)

            tic = time()

            if (self.idx_now is None) or (self.idx_now.shape[0] == 0):
                # first iteration: choose sufficient samples for the first run
                idx_pick = np.random.choice(self.idx_left, self.first_batch,
                                            replace=False)

            else:
                # other iterations:  Pool selection by probability
                # step 1: clustering current training set
                NCluster = int(self.idx_now.shape[0] / self.cluster_sz)
                cls = KMeans(n_clusters=NCluster, init='k-means++',
                             precompute_distances=True, copy_x=True)
                lab_now = cls.fit_predict(self.X_train[self.idx_now])
            # predict the label of current candidates
                idx_cand = self.idx_left
                lab_cand = cls.predict(self.X_train[idx_cand])

                p_chose_tmp = np.zeros((idx_cand.shape[0],), dtype=float)
            # step 2: predict the uncertainty by GP
                for l in set(lab_cand):
                    idx_now_with_this_label = self.idx_now[lab_now == l]
                    idx_cand_with_this_label = idx_cand[lab_cand == l]

                    self.gp.fit(self.X_train[idx_now_with_this_label, :],
                                self.Y_train[idx_now_with_this_label])
                    prd_cand_with_this_label, uct_cand_with_this_label = self.gp.predict(self.X_train[idx_cand_with_this_label], return_std=True)
            # step 3: update selection probability
                    p_err = np.average(self.err_train[idx_now_with_this_label])

                    p_chose_tmp[lab_cand==l] = self.TRAINERR_weight * p_err + self.STD_weight * uct_cand_with_this_label
            # step 4: sample from updated probability
                nr_pick = min(p_chose_tmp.shape[0], self.batch)
                # p_chose_tmp[p_chose_tmp < 1e-4] = 1e-4 # Set the lowest probability
                p_chose_tmp = p_chose_tmp / np.sum(p_chose_tmp)

                idx_pick = np.random.choice(idx_cand, nr_pick, replace=False,
                                            p=p_chose_tmp)

            # update energy of those samples newly put into training set
            self.idx_left = self.idx_left[~np.in1d(self.idx_left, idx_pick)]
            print("Calculating the energy...")
            calc_energy = Energy(self.coords, idx_pick)
            #os.chdir(file_folder)
            idx_pick = calc_energy.idx_pick
            print(F'Number of selected configurations in this iteration: {len(idx_pick)}')
            if (self.idx_now is None) or (self.idx_now.shape[0] == 0):
                self.idx_now = idx_pick
            else:
                self.idx_now = np.hstack((self.idx_now, idx_pick))
            self.Y_train[idx_pick] = calc_energy.energy

            new_train = os.path.join(self.calculations, 'it_' + str(self.t),
                                     'tr_set.xyz')
            with open(new_train, 'r+') as labeled_file:
                newxyz = labeled_file.read()
            with open(self.train_out, 'a+') as oldxyz:
                oldxyz.write(newxyz)

            train_weights, _ = utils.get_weights(self.Y_train[self.idx_now])

            print("Fitting the model...")
            # output_folder+file_train_tmp)
            train_err = self.model.fit(ite=self.t)
            # use either weighted training error or non-weighted training error
            self.err_train[self.idx_now] = np.abs(train_err) * np.sqrt(train_weights)
            # section: create new training set and train the model
            print("Creating restart file...")
            file_train_idx = 'trainset_' + str(self.t) + '.RESTART'
            restart_file = pd.DataFrame()
            restart_file['idx'] = self.idx_now
            restart_file['energy'] = self.Y_train[self.idx_now]
            restart_file['error'] = self.err_train[self.idx_now]
            restart_file.to_csv(self.output + file_train_idx, sep='\t',
                                index=False)

            # section: evaluate current trained model
            test_err, test_weights = self.model.evaluate(ite=self.t)

            train_mse = np.sqrt(np.mean(np.square(train_err)))
            train_wmse = np.sqrt(np.mean(np.square(train_err) * train_weights))

            test_mse = np.sqrt(np.mean(np.square(test_err)))
            test_wmse = np.sqrt(np.mean(np.square(test_err) * test_weights))

            toc = time.time()
            print('time consumed this iteration [s]: ', toc-tic)
            with open(self.logfile, 'a') as f:
                print('{0:d}\t{1:d}\t{2:d}\t{3:.8f}\t{4:.8f}\t{5:.8f}\t{6:.8f}\t{7:.2f}'.format(
                    self.t, self.idx_now.shape[0], self.idx_left.shape[0],
                    train_mse, train_wmse, test_mse, test_wmse,
                    toc-tic), file=f, end='\n')

            self.t += 1
