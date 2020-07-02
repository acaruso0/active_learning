import os
import copy
import pickle
import utils
import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.cluster import KMeans
from fitting import FittingModel
from loader import Loader
from submit import SubmitFit, SubmitMolpro
from time import time
from calc_en import Energy
from generator import Generator


settings = Loader()

kernel = C(1.0, (1e-5, 1e5)) * RBF(15, (1e-5, 1e5))
gp = GPR(kernel=kernel, n_restarts_optimizer=9, alpha=1e-6)
model = FittingModel()

coords, _ = utils.read_data(settings.train_set)
gen = Generator()
X_train = gen.mbtr_calc(coords[0])

new_coords, new_desc = gen.generate(coords)
coords = np.append(coords, new_coords, axis=0)
X_train = np.append(X_train, new_desc, axis=0)
if settings.restart:
    with open(settings.desc_file, 'rb') as pickled:
        coords, X_train = pickle.load(pickled)
Y_train = np.zeros(X_train.shape[0])

idx_all = np.arange(X_train.shape[0])
idx_left = copy.deepcopy(idx_all)
idx_now = idx_all[~np.in1d(idx_all, idx_left)]
# idx_failed = np.ones(idx_all.shape[0], dtype=bool)
err_train = np.zeros_like(idx_all, dtype=float)

os.makedirs(settings.output, exist_ok=True)
os.makedirs(settings.calculations, exist_ok=True)

utils.write_energy_file(settings.test_set, os.path.join(settings.output,
                        'val_refer.dat'), col_to_write=1)

SubmitFit(settings.fit_fold)
SubmitMolpro(settings.calculations)

# Logfile
logfile = os.path.join(settings.output, '_PickSet.log')
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
    'Nr of samples in first iteration': settings.first_batch,
    'Nr of samples picked in other iterations': settings.batch,
    'cluster size': settings.cluster_sz,
    'STD weight': settings.std_w,
    'TRAIN ERROR weight': settings.trainerr_w,
    'Used Guassian Process model': gp,
}

with open(os.path.join(settings.output, '_setting.ini'), 'w') as f:
    for key, value in to_record.items():
        try:
            print(key + '\t' + str(value), end='\n', file=f)
        except:
            print(key, end='\n', file=f)
            print(value, end='\n', file=f)

def label(picks):
    idx_left = idx_left[~np.in1d(idx_left, picks)]
    print("Calculating the energy...")
    calc_energy = Energy(coords, picks, settings.t)
    picks = calc_energy.idx_pick
    Y_train[picks] = calc_energy.energy
    return picks

# while idx_left.shape[0] > 0:
while settings.t < settings.tmax:
    print('Start iteration: ', settings.t)

    tic = time()

    if (idx_now is None) or (idx_now.shape[0] == 0):
        # first iteration: choose sufficient samples for the first run
        idx_pick = np.random.choice(idx_left, first_batch,
                                    replace=False)

    else:
        # other iterations:  Pool selection by probability
        # step 1: clustering current training set
        NCluster = int(idx_now.shape[0] / settings.cluster_sz)
        cls = KMeans(n_clusters=NCluster, init='k-means++',
                     precompute_distances=True, copy_x=True)
        lab_now = cls.fit_predict(X_train[idx_now])
    # predict the label of current candidates
        idx_cand = idx_left
        lab_cand = cls.predict(X_train[idx_cand])

        p_chose_tmp = np.zeros((idx_cand.shape[0],), dtype=float)
    # step 2: predict the uncertainty by GP
        for l in set(lab_cand):
            idx_now_with_this_label = idx_now[lab_now == l]
            idx_cand_with_this_label = idx_cand[lab_cand == l]

            gp.fit(X_train[idx_now_with_this_label, :],
                        Y_train[idx_now_with_this_label])
            prd_cand_with_this_label, uct_cand_with_this_label = gp.predict(X_train[idx_cand_with_this_label], return_std=True)
    # step 3: update selection probability
            p_err = np.average(err_train[idx_now_with_this_label])

            p_chose_tmp[lab_cand==l] = settings.trainerr_w * p_err + settings.std_w * uct_cand_with_this_label
    # step 4: sample from updated probability
        nr_pick = min(p_chose_tmp.shape[0], settings.batch)
        # p_chose_tmp[p_chose_tmp < 1e-4] = 1e-4 # Set the lowest probability
        p_chose_tmp = p_chose_tmp / np.sum(p_chose_tmp)

        idx_pick = np.random.choice(idx_cand, nr_pick, replace=False,
                                    p=p_chose_tmp)

    idx_pick = label(idx_pick)

    #if (self.idx_now is None) or (self.idx_now.shape[0] == 0):
    #    while (len(idx_pick) < self.first_batch):
    #        new_picks = np.random.choice(self.idx_left,
    #                                     self.first_batch - len(idx_pick),
    #                                     replace=False)
    #        self.label(new_picks)
    #        idx_pick = np.append(idx_pick, new_picks)
    #else:
    #    while (len(idx_pick) < nr_pick):
    #        new_picks = np.random.choice(self.idx_left,
    #                                     nr_pick - len(idx_pick),
    #                                     replace=False,
    #                                     p=p_chose_tmp)
    #        self.label(new_picks)
    #        idx_pick = np.append(idx_pick, new_picks)

    print(F'Number of selected configurations in this iteration: {len(idx_pick)}')
    if (idx_now is None) or (idx_now.shape[0] == 0):
        idx_now = idx_pick
    else:
        idx_now = np.hstack((idx_now, idx_pick))

    new_train = os.path.join(settings.calculations, 'it_' + str(settings.t),
                             'tr_set.xyz')
    with open(new_train, 'r+') as labeled_file:
        newxyz = labeled_file.read()
    with open(settings.train_out, 'a+') as oldxyz:
        oldxyz.write(newxyz)

    train_weights = utils.get_weights(Y_train[idx_now],
                                         settings.delta_e, settings.e_min)

    print("Fitting the model...")
    train_err = model.fit(ite=settings.t)
    err_train[idx_now] = np.abs(train_err) * np.sqrt(train_weights)

    if settings.t > 0:
        p_pick = p_chose_tmp[np.in1d(idx_cand, idx_pick)]
        new_coords, new_desc = gen.generate(coords[idx_pick], p_pick)
        idx_left = np.append(idx_left,
                             np.arange(len(X_train),
                                       len(X_train) + len(new_coords)))
        coords = np.append(coords, new_coords, axis=0)
        X_train = np.append(X_train, new_desc, axis=0)
        Y_train = np.append(Y_train, np.zeros(len(new_desc)))
        err_train = np.append(err_train, np.zeros(len(new_desc)))

    print("Creating restart file...")
    train_set_idx = 'trainset_' + str(settings.t) + '.RESTART'
    restart_path = os.path.join(settings.output, train_set_idx)
    restart_file = pd.DataFrame()
    restart_file['idx'] = idx_now
    restart_file['energy'] = Y_train[idx_now]
    restart_file['error'] = err_train[idx_now]
    restart_file.to_csv(restart_path, sep='\t', index=False)
    pool_path = os.path.join(output, "pool")
    with open(pool_path, 'wb') as fp:
        pickle.dump((coords, X_train), fp)

    # section: evaluate current trained model
    test_err, test_weights = model.evaluate(ite=settings.t)

    train_mse = np.sqrt(np.mean(np.square(train_err)))
    train_wmse = np.sqrt(np.mean(np.square(train_err) * train_weights))

    test_mse = np.sqrt(np.mean(np.square(test_err)))
    test_wmse = np.sqrt(np.mean(np.square(test_err) * test_weights))

    toc = time()
    print('time consumed this iteration [s]: ', toc-tic)
    with open(logfile, 'a') as f:
        print('{0:d}\t{1:d}\t{2:d}\t{3:.8f}\t{4:.8f}\t{5:.8f}\t{6:.8f}\t{7:.2f}'.format(
            settings.t, idx_now.shape[0], idx_left.shape[0],
            train_mse, train_wmse, test_mse, test_wmse,
            toc-tic), file=f, end='\n')

    settings.t += 1
