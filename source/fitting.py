import os
import shutil
import mmap
import re
import utils
import numpy as np
import pandas as pd
import subprocess as sp
from submit import SComputer
from loader import Loader


class FittingModel(Loader):
    def __init__(self):
        super().__init__()
        self.xyz_test, self.e_test = utils.read_data(self.file_test,
                                                     E_columns=4)
        self.weights_test, _ = utils.get_weights(self.e_test[:, 0],
                                                 self.delta_E, self.E_min)
        self.y_test_ref = self.e_test[:, 1]

    # This is the fitting procedure
    # Output is the training err on each sample and the corresponding weight
    def fit(self, ite=None):
        if ite is not None:
            ite = '_' + str(ite)
        else:
            ite = ''

        # Next line: command to run the fitting code
        os.chdir(self.fit_fold)
        if os.path.exists('logs'):
            shutil.rmtree('logs')
        os.mkdir('logs')
        os.chdir('logs')
        submit_fit = os.path.join(self.fit_fold, "submit_fit.sh")
        instance = SComputer(os.getcwd(), self.username)
        instance.run(list(range(self.nfits)), submit_fit)
        instance.check()

        to_sort = {}
        for n in range(self.nfits):
            path_to_out = os.path.join(self.fit_fold, 'logs', 'fit_' + str(n),
                                       'fit.log')
            with open(path_to_out, 'r+') as outfile:
                outfile = mmap.mmap(outfile.fileno(), 0)
            is_there = re.search(b'converged.*\n', outfile)
            if is_there:
                line_wrmsd = re.search(b'err\[wL2\].*\n', outfile)
                wrmsd = float(line_wrmsd.group(0).split()[2])
                to_sort[n] = wrmsd

        fit_cdl_file = self.fit_cdl + '.cdl'
        fit_nc_file = self.fit_fold + self.fit_cdl + '.nc'

        sorted_dct = sorted(to_sort.items(), key=lambda item: item[1])
        best_fit = sorted_dct[0][0]
        shutil.copytree('fit_' + str(best_fit), 'best_fit')
        os.chdir('best_fit')
        if os.path.exists(fit_cdl_file):
            sp.call(['ncgen', '-o', fit_nc_file, fit_cdl_file])

        # transform fitting coefficients
        os.rename(self.fit_fold + 'logs', self.fit_fold + 'logs' + ite)

        # Next line: rename training set error file
        fold_best_fit = os.path.join(self.fit_fold, 'logs' + ite, 'best_fit')
        os.chdir(fold_best_fit)

        corr_file = os.path.join(fold_best_fit, 'correlation.dat')
        train_err = pd.read_csv(corr_file, sep='\s+', header=None)

        err_sq = np.sqrt(train_err.iloc[:, 3])
        return err_sq

    # test/evaluation stage
    # Output is the test error on each sample and the corresponding weight
    def evaluate(self, ite=None):
        if ite is not None:
            ite = '_' + str(ite)
        else:
            ite = ''

        # evaluating
        fit_nc_file = self.fit_fold + self.fit_cdl + '.nc'
        os.chdir(self.fit_fold)
        with open('val_energy' + ite + '.dat', 'w+') as energy_log:
            p = sp.call([self.eval_exe, fit_nc_file, self.test_file],
                        stdout=energy_log)
        print('evaluate status: ', p)

        y_test_pred = pd.read_csv("val_energy" + ite + ".dat",
                                  sep='\s+').iloc[:, 1].values

        err = y_test_pred - self.y_test_ref
        weights = self.weights_test
        return err, weights
