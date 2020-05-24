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
        self.xyz_test, self.e_test = utils.read_data(self.test_set,
                                                     E_columns=4)
        self.weights_test = utils.get_weights(self.e_test[:, 0],
                                              self.delta_e, self.e_min)
        self.y_test_ref = self.e_test[:, 1]

    # This is the fitting procedure
    # Output is the training err on each sample and the corresponding weight
    def fit(self, ite=None):
        if ite is not None:
            ite = '_' + str(ite)
        else:
            ite = ''

        # Next line: command to run the fitting code
        logs_fld = os.path.join(self.fit_fold, "logs")
        if os.path.exists(logs_fld):
            shutil.rmtree(logs_fld)
        os.mkdir(logs_fld)
        submit_fit = os.path.join(self.fit_fold, "submit_fit.sh")
        instance = SComputer(logs_fld)
        instance.run(list(range(self.nfits)), submit_fit)
        instance.check()

        to_sort = {}
        for n in range(self.nfits):
            path_to_out = os.path.join(logs_fld, str(n), 'fit.log')
            with open(path_to_out, 'r+') as outfile:
                outfile = mmap.mmap(outfile.fileno(), 0)
            is_there = re.search(b'converged.*\n', outfile)
            if is_there:
                line_wrmsd = re.search(b'err\[wL2\].*\n', outfile)
                wrmsd = float(line_wrmsd.group(0).split()[2])
                to_sort[n] = wrmsd

        fit_cdl_file = self.fit_exe + '.cdl'
        fit_nc_file = os.path.join(self.fit_fold, self.fit_exe + '.nc')

        sorted_dct = sorted(to_sort.items(), key=lambda item: item[1])
        best_fit = sorted_dct[0][0]
        best_old = os.path.join(logs_fld, str(best_fit))
        best_new = os.path.join(logs_fld, 'best_fit')
        shutil.copytree(best_old, best_new)
        os.chdir(best_new)
        if os.path.exists(fit_cdl_file):
            sp.call(['ncgen', '-o', fit_nc_file, fit_cdl_file])

        # transform fitting coefficients
        os.rename(logs_fld, logs_fld + ite)

        # Next line: rename training set error file
        fold_best_fit = os.path.join(logs_fld + ite, 'best_fit')

        corr_file = os.path.join(fold_best_fit, 'correlation.dat')
        train_err = pd.read_csv(corr_file, sep='\s+', header=None)
        os.chdir(self.main)

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
        fit_nc_file = self.fit_exe + '.nc'
        os.chdir(self.fit_fold)
        with open('val_energy' + ite + '.dat', 'w+') as energy_log:
            p = sp.call([self.eval_path, fit_nc_file, self.test_set],
                        stdout=energy_log)
        print('evaluate status: ', p)

        y_test_pred = pd.read_csv("val_energy" + ite + ".dat",
                                  sep='\s+').iloc[:, 1].values

        err = y_test_pred - self.y_test_ref
        weights = self.weights_test
        os.chdir(self.main)

        return err, weights
