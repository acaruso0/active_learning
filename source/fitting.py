import os, shutil, io, mmap, re
import numpy as np
import pandas as pd
import subprocess as sp
from time import sleep


class FittingModel:
    def __init__(self, fit_fold=None, fit_cdl=None, eval_exe=None):
        self.fit_fold = fit_fold
        self.fit_cdl = fit_cdl
        self.eval_exe = eval_exe

    def init(self, folder=None, test_file=None, E_min=None):
        self.running_folder = folder

        self.test_file = test_file
        self.E_min = E_min
        self.xyz_test, self.e_test = read_data(test_file, E_columns=4)
        self.weights_test, _ = get_weights(self.e_test[:, 0], E_min=E_min)
        self.y_test_ref = self.e_test[:, 1]

    # This is the fitting procedure including saving a copy of cdl file to nc file
    # Output is the training err on each sample and the corresponding weight

    def run(self, nfits, subscript):
        jobs = []
        path_script = os.path.join(self.path, subscript)

        for n in range(nfits):
            os.mkdir('fit_' + str(n))
            os.chdir('fit_' + str(n))
            # trainset = os.path.join(self.fit_fold, file_lbl)
            job = sp.Popen(["sbatch", submit_fit], stdout=sp.PIPE)
            check = job.communicate()[0]

            while b'Submitted batch job' not in check:
                sleep(60)
                job = sp.Popen(["sbatch", path_script], stdout=sp.PIPE)
                check = job.communicate()[0]

            job_idx = int(check.decode('utf-8').split()[3])
            jobs.append(job_idx)

            os.chdir('..')
        os.chdir(self.path)

        return np.array(jobs)

    def check(self):
        check_str = ['squeue', '-u', self.username]
        check = sp.Popen(check_str, stdout=sp.PIPE)
        out_str = io.StringIO(check.communicate()[0].decode('utf-8'))
        df = pd.read_csv(out_str, sep='\s+')

        return df['JOBID'].to_numpy().astype(int)

    def fit(self, ite=None, file_lbl=None):
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
        jobs = self.run(self.pick, submit_fit)

        check = self.check()
        while np.isin(check, jobs).any():
            sleep(60)
            check = self.check()

        to_sort = {}
        for n in range(nfits):
            path_to_out = os.path.join(self.fit_fold, 'logs', 'fit_' + str(n), 'fit.log')
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

        err = train_err.iloc[:, 3]

        return err

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
            p = sp.call([self.eval_exe, fit_nc_file, self.test_file], stdout=energy_log)
        print('evaluate status: ', p)

        y_test_pred = pd.read_csv("val_energy" + ite + ".dat", sep='\s+').iloc[:, 1].values

        err = y_test_pred - self.y_test_ref
        weights = self.weights_test

        return err, weights
