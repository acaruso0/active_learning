import os, io
import pandas as pd
import numpy as np
import subprocess as sp
from time import sleep


class SComputer():
    def __init__(self, path, username):
        self.username = username
        self.path = path
        self.jobs = None

    def run(self, job_lst, submit):
        jobs = []
        path_script = os.path.join(self.path, submit)

        for i in job_lst:
            pick_fld = os.path.join(self.path, str(i))
            os.chdir(pick_fld)
            job = sp.Popen(["sbatch", path_script], stdout=sp.PIPE)
            check = job.communicate()[0]

            while b'Submitted batch job' not in check:
                sleep(60)
                job = sp.Popen(["sbatch", path_script], stdout=sp.PIPE)
                check = job.communicate()[0]

            job_idx = int(check.decode('utf-8').split()[3])
            jobs.append(job_idx)
        os.chdir(self.path)

        self.jobs = np.array(jobs)

    def running_jobs(self):
        check_str = ['squeue', '-u', self.username]
        check = sp.Popen(check_str, stdout=sp.PIPE)
        out_str = io.StringIO(check.communicate()[0].decode('utf-8'))
        df = pd.read_csv(out_str, sep='\s+')

        return df['JOBID'].to_numpy().astype(int)

    def check(self):
        rjobs = self.running_jobs()
        while np.isin(rjobs, self.jobs).any():
            sleep(60)
            rjobs = self.running_jobs()
        self.jobs = None

        return None


class SubmitFile():
    def __init__(self, username, domain):
        self.filename = ''
        self.username = username
        self.email = self.username + domain
        self.load()

    def load(self):
        with open('job_template', 'r') as job_temp:
            job_temp = job_temp.read()
        self.template = job_temp.replace('$EMAIL', self.email)

        return None

    def write_file(self):
        with open(self.filename, 'w+') as submit_file:
            submit_file.writelines(self.template)

        return None


class SubmitMain(SubmitFile):
    def __init__(self, username, domain):
        super().__init__(username, domain)
        self.filename = 'submit.sh'
        self.load_settings()

    def load_settings(self):
        line1 = ' '.join(['module', 'load', 'python'])
        line2 = ' '.join(['module', 'load', 'gsl'])
        line3 = ' '.join(['module', 'load', 'netcdf'])
        line4 = ' '.join(['python3', 'AL_framework.py', '>', 'stdout', '2>', 'stderr'])
        command = '\n'.join([line1, line2, line3, line4])

        self.template = self.template.replace('$JOBNAME', 'act_learn')
        self.template = self.template.replace('$CPU', '1')
        self.template = self.template.replace('$HOURS', '48')
        self.template = self.template.replace('$COMMAND', command)


class SubmitFit(SubmitFile):
    def __init__(self, username, domain, fitting_code, train_set, delta, alpha=0.0005):
        super().__init__(username, domain)
        self.filename = 'submit_fit.sh'
        self.fitting_code = fitting_code
        self.train_set = train_set
        self.delta = delta
        self.alpha = alpha
        self.load_settings()

    def load_settings(self):
        line1 = ' '.join(['module', 'load', 'gsl'])
        line2 = ' '.join(['module', 'load', 'netcdf'])
        line3 = ' '.join([self.fitting_code, self.train_set, str(self.delta), str(self.alpha), '>', 'fit.log', '2>', 'fit.err'])
        command = '\n'.join([line1, line2, line3])

        self.template = self.template.replace('$JOBNAME', 'fit')
        self.template = self.template.replace('$CPU', '1')
        self.template = self.template.replace('$HOURS', '2')
        self.template = self.template.replace('$COMMAND', command)


test = SubmitFit("acaruso", "@ucsd.edu", '/oasis/blabla/fit-2b', 'training_set.xyz', 30)
