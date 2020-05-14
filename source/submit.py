import os
import io
import pandas as pd
import numpy as np
import subprocess as sp
from time import sleep
from loader import Loader


class SComputer(Loader):
    def __init__(self, path):
        super().__init__()
        self.path = path
        self.jobs = None

    def run(self, job_lst, submit):
        jobs = []
        path_script = os.path.join(self.path, submit)

        for i in job_lst:
            pick_fld = os.path.join(self.path, str(i))
            os.makedirs(pick_fld, exist_ok=True)
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


class _SubmitScript(Loader):
    def __init__(self, path):
        super().__init__()
        self.path = path
        self.load_template()

    def load_template(self):
        with open('job_template', 'r') as job_temp:
            self.template = job_temp.read()
        self.template = self.template.replace('$EMAIL', self.email)
        return None

    def write_file(self):
        with open(os.path.join(self.path, self.filename), 'w+') as submit_file:
            submit_file.writelines(self.template)
        return None


class SubmitMain(_SubmitScript):
    def __init__(self, path):
        super().__init__(path)
        self.filename = 'submit.sh'
        self.load_settings()
        self.write_file()

    def load_settings(self):
        line1 = ' '.join(['module', 'load', 'python'])
        line2 = ' '.join(['module', 'load', 'gsl'])
        line3 = ' '.join(['module', 'load', 'netcdf'])
        line4 = ' '.join(['python3', 'AL_framework.py', '>', 'stdout', '2>',
                          'stderr'])
        command = '\n'.join([line1, line2, line3, line4])

        self.template = self.template.replace('$JOBNAME', 'act_learn')
        self.template = self.template.replace('$CPU', '1')
        self.template = self.template.replace('$HOURS', '48')
        self.template = self.template.replace('$COMMAND', command)


class SubmitFit(_SubmitScript):
    def __init__(self, path, alpha=0.0005):
        super().__init__(path)
        self.filename = 'submit_fit.sh'
        self.alpha = alpha
        self.load_settings()
        self.write_file()

    def load_settings(self):
        line1 = ' '.join(['module', 'load', 'gsl'])
        line2 = ' '.join(['module', 'load', 'netcdf'])
        submit_lst = [self.fit_exe, self.train_out, str(self.delta_E),
                      str(self.alpha), '>', 'fit.log', '2>', 'fit.err']
        line3 = ' '.join(submit_lst)
        command = '\n'.join([line1, line2, line3])

        self.template = self.template.replace('$JOBNAME', 'fit')
        self.template = self.template.replace('$CPU', '1')
        self.template = self.template.replace('$HOURS', '2')
        self.template = self.template.replace('$COMMAND', command)


class SubmitMolpro(_SubmitScript):
    def __init__(self, path, cpu=4):
        super().__init__(path)
        self.filename = 'submit_calc.sh'
        self.cpu = str(cpu)
        self.load_settings()
        self.write_file()

    def load_settings(self):
        tempdir = os.path.join('/', 'oasis', 'scratch', 'comet', self.username,
                               'temp_project', 'batch_serial.XXXXXXXX')
        line1 = ' '.join(['module', 'unload', 'mvapich2_ib'])
        line2 = ' '.join(['module', 'load', 'lapack'])
        line3 = ' '.join(['export', 'PBS_NODEFILE=`generate_pbs_nodefile`'])
        line4 = ' '.join(['export', 'SLURM_NODEFILE=`generate_pbs_nodefile`\n'])
        line5 = ' '.join(['SCRATCH=`mktemp', '-d', tempdir + '`'])
        line6 = ' '.join([self.molpro, '-n', self.cpu, '-o', 'input.log',
                          '-d', '"${SCRATCH}"', 'input'])
        line7 = ' '.join(['rm', '-rf', '"${SCRATCH}"'])
        command = '\n'.join([line1, line2, line3, line4, line5, line6, line7])

        self.template = self.template.replace('$JOBNAME', 'molpro')
        self.template = self.template.replace('$CPU', self.cpu)
        self.template = self.template.replace('$HOURS', '2')
        self.template = self.template.replace('$COMMAND', command)
