class Energy():
    def __init__(self, coords, pick, atom_lbl, username, path, t):
        self.atoms = atom_lbl
        self.coords = coords
        self.pick = pick
        self.it = t
        self.username = username
        self.path = path
        self.energy, self.idx_pick, self.idx_fail = self.calculate_energy()

    def calculate_energy(self):
        subscript = self.split()
        jobs = self.run(self.pick, subscript)

        check = self.check()
        while np.isin(check, jobs).any():
            sleep(60)
            check = self.check()

        it_fld = os.path.join(self.path, 'it_' + str(self.it))
        os.mkdir(it_fld)
        failed = []
        for i in self.pick:
            fail_str = ['tail', '-n', '1', F'{str(i)}/input.log']
            check_fail = sp.Popen(fail_str, stdout=sp.PIPE)
            check_fail = check_fail.communicate()[0]
            pick_fld = os.path.join(self.path, str(i))
            if check_fail != b' Variable memory released\n':
                failed.append(i)
                shutil.rmtree(pick_fld)
            else:
                shutil.move(pick_fld, it_fld)
        idx_pick = self.pick[ ~np.in1d(self.pick, failed) ]

        energies = self.extract(idx_pick)

        return np.array(energies), idx_pick, np.array(failed)

    def check(self):
        check_str = ['squeue', '-u', self.username]
        check = sp.Popen(check_str, stdout=sp.PIPE)
        out_str = io.StringIO(check.communicate()[0].decode('utf-8'))
        df = pd.read_csv(out_str, sep='\s+')

        return df['JOBID'].to_numpy().astype(int)

    def split(self):
        # split
        for n in self.pick:
            pick_fld = os.path.join(self.path, str(n))
            os.mkdir(pick_fld)
            input_path = os.path.join(pick_fld, 'input.xyz')
            with open(input_path, 'w+') as infile:
                infile.write(F'{len(self.atoms)}\n\n')
                for atom_n, atom in enumerate(self.atoms):
                    infile.write(F'{atom} {self.coords[n][atom_n][0]} {self.coords[n][atom_n][1]} {self.coords[n][atom_n][2]}\n')
            self.create_input(n, pick_fld)
        subscript = self.create_submit()

        return subscript

    def run(self, job_lst, subscript):
        jobs = []
        path_script = os.path.join(self.path, subscript)

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

        return np.array(jobs)

    def extract(self, idx_pick):
        mon2_opt = -76.37486030 #! CHANGE THIS: USER INPUT
        energies = []
        for dct in idx_pick:
            path_to_out = os.path.join(self.path, 'it_' + str(self.it), str(dct), 'input.log')
            with open(path_to_out, 'r+') as outfile:
                outfile = mmap.mmap(outfile.fileno(), 0)
            is_there = re.search(b'IE_CBS.*\n', outfile)
            if is_there:
                IE = float(is_there.group(0).split()[2])
                energies.append(IE)
            is_there = re.search(b'E_A_A_CBS.*\n', outfile)
            if is_there:
                # Formation energy of MON1 CONVERSION IN KCAL/MOL:
                MON2 = (float(is_there.group(0).split()[2]) - mon2_opt) * 627.509
                MON1 = 0.0 #! Generalize this
            BE = IE + MON1 + MON2

            # Writes new .xyz
            path_to_xyz = os.path.join(self.path, 'it_' + str(self.it), str(dct), "input.xyz")
            path_to_main = os.path.join(self.path, 'it_' + str(self.it), "tr_set.xyz")

            with open(path_to_xyz, 'r') as single_xyz:
                coords = single_xyz.readlines()
            coords[1] = F'{BE} {IE} {MON1} {MON2}\n'
            with open(path_to_main, 'a+') as main_xyz:
                main_xyz.writelines(coords)
        return energies

    def create_submit(self):
        email = F'{self.username}@ucsd.edu'
        molpro = F'/home/{self.username}/codes/software/molpro/bin'
        submit_path = os.path.join(self.path, 'comet_molpro.sh')
        with open(submit_path, 'w+') as sub_file:
            sub_file.write(F'''#!/bin/bash
#SBATCH --job-name=molpro
#SBATCH --output=STDOUT
#SBATCH --partition=shared
#SBATCH --mail-type=ALL
#SBATCH --mail-user={email}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --export=ALL
#SBATCH -t 02:00:00

cores_for_job=4

#Environment
module unload mvapich2_ib
module load lapack
export PBS_NODEFILE=`generate_pbs_nodefile`
export SLURM_NODEFILE=`generate_pbs_nodefile`

SLURM_SUBMIT_DIR=`pwd`
MOLPRO_HOME="{molpro}"

SCRATCH=`mktemp -d /oasis/scratch/comet/$USER/temp_project/batch_serial.XXXXXXXX`

START_TIME=`date`
echo The calculation started at: ${{START_TIME}}...

        "${{MOLPRO_HOME}}/molpro"    -n $cores_for_job  \\
                                     -o input.log       \\
                                     -d "${{SCRATCH}}"  \\
                                        input
END_TIME=`date`
echo The calculation ended at: ${{END_TIME}}..
rm -rf "${{SCRATCH}}"

cd $SLURM_SUBMIT_DIR''')

        return "comet_molpro.sh"

    def create_input(self, conf, pick_fld):
        input_path = os.path.join(pick_fld, 'input')
        with open(input_path, 'w+') as sub_file:
            sub_file.write('''memory,256,M

gthresh,zero=1.0e-16,twoint=3.0e-15,energy=1.0e-12,gradient=1.0e-8
gprint,orbitals

X_LOW=3
X_HIGH=4

PROC EXTRAP
  A=EXP(-1.63*(X_LOW-X_HIGH))
  E_HF_CBS=(HF_LOW - A*HF_HIGH)/(1-A)
  B=(X_LOW/X_HIGH)^(-4.255221)
  E_CCSD_CBS=(CCSD_LOW - B*CCSD_HIGH)/(1-B)
  C=(X_LOW/X_HIGH)^(-3.195354)
  E_T_CBS=(T_LOW - C*T_HIGH)/(1-C)
ENDPROC

SYMMETRY,NOSYM
geomtyp=xyz
geometry={
''')
            for atom_n, atom in enumerate(self.atoms):
                sub_file.write(F'{atom} {self.coords[conf][atom_n][0]} {self.coords[conf][atom_n][1]} {self.coords[conf][atom_n][2]}\n')
            sub_file.write('''
}

basis={
default=avtz
}

CHARGE=-1,SPIN=0
hf
HF_LOW=ENERGY
{mp2}
{ccsd(t)-f12b,THRDEN=1.0e-9,THRVAR=1.0e-11}
CCSD_LOW=ENERGC-ENERGR
T_LOW=ENERGY-ENERGC

e_AB_AB_tz=energy


basis={
default=avqz
}

CHARGE=-1,SPIN=0
hf
HF_HIGH=ENERGY
{mp2}
{ccsd(t)-f12b,THRDEN=1.0e-9,THRVAR=1.0e-11}
CCSD_HIGH=ENERGC-ENERGR
T_HIGH=ENERGY-ENERGC

e_AB_AB_qz=energy

EXTRAP

e_AB_AB_cbs=E_HF_CBS+E_CCSD_CBS+E_T_CBS


geometry={
''')
            for atom_n, atom in enumerate(self.atoms[-3:]):
                sub_file.write(F'{atom} {self.coords[conf][atom_n + 1][0]} {self.coords[conf][atom_n + 1][1]} {self.coords[conf][atom_n + 1][2]}\n')
            sub_file.write('''
}

basis={
default=avtz
}

CHARGE=0,SPIN=0
hf
HF_LOW=ENERGY
{mp2}
{ccsd(t)-f12b,THRDEN=1.0e-9,THRVAR=1.0e-11}
CCSD_LOW=ENERGC-ENERGR
T_LOW=ENERGY-ENERGC

e_A_A_tz=energy


basis={
default=avqz
}

CHARGE=0,SPIN=0
hf
HF_HIGH=ENERGY
{mp2}
{ccsd(t)-f12b,THRDEN=1.0e-9,THRVAR=1.0e-11}
CCSD_HIGH=ENERGC-ENERGR
T_HIGH=ENERGY-ENERGC

e_A_A_qz=energy

EXTRAP

e_A_A_cbs=E_HF_CBS+E_CCSD_CBS+E_T_CBS

e_B_B_tz=-459.81991694
e_B_B_qz=-459.82933131
e_B_B_cbs=-459.83321874


IE_tz=(e_AB_AB_tz-e_A_A_tz-e_B_B_tz)*tokcal
IE_qz=(e_AB_AB_qz-e_A_A_qz-e_B_B_qz)*tokcal
IE_cbs=(e_AB_AB_cbs-e_A_A_cbs-e_B_B_cbs)*tokcal
''')

        return None
