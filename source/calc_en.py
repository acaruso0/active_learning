import os
import shutil
import mmap
import re
import numpy as np
import pandas as pd
import subprocess as sp
from loader import Loader
from submit import SComputer, SubmitMolpro


class Energy(Loader):
    def __init__(self, coords, pick):
        super().__init__()
        self.coords = coords
        self.pick = pick
        self.energy, self.idx_pick, self.idx_fail = self.calculate_energy()

    def calculate_energy(self):
        self.split()
        submit = SubmitMolpro(self.username, self.email, self.calculations,
                              self.molpro)
        submit_en = os.path.join(self.calculations, submit.filename)
        SComputer(self.calculations, self.username)
        SComputer.run(self.pick, submit_en)
        SComputer.check()

        it_fld = os.path.join(self.calculations, 'it_' + str(self.t))
        os.mkdir(it_fld)
        failed = []
        for i in self.pick:
            fail_str = ['tail', '-n', '1', F'{str(i)}/input.log']
            check_fail = sp.Popen(fail_str, stdout=sp.PIPE)
            check_fail = check_fail.communicate()[0]
            pick_fld = os.path.join(self.calculations, str(i))
            if check_fail != b' Variable memory released\n':
                failed.append(i)
                shutil.rmtree(pick_fld)
            else:
                shutil.move(pick_fld, it_fld)
        idx_pick = self.pick[~np.in1d(self.pick, failed)]

        energies = self.extract(idx_pick)
        return np.array(energies), idx_pick, np.array(failed)

    def split(self):
        # split
        for n in self.pick:
            pick_fld = os.path.join(self.calculations, str(n))
            os.mkdir(pick_fld)
            input_path = os.path.join(pick_fld, 'input.xyz')
            with open(input_path, 'w+') as infile:
                infile.write(F'{len(self.atoms)}\n\n')
                for atom_n, atom in enumerate(self.atoms):
                    infile.write(F'{atom} {self.coords[n][atom_n][0]} '
                                 + F'{self.coords[n][atom_n][1]} '
                                 + F'{self.coords[n][atom_n][2]}\n')
            self.InputMolpro(self.coords[n], pick_fld)
        return None

    def extract(self, idx_pick):
        energies = []
        for dct in idx_pick:
            path_to_out = os.path.join(self.calculations, 'it_' + str(self.t),
                                       str(dct), 'input.log')
            with open(path_to_out, 'r+') as outfile:
                outfile = mmap.mmap(outfile.fileno(), 0)
            is_there = re.search(b'IE_CBS.*\n', outfile)
            if is_there:
                IE = float(is_there.group(0).split()[2])
                energies.append(IE)
            is_there = re.search(b'E_A_A_CBS.*\n', outfile)
            if is_there:
                # Formation energy of MON2 CONVERSION IN KCAL/MOL:
                MON1 = (float(is_there.group(0).split()[2])
                        - self.mon1_opt) * 627.509
            is_there = re.search(b'E_B_B_CBS.*\n', outfile)
            if is_there:
                # Formation energy of MON1 CONVERSION IN KCAL/MOL:
                MON2 = (float(is_there.group(0).split()[2])
                        - self.mon2_opt) * 627.509
            BE = IE + MON1 + MON2

            # Writes new .xyz
            path_to_xyz = os.path.join(self.calculations, 'it_' + str(self.t),
                                       str(dct), "input.xyz")
            path_to_main = os.path.join(self.calculations, 'it_' + str(self.t),
                                        "tr_set.xyz")

            with open(path_to_xyz, 'r') as single_xyz:
                coords = single_xyz.readlines()
            coords[1] = F'{BE} {IE} {MON1} {MON2}\n'
            with open(path_to_main, 'a+') as main_xyz:
                main_xyz.writelines(coords)
        return energies


class _InputFile(Loader):
    def __init__(self, conf, pick_fld):
        super().__init__()
        self.conf = conf
        self.pick_fld = pick_fld
        self.load_template()

    def load_template(self):
        with open('en_template', 'r') as en_temp:
            self.template = en_temp.read()
        return None

    def write_file(self):
        with open(os.path.join([self.pick_fld, "input"]), 'w+') as input_file:
            input_file.writelines(self.template)
        return None


class InputMolpro(_InputFile):
    def __init__(self, conf, pick_fld):
        super().__init__(conf, pick_fld)
        self.load_settings()
        self.write_file()

    def load_settings(self):
        slices = np.insert(np.cumsum(self.molecules), 0, 0)

        df = pd.DataFrame(self.conf, index=self.atoms)
        self.template = self.template.replace('$GEOMETRY_TOT',
                                              df.to_string(header=False))
        for n in range(slices.shape[0]-1):
            df = pd.DataFrame(self.conf,
                              index=self.atoms)[slices[n]:slices[n+1]]
            self.template = self.template.replace(F'$GEOMETRY_{n+1}',
                                                  df.to_string(header=False))
