import numpy as np
import pandas as pd
import subprocess as sp


def read_data(xyz_path, picked_idx=[], energies=False):
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


def get_weights(E, delta_E, E_min=None):
    if E_min is None:
        E_min = np.min(E)
    w = np.square(delta_E/(E - E_min + delta_E))
    w_mean = np.mean(w)
    w /= w_mean

    return w, E_min


def write_energy_file(infile, outfile='tofitE.dat', picked_idx=[],
                      col_to_write=0):
    max_col_idx = 0
    if type(col_to_write) is int:
        max_col_idx = col_to_write + 1
    elif type(col_to_write) is list:
        try:
            max_col_idx = max(col_to_write)
        except:
            print('Error !!!')
            return False

    _, e = read_data(infile, picked_idx=picked_idx, E_columns=max_col_idx)
    np.savetxt(outfile,  e[:, col_to_write], delimiter='    ')
    return True


def generate_set(infile=None, outfile='set.xyz', picked_idx=[]):
    try:
        data = pd.read_csv(infile, sep='\s+', names=range(4))
        Natom = int(data.iloc[0][0])
        with open(outfile, 'w') as f:
            for i in picked_idx:
                line_start = i * (Natom + 2)
                try:
                    print('{0:s} '.format(data.iloc[line_start, 0]), end='\n',
                          file=f)  # number of atoms in one configuration
                    for d in data.iloc[line_start+1, :]:
                        try:
                            if np.isnan(d):
                                continue
                            else:
                                print(' {0:s} '.format(str(d)), end='\t',
                                      file=f)  # energies
                        except:
                            # energies
                            print(' {0:s} '.format(str(d)), end='\t', file=f)
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
