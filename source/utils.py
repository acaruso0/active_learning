def write_energy_file(self, infile, outfile='tofitE.dat', picked_idx=[], col_to_write=0):
    max_col_idx = 0
    if type(col_to_write) is int:
        max_col_idx = col_to_write + 1
    elif type(col_to_write) is list:
        try:
            max_col_idx = max(col_to_write)
        except:
            print('Error !!!')
            return False

    _, e = self.read_data(infile, picked_idx=picked_idx, E_columns=max_col_idx)
    np.savetxt(outfile,  e[:, col_to_write], delimiter='    ')
    return True

def generate_set(self, infile=None, outfile='set.xyz', picked_idx=[]):
    try:
        data = pd.read_csv(infile, sep='\s+', names=range(4))
        Natom = int(data.iloc[0][0])
        with open(outfile, 'w') as f:
            for i in picked_idx:
                line_start = i * (Natom + 2)
                try:
                    print('{0:s} '.format(data.iloc[line_start, 0]), end='\n', file=f)  # number of atoms in one configuration
                    for d in data.iloc[line_start+1, :]:
                        try:
                            if np.isnan(d):
                                continue
                            else:
                                print(' {0:s} '.format(str(d)), end='\t', file=f)  # energies
                        except:
                            print(' {0:s} '.format(str(d)), end='\t', file=f)  # energies
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
