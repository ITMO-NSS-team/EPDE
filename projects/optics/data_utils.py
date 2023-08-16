import os
import numpy as np

def read_data(r0_fix):

    data_dir=os.path.abspath(os.path.join(os.path.dirname( __file__ ), 'optics_data'))

    grid_file=os.path.join(data_dir,'grid_{}.csv'.format(r0_fix))

    rv_file=os.path.join(data_dir,'R_{}.csv'.format(r0_fix))

    grid=np.genfromtxt(grid_file, delimiter=',')

    rv=np.genfromtxt(rv_file, delimiter=',')

    return grid,rv
