import h5py as h
import numpy as np

f = h.File('new_output.h5','r')
x = f['r'].items()

data = x[0][1]
uids = x[1][1]

np.save('data.npy',data)
np.save('uid.npy',uids)
