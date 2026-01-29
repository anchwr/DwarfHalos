'''
Step 5 of stellar halo pipeline
Stores the unique ID of each star particle's host at formation time.
Creates an hdf5 file that contains this in addition to all of the data
from the <sim>_stardata_<snapshot>.h5 files. Note that all star particles
that don't have a host in the snapshot after they formed will be assigned 
a unique ID of <snapshot_index>_0 and a particle host (i.e., host at
formation time) of -1. It is recommended that you use the TrackDownStars
Jupyter notebook to try to manually identify hosts for these stars and then
use FixHostIDs_rz.py to amend <sim>_allhalostardata.h5.

Output: <sim>_allhalostardata.h5

Usage:   python StoreUniqueHostID_rz.py <sim>
Example: python StoreUniqueHostID_rz.py r634 

Note that this is currently set up for MMs, but should be easily adapted 
by e.g., changing the paths or adding a path CL argument.
'''

import numpy as np
import h5py
import glob
from collections import defaultdict
import sys

if len(sys.argv) != 2:
    print ('Usage: python StoreUniqueHostID_rz.py <sim>')
    sys.exit()
else:
    cursim = str(sys.argv[1])

hsfiles = glob.glob('/Users/Anna/Research/Outputs/M33Analogs/MM/'+cursim+'/'+cursim+'_stardata_??????.h5')
hostfile = '/Users/Anna/Research/Outputs/M33Analogs/MM/'+cursim+'/'+cursim+'_uniquehalostarhosts.txt'
ofile = '/Users/Anna/Research/Outputs/M33Analogs/MM/'+cursim+'/'+cursim+'_allhalostardata.h5'

lhlist = []
tslist = []
idlist = []
pplist = []
ctlist = []
ghlist = []

# Grab data from each stardata hdf5 file
def GrabHosts(fname):
    with h5py.File(fname,'r') as f:
        print (fname,f.keys())
        hst = f['particle_hosts'][:]
        hst[hst==0] = -1
        tst = f['timestep_location'][:]
        idlist.extend(f['particle_IDs'][:])
        pplist.extend(f['particle_positions'][:])
        ctlist.extend(f['particle_creation_times'][:])
        lochostlist = [np.string_(hd[str(int(t))+','+str(int(h))]) for t,h in zip(tst,hst)]
        ghlist.extend(lochostlist)
        lhlist.extend(hst)
        tslist.extend(tst)
    return

# Assign local IDs to corresponding unique IDs
rf = open(hostfile,'r')
hd = {}
hd = defaultdict(lambda:'',hd)
rlist = rf.readline()
while rlist != '':
    rs = str(rlist).split()
    for ri in rs[1:]:
        rg = str(ri).split(',')
        hd[str(rs[0])+','+str(rg[1])] = str(rg[0])
    rlist = rf.readline()
rf.close()

for ohf in hsfiles:
    GrabHosts(ohf)

# Write out your data
with h5py.File(ofile,'w') as f:
    f.create_dataset('particle_IDs', data=idlist) # iords
    f.create_dataset('particle_positions', data=pplist) # position at formation time
    f.create_dataset('particle_creation_times', data=ctlist) # time of formation
    f.create_dataset('timestep_location', data=tslist) # snapshot where star first appears
    f.create_dataset('particle_hosts', data=lhlist) # host at that snapshot
    f.create_dataset('host_IDs', data=ghlist, dtype="S10") # unique ID of host
      
'''
Created on Aug 20, 2021

@author: anna
'''
