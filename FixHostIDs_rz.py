'''
Optional: Step 6b of stellar halo pipeline
Updates the host_ID values stored in the allhalostardata hdf5 file based
on user input. This is designed as a follow-up to TrackDownStars and can 
be used in a couple of ways. If ffile=True, this script will look for 
numpy files with names <sim>_new_ID_?.npy and will assign the particles
with the iords in a given file to new_ID. If ffile=False, it will assign
all particles with a host_ID in the old_ID list to new_ID.

Output: <sim>_allhalostardata_upd.h5

Usage:   python FixHostIDs_rz.py <sim>
Example: python FixHostIDs_rz.py r634 

The script will print out all host_IDs for which the number of assigned particles 
changed and how many particles each gained/lost. If the output looks correct,
the user should manually rename <sim>_allhalostardata_upd.h5 to <sim>_allhalostardata.h5.
It's often necessary to go back and forth between this and TrackDownStars, in which 
case I usually move the *.npy files that have already been processed to a subfolder. 
'''

import tangos as db
import numpy as np
import h5py
from collections import defaultdict
import glob
import sys

if len(sys.argv) != 2:
    print ('Usage: python FixHostIDs_rz.py <sim>')
    sys.exit()
else:
    cursim = str(sys.argv[1])

ffile = False
# If you're not using the autodetection from file name method,
# enter the host ID(s) you want to correct in old_ID and the IDs
# you want to replace them with in new_ID. If ffile=True, these
# will both be ignored
old_ID = ['1536_-1']
new_ID = '4096_1'

odir = '/Users/Anna/Research/Outputs/M33Analogs/MM/'+cursim+'/' # Where does your allhalostardata hdf5 file live?

with h5py.File(odir+cursim+'_allhalostardata.h5','r') as f:
    hostids = f['host_IDs'].asstr()[:]
    partids = f['particle_IDs'][:]
    pct = f['particle_creation_times'][:]
    ph = f['particle_hosts'][:]
    pp = f['particle_positions'][:]
    ts = f['timestep_location'][:]
uIDs = np.unique(hostids)

# Make a dictionary of host_IDs
orig = {}
for i in uIDs:
    nparts = len(partids[hostids==i])
    orig[i] = nparts

uphost = []
newIDlist = []
if ffile == True: # if we're using files 
    partfiles = glob.glob(odir+cursim+'_????_*.npy')
    uphost = np.copy(hostids)
    for pf in partfiles: # for each file
        nstr = pf.split('/')[-1].split('_') # figure out new host name
        new_ID = nstr[1]+'_'+nstr[2]
        if new_ID not in orig and new_ID not in newIDlist: 
            newIDlist.append(new_ID)
        curexparts = np.load(pf)
        uphost[np.isin(partids,curexparts)] = new_ID # update relevant particles
    uphost = uphost.tolist()
else: # if we're going off of host names
    exparts = partids[np.isin(hostids,old_ID)] 
    for ctr in range(0,len(partids)):
        if partids[ctr] in exparts: # update relevant particles
            uphost.append(new_ID)
        else:
            uphost.append(hostids[ctr])
    if new_ID not in orig:
        newIDlist.append(new_ID)

assert(len(hostids)==len(uphost)) # Make sure we didn't somehow lose some particles

# write out new data
with h5py.File(odir+cursim+'_allhalostardata_upd.h5','w') as f:
    f.create_dataset('particle_IDs', data=partids)
    f.create_dataset('particle_positions', data=pp)
    f.create_dataset('particle_creation_times', data=pct)
    f.create_dataset('timestep_location', data=ts)
    f.create_dataset('particle_hosts', data=ph)
    f.create_dataset('host_IDs', data=uphost, dtype="S10")

# let the user know which hosts lost/gained particles
for key,item in orig.items():
    upd_npart = len(partids[np.array(uphost)==key])
    d_npart = upd_npart-item
    if d_npart != 0:
        if d_npart>0:
            chword = 'gained'
        else:
            chword = 'lost'
        print (key+': '+str(np.abs(d_npart))+' particles '+chword)

for i in newIDlist:
    upd_npart = len(partids[np.array(uphost)==i])
    print (i+': '+str(upd_npart)+' particles gained')

print ('---------------------------------')
print ('If this looks correct, run')
print ('mv '+odir+cursim+'_allhalostardata_upd.h5 '+odir+cursim+'_allhalostardata.h5')

'''
Created on Mar 4, 2024

@author: anna
'''
