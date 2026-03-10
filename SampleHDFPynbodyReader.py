'''
Sample script to read in data from allhalostardata file and use it with pynbody
'''

import pynbody
import numpy as np
import h5py
import matplotlib.pyplot as plt

plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14

cursim = 'r442'
datapath = '/home/awright/dwarf_stellar_halos/'+cursim+'/'

s = pynbody.load('/data/REPOSITORY/romulus_zooms/'+cursim+'.romulus25.3072g1HsbBH/'+cursim+'.romulus25.3072g1HsbBH.004096/'+cursim+'.romulus25.3072g1HsbBH.004096')
s.physical_units()
h = s.halos()

allstars = s.s[s.s['tform']>0] # exclude black holes

# read in star particle information
with h5py.File(datapath+cursim+'_allhalostardata.h5','r') as f:
    hostids = f['host_IDs'][:] # unique host IDs
    partids = f['particle_IDs'][:] # iords
    pct = f['particle_creation_times'][:] # formation times in Gyr
    ph = f['particle_hosts'][:] # local host IDs (i.e., at formation time)
    pp = f['particle_positions'][:] # position at formation time
    ts = f['timestep_location'][:] # snapshot where star particle first appears
uIDs = np.unique(hostids)

# create a version of allstars that's in the same order as the data in the hdf5 file
if np.array_equal(partids,allstars['iord']): # if these are the same, we don't need to do anything else
    pass
else: # If they're not, re-order sim data so that the stars are in the same order as the hdf5 file 
    if len(allstars['iord'])!=len(partids):
        print ('WARNING: You have '+str(len(partids))+' stars in your allhalostardata file and '+str(len(allstars['iord']))+' stars in your simulation')
        # note that this will definitely trigger for r431 due to the corruption of a few initial files (+snaphost 775)

    allstarinds = allstars['iord']
    index = np.argsort(allstarinds)
    sorted_allstars = allstarinds[index]
    sorted_index = np.searchsorted(sorted_allstars,partids)
    pindex = np.take(index,sorted_index,mode="clip")
    mask = allstarinds[pindex] != partids
    res = np.ma.array(pindex,mask=mask)

    allstars_inhdf5 = allstars[np.ma.compressed(res)]

    assert(np.array_equal(partids,allstars_inhdf5['iord'])) # these had better be the same

# make a plot of the z=0 radial distributions of in situ and ex situ stars within 20 kpc of halo 1
pynbody.analysis.halo.center(h[1])
stars_r = np.linalg.norm(allstars_inhdf5['pos'],axis=1)
is_stars = hostids==b'4096_1' # mask for in situ stars
plt.hist(stars_r[is_stars],bins=np.arange(0,21,1),weights=(np.ones(len(stars_r[is_stars]))/len(stars_r[is_stars])),color='k',label='In Situ Stars',alpha=0.3)
plt.hist(stars_r[~is_stars],bins=np.arange(0,21,1),weights=(np.ones(len(stars_r[~is_stars]))/len(stars_r[~is_stars])),color='r',label='Ex Situ Stars',alpha=0.3)
plt.legend(loc=1,fontsize=14)
plt.xlim(0,20)
plt.ylabel('Fraction of In/Ex Situ Stars',fontsize=14)
plt.xlabel('Radius (kpc)', fontsize=14)
plt.show()