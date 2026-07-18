'''
Adapted from old powderday code

Calculates luminosities of star particles in user-specified filters
using FSPS with MIST isochrones. Note that you will need to have pyFSPS
(https://python-fsps.readthedocs.io/en/latest/) installed. You should also
expect this script to take longer than most of the others in this pipeline.
You can re-run this script if you decide you'd like to add more filters - 
it will check for the existence of the output file and add new datasets to it 
if it exists; just make sure to keep an eye on the file size!

Output: <sim>_luminosities.h5
        Order of star particles should be identical to allhalostardata 
        file.

Usage:   python SaveLums_rz.py <sim> optional:<nproc> 
Example: python SaveLums_rz.py r634 4

Please note that this currently assumes you want to use MIST isochrones
and that you've compiled your FSPS accordingly! This is the default, so
you don't need to worry about this unless you altered this during pyFSPS 
set-up. If you do want to use different isochrones, you will need to 
download corresponding isochrone files (equivalent of zlegend.mist.dat)
and alter the script a bit.
'''

import numpy as np
from multiprocessing import Pool
import fsps
import os
import pickle
import pynbody
import h5py
import sys

n_processes = 4 # How many processes would you like to run with?

if len(sys.argv) < 2:
    print ('Usage: python SaveLums_rz.py <sim> optional:<nproc> ')
    print ('Default is 4 processes')
    sys.exit()
else:
    cursim = str(sys.argv[1])
    if len(sys.argv) == 3:
        n_processes = int(sys.argv[2])
    elif len(sys.argv) > 3:
        print ('Usage: python SaveLums_rz.py <sim> optional:<nproc> ')
        print ('Default is 4 processes')
        sys.exit()        

mform = 994 # formation mass of stars in Msol
imf = 2 # FSPS IMF type - 2 is a Kroupa+01 IMF
spec = '.romulus25.3072g1HsbBH' # what's the rest of the rootname for the simulation?
simpath = '/Volumes/Abhorsen/Data/RomZooms/' # where does your simulation live?
opath = '/Users/Anna/Research/Outputs/dwarf_stellar_halos/'+cursim+'/' # where would you like this file to be written?
datapath = '/Users/Anna/Research/Outputs/dwarf_stellar_halos/ahsdfiles/' # where does your allhalostardata file live?
st = 4096 # What snapshot are you running this at? Assumed to be same as snapshot allhalostardata was generated for
pform = 1   # If pform=1, script will assume that snapshots are located inside of
            # snapshot folders. If pform=2, it will assume that snapshots are
            # all located on the same level.
Lsol = 3.828*10**33 # luminosity of Sun in ergs/s

# What set of filters would you like to run? Several examples of sets of FSPS filters are listed below
filterlist = ['Roman_F062', 'Roman_F087', 'Roman_F106', 'Roman_F129', 'Roman_F158', 'Roman_F184']
#filterlist = ['U', 'B', 'V', 'Cousins_R', 'Cousins_I','SDSS_g','SDSS_r']
# filterlist = ['LSST_g']
#filterlist = ['Euclid_VIS']

sp = fsps.StellarPopulation() # initialize fsps object
metleg = '/Users/Anna/DwarfHalos/Imaging/zlegend.mist.dat'
zpfile = '/Users/Anna/DwarfHalos/Imaging/FSPS_SolABMags.pkl'

with open(zpfile,'rb') as f:
    BandSun = pickle.load(f)
    
class Stars:
    def __init__(self,mass,age,lum=-1,fsps_zmet=20,id=-1):
        self.mass = mass
        self.age = age
        self.lum = lum
        self.fsps_zmet = fsps_zmet
        self.id = id
    def info(self):
        return(self.mass,self.age,self.lum,self.fsps_zmet,self.id)

def fsps_metallicity_interpolate(metals,nstars):

    # takes a list of metallicities for star particles, and returns a
    # list of interpolated metallicities
    
    fsps_metals = np.loadtxt(metleg)
    
    zmet = []

    for i in range(nstars):
        zmet.append(find_nearest_zmet(fsps_metals,metals[i]))
    
    return zmet
    
def find_nearest_zmet(array,value):
    # this is modified from the normal find_nearest in that it forces
    # the output to be 1 index higher than the true value since the
    # minimum zmet value fsps will take is 1 (not 0)

    idx = (np.abs(array-value)).argmin()
    
    return idx+1
    
def find_nearest(array,value):
    idx = (np.abs(array-value)).argmin()

    return idx
    
def newstars_gen(stars_list):

    stellar_lum = np.zeros((len(stars_list),len(filterlist)))
    stellar_id = np.zeros(len(stars_list))
    for i in range(len(stars_list)):
        # Set parameters
        sp.params["tage"] = stars_list[i].age
        sp.params["imf_type"] = imf
        sp.params["sfh"] = 0
        sp.params["zmet"] = stars_list[i].fsps_zmet
        sp.params["add_neb_emission"] = False
        
        # Grab spectrum
        Mv = sp.get_mags(tage=stars_list[i].age,zmet=stars_list[i].fsps_zmet,bands=filterlist)
        Lv = [Lsol*(10**(-1*(m-BandSun[filt])/2.5))*stars_list[i].mass for m,filt in zip(Mv,filterlist)]
        stellar_lum[i] = Lv
        stellar_id[i] = stars_list[i].id

    return [stellar_lum,stellar_id]

if __name__ == '__main__':
    if pform == 1:
        simloc = simpath+cursim+spec+'/'+cursim+spec+'.'+str(st).zfill(6)+'/'+cursim+spec+'.'+str(st).zfill(6)
    elif pform ==2:
        simloc = simpath+cursim+spec+'/'+cursim+spec+'.'+str(st).zfill(6)
    else:
        print ('Error: Path format not understood')
        sys.exit()

    with h5py.File(datapath+cursim+'_allhalostardata.h5','r') as f:
        partids = f['particle_IDs'][:]

    s = pynbody.load(simloc)
    starmask = s.s['tform']>0
    ct = s.properties['time'].in_units('Gyr')
    age = ct-s.s['tform'][starmask].in_units('Gyr')
    metals = s.s['metals'][starmask]
    ids = s.s['iord'][starmask]
    mass = np.ones(len(age))*mform
    nstars = len(mass)
    zmet = fsps_metallicity_interpolate(metals,nstars) # find closest metallicities we have isochrones for

    stars_list = []
    for i in range(nstars):
        stars_list.append(Stars(mass[i],age[i],fsps_zmet=zmet[i],id=ids[i]))
        
    nprocesses = np.min([n_processes,len(stars_list)]) #the pool.map will barf if there are less star bins than process threads
    print ('Initializing ',nprocesses)

    #initialize the process pool and build the chunks
    p = Pool(processes = nprocesses)
    nchunks = nprocesses

    chunk_start_indices = []
    chunk_start_indices.append(0) #the start index is obviously 0

    #this should just be int(nstars/nchunks) but in case nstars < nchunks, we need to ensure that this is at least  1
    delta_chunk_indices = np.max([int(nstars / nchunks),1])

    for n in range(1,nchunks):
        chunk_start_indices.append(chunk_start_indices[n-1]+delta_chunk_indices)

    list_of_chunks = []
    for n in range(nchunks):
        stars_list_chunk = stars_list[chunk_start_indices[n]:chunk_start_indices[n]+delta_chunk_indices]
        #if we're on the last chunk, we might not have the full list included, so need to make sure that we have that here
        if n == nchunks-1:
            stars_list_chunk = stars_list[chunk_start_indices[n]::]

        list_of_chunks.append(stars_list_chunk)

    chunk_sol = p.map(newstars_gen, [arg for arg in list_of_chunks])

    p.close()
    p.terminate()
    p.join()

    orderedlumlist = np.zeros((nstars,len(filterlist)))
    for i in range(0,len(chunk_sol)):
        pos = np.where(ids==chunk_sol[i][1][0])
        orderedlumlist[pos[0][0]:(pos[0][0]+len(chunk_sol[i][1]))] = chunk_sol[i][0]
    
    if not np.array_equal(partids,ids): # if these are not the same, we need to re-order our luminosities
        index = np.argsort(ids)
        sorted_allstars = ids[index]
        sorted_index = np.searchsorted(sorted_allstars,partids)
        pindex = np.take(index,sorted_index,mode="clip")
        mask = ids[pindex] != partids
        res = np.ma.array(pindex,mask=mask)

        allstars_inhdf5 = ids[np.ma.compressed(res)]

        assert(np.array_equal(partids,allstars_inhdf5)) # these had better be the same

        # now re-order luminosities
        orderedlumlist = orderedlumlist[np.ma.compressed(res),:]

    ofile = opath+cursim+'_luminosities.h5'
    exstat = 0
    if os.path.isfile(ofile):
        partids_lum = f['particle_IDs'][:]
        if not np.array_equal(partids_lum,partids): # make sure the iords in the existing file match our current order
            print ('The iords in your existing luminosity file do not match those of the stars you are attempting to write. Please reconsider.')
            exit()
        exstat = 1

    print ('Writing luminosities to '+ofile)

    with h5py.File(ofile,'a') as f:
        if exstat == 0:
            f.create_dataset('particle_IDs', data=partids)
        for i in range(0,len(filterlist)):
            f.create_dataset(filterlist[i],data=orderedlumlist[:,i])
'''
Created on Aug 20, 2021

@author: anna
'''
