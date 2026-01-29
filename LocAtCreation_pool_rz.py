'''
Step 2 of stellar halo pipeline
Identifies the host of each star particle in <sim>_tf.npy at the 
time it was formed. Note that what is stored is NOT the amiga.grp 
ID, but the index of that halo in the tangos database. The amiga.grp
ID can be backed out via tangos with sim[stepnum][halonum].finder_id.

Output: <sim>_stardata_<snapshot>.h5
        where <snapshot> is the first snapshot that a given process
        analyzed. There will be <nproc> of these files generated
        and processes will not necessarily analyze adjacent snapshots

Usage:   python LocAtCreation_pool_rz.py <sim> optional:<nproc>
Example: python LocAtCreation_pool_rz.py r634 2

Includes an optional argument to specify number of processes to run
with; default is 4. Note that this will get reduced if you've specified
more processes than you have snapshots to process.

Note that this is currently set up for MMs, but should be easily adapted 
by e.g., changing the paths or adding a path CL argument. 
'''

import numpy as np
import h5py
from astropy.table import Table
from multiprocessing import Pool
import pynbody
import tangos as db
from collections import defaultdict
import sys

n_processes = 4 # default number of processes 

if len(sys.argv)<2 or len(sys.argv)>3:
    print ('Usage: python LocAtCreation_pool_rz.py <sim> opt:<nproc>')
    print ('       default number of processes is '+int(n_processes))
    sys.exit()
elif len(sys.argv)==2:
    cursim = str(sys.argv[1])
else:
    cursim = str(sys.argv[1])
    n_processes = int(sys.argv[2])

opath = '/Users/Anna/Research/Outputs/M33Analogs/MM/'+cursim+'/'
halostarsfile = '/Users/Anna/Research/Outputs/M33Analogs/'+cursim+'_tf.npy'
simpath = '/Volumes/Audiobooks/RomZooms/'+cursim+'.romulus25.3072g1HsbBH/'

dat = np.load(halostarsfile) # load in data
halostars = dat[0]
createtime = dat[1]

# Grab times for all available snapshots
tst = [] # name of snapshot
tgyr = [] # time of snapshot in Gyr
sim = db.get_simulation(cursim+'%')
ts = sim.timesteps
for d in ts:
    tgyr.append(d.time_gyr)
    tst.append(d.extension.split('/')[0])
tgyr = np.array(tgyr)
tst = np.array(tst)
sortord = np.argsort(tgyr)
tgyr = list(tgyr[sortord])
tst = list(tst[sortord])

def FindHaloStars(dsnames):

    compidarr = []
    compposarr = []
    compctarr = []
    compsteparr = []
    comphostarr = []
    print ('MyFirstStep: ',dsnames[0].split('.')[-1])
    
    # initialize output hdf5 file
    filename = opath+cursim+'_stardata_'+dsnames[0].split('.')[-1]+'.h5'
    with h5py.File(filename,'w') as f:
        f.create_dataset('particle_IDs', (1000000,))
        f.create_dataset('particle_positions', (1000000,3))
        f.create_dataset('particle_creation_times', (1000000,))
        f.create_dataset('timestep_location', (1000000,))
        f.create_dataset('particle_hosts', (1000000,))

    # iterate through the snapshots this process has been assigned
    ctr = 0
    for step in dsnames:
        s = pynbody.load(simpath+step+'/'+step) # load in snapshot
        assert(step==s.filename.split('/')[-1]) # and make sure it's the right one

        # identify the timespan we should be checking for new stars
        # i.e., the time between the previous snapshot and this one
        ind = np.where(np.array(tst)==s.filename.split('/')[-1])[0][0]
        if ind != 0:
            low_time = tgyr[ind-1]
        else:
            low_time = 0
        high_time = tgyr[ind]
        # Which stars formed during this span?
        starinds = np.where((createtime>=low_time) & (createtime<high_time))[0]
        print (str(len(starinds))+' relevant stars in '+str(step))
        
        # In addition to the iords and formation times, grab the formation positions
        # and formation hosts of each star particle. Also store the name of the snapshot
        # that this star particle is first found in for future convenience
        x = s.s['iord']
        y = halostars[starinds]
        index = np.argsort(x)
        sorted_x = x[index]
        sorted_index = np.searchsorted(sorted_x,y)
        yindex = np.take(index,sorted_index,mode="clip")
        mask = x[yindex] != y
        res = np.ma.array(yindex,mask=mask)
        posarr = s.s['pos'][np.ma.compressed(res)].in_units('Mpc')
        ctarr = s.s['tform'][np.ma.compressed(res)].in_units('Gyr')
        idarr = s.s['iord'][np.ma.compressed(res)]
        hostarr = s.s['amiga.grp'][np.ma.compressed(res)]
        starr = np.repeat(float(s.filename.split('.')[-1]),len(ctarr))
        
        # Convert the 0s that amiga uses for the hosts of particles that aren't bound to
        # a halo to -1s and convert all other host IDs to their index in the tangos database
        fid = {}
        fid['0'] = -1
        for i in range(1,len(sim[int(ind)].halos[:])+1):
            fid[str(sim[int(ind)][int(i)].finder_id)] = i
        dbhostarr = np.array([fid[str(x)] for x in hostarr])
        assert(len(starinds)==len(idarr)) # make sure you got everything

        # periodically write data to output file
        compidarr.extend(idarr)
        compposarr.extend(posarr)
        compctarr.extend(ctarr)
        compsteparr.extend(starr)
        comphostarr.extend(dbhostarr)
        if ctr%4 == 0 or ctr==(len(dsnames)-1):
            with h5py.File(filename,'a') as f:
                del f['particle_IDs']
                del f['particle_creation_times']
                del f['timestep_location']
                del f['particle_positions']
                del f['particle_hosts']
                f.create_dataset('particle_IDs',data=compidarr)
                f.create_dataset('particle_creation_times',data=compctarr)
                f.create_dataset('particle_positions',data=compposarr)
                f.create_dataset('timestep_location',data=compsteparr)
                f.create_dataset('particle_hosts',data=comphostarr)
        ctr = ctr+1
    return
    
if __name__ == '__main__':

    # which snapshots actually contain new star particles?
    stardist = np.histogram(createtime,bins=[0]+tgyr)
    steplist = np.array(tst)[stardist[0]>0]
    print ('Stars from '+str(len(steplist))+' steps left to deal with')
    np.random.shuffle(steplist) # for load balancing

    nsteps = len(steplist)
    nprocesses = np.min([n_processes,nsteps]) # make sure we have at least as many steps as we have processes
    print ('Initializing ',nprocesses)

    #initialize the process pool and build the chunks to send to each process - adapted from powderday
    p = Pool(processes = nprocesses)
    nchunks = nprocesses
    chunk_start_indices = []
    chunk_start_indices.append(0)

    #this should just be int(nsteps/nchunks) but in case nsteps < nchunks, we need to ensure that this is at least  1
    delta_chunk_indices = np.max([int(nsteps / nchunks),1])

    for n in range(1,nchunks):
        chunk_start_indices.append(chunk_start_indices[n-1]+delta_chunk_indices)

    list_of_chunks = []
    for n in range(nchunks):
        steps_list_chunk = steplist[chunk_start_indices[n]:chunk_start_indices[n]+delta_chunk_indices]
        #if we're on the last chunk, we might not have the full list included, so need to make sure that we have that here
        if n == nchunks-1:
            steps_list_chunk = steplist[chunk_start_indices[n]::]
        list_of_chunks.append(steps_list_chunk)
    
    p.map(FindHaloStars, [arg for arg in list_of_chunks])

    p.close()
    p.terminate()
    p.join()

'''
Created on Mar 4, 2024

@author: anna
'''