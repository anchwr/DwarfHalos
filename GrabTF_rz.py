'''
Step 1 of stellar halo pipeline
Grabs formation time and iord for every star in specified simulation
Prints out number of stars found as a sanity check

Output: <sim>_tf.npy

Usage:   python GrabTF_rz.py <sim>
Example: python GrabTF_rz.py r634

Includes an optional argument to specify the file hierarchy format.
By default (pathformat=1), it will assume that snapshots are located inside of
snapshot folders. If pathformat=2, it will assume that snapshots are
all located on the same level.
'''

import numpy as np
import pynbody
import sys

if len(sys.argv) != 2:
    print ('Usage: python GrabTF_rz.py <sim> ')
    sys.exit()
else:
    cursim = str(sys.argv[1])

ofile = '/Users/Anna/Research/Outputs/M33Analogs/'+cursim+'_tf.npy'
spec = '.romulus25.3072g1HsbBH' # what's the rest of the rootname for the simulation?
simpath = '/Volumes/Abhorsen/Data/RomZooms/'+cursim+spec+'/' # where does your simulation live?
st = 4096 # What snapshot are you running this at?
pform = 1   # If pform=1, script will assume that snapshots are located inside of
            # snapshot folders. If pform=2, it will assume that snapshots are
            # all located on the same level.

if pform == 1:
    simloc = simpath+cursim+spec+'/'+cursim+spec+'.'+str(st).zfill(6)+'/'+cursim+spec+'.'+str(st).zfill(6)
elif pform ==2:
    simloc = simpath+cursim+spec+'/'+cursim+spec+'.'+str(st).zfill(6)
else:
    print ('Error: Path format not understood')
    sys.exit()

s = pynbody.load(simloc)
tf = s.s['tform'][s.s['tform']>0].in_units('Gyr')
iord = s.s['iord'][s.s['tform']>0]

print (str(len(tf))+' stars found!')

outarr = np.vstack((iord,tf))

np.save(ofile,outarr)

'''
Created on Mar 4, 2024

@author: anna
'''