'''
Step 1 of stellar halo pipeline
Grabs formation time and iord for every star in specified simulation
Prints out number of stars found as a sanity check

Output: <sim>_tf.npy

Usage:   python GrabTF_rz.py <simpath> <outpath>
Example: python GrabTF_rz.py /data/REPOSITORY/romulus_zooms/r431.romulus25.3072g1HsbBH/r431.romulus25.3072g1HsbBH.004096/r431.romulus25.3072g1HsbBH.004096 /home/awright/dwarf_stellar_halos/r431/
 
'''

import numpy as np
import pynbody
import sys

if len(sys.argv) != 3:
    print ('Usage: python GrabTF_rz.py <simpath> <outpath>')
    sys.exit()
else:
    simpath = str(sys.argv[1])
    outpath = str(sys.argv[2])

cursim = simpath.split('/')[-1].split('.')[0]
ofile = outpath+cursim+'_tf.npy'

s = pynbody.load(simpath)
tf = s.s['tform'][s.s['tform']>0].in_units('Gyr')
iord = s.s['iord'][s.s['tform']>0]

print (str(len(tf))+' stars found!')

outarr = np.vstack((iord,tf))

np.save(ofile,outarr)

'''
Created on Mar 4, 2024

@author: anna
'''
