'''
Step 1 of stellar halo pipeline
Grabs formation time and iord for every star in specified simulation
Prints out number of stars found as a sanity check

Output: <sim>_tf.npy

Usage:   python GrabTF_rz.py <sim>
Example: python GrabTF_rz.py r634

Note that this is currently set up for MMs, but should be easily adapted 
by e.g., changing the paths or adding a path CL argument. 
'''

import numpy as np
import pynbody
import sys

if len(sys.argv) != 2:
    print ('Usage: python GrabTF_rz.py <sim>')
    sys.exit()
else:
    cursim = str(sys.argv[1])

ofile = '/Users/Anna/Research/Outputs/M33Analogs/'+cursim+'_tf.npy'
simpath = '/Volumes/Audiobooks/RomZooms/'+cursim+'.romulus25.3072g1HsbBH/'

s = pynbody.load(simpath+simpath.split('/')[-2]+'.004096/'+simpath.split('/')[-2]+'.004096')
tf = s.s['tform'][s.s['tform']>0].in_units('Gyr')
iord = s.s['iord'][s.s['tform']>0]

print (str(len(tf))+' stars found!')

outarr = np.vstack((iord,tf))

np.save(ofile,outarr)

'''
Created on Mar 4, 2024

@author: anna
'''