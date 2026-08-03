'''
Plot the radial evolution of stars that are classified as disk
and halo stars at z=0, separating groups of stars by user-specified
formation times.

Usage:   python RadialEvolution_rz.py 
Example: python RadialEvolution_rz.py 

Outputs: plots showing radial distribution of star particles formed between
         <tllim[i]> and <tulim[i]> at every snapshot at which such stars exist

Note that the user is expected to edit tllim, tulim, and cursim directly. Assumes
you're interested in in situ stars, but this can technically be changed by updating
<hid> and <ishalo>.
'''

import numpy as np
import matplotlib.pyplot as plt
import pynbody
import h5py
import tangos as db
from scipy import stats
import sys

fsize=14

plt.style.use('default')
plt.rcParams['xtick.labelsize'] = fsize
plt.rcParams['ytick.labelsize'] = fsize

cursim = 'r431' # What is the current simulation?
# The arrays below create time bounds for formation times of different groups of stars and can be any length
# the user wants, so long as they are the same length. So, for instance, if tllim = [0,5] and tulim = [4,10],
# two groups of stars will be plotted: those that formed between 0 and 4 Gyr and those that formed between 5 and 
# 10 Gyr. I usually time these groups such that they reflect merger timing (i.e., stars formed before a major merger,
# stars formed during a major merger, and stars formed after a major merger)
tllim = [0,3.7,5.5,6.5,8.6] 
tulim = [3.7,5.5,6.5,8.6,14]

opath = '/Users/Anna/Research/Outputs/dwarf_stellar_halos/'+str(cursim)+'/' # Where do you want outputs from this script to be written?
spec = '.romulus25.3072g1HsbBH' # what's the rest of the rootname for the simulation?
datapath = '/Users/Anna/Research/Outputs/dwarf_stellar_halos/ahsdfiles/' # Where does your allhalostardata file live?
simpath = '/Volumes/Abhorsen/Data/RomZooms/' # Where does your simulation live?
plotlim = 15 # How far from this halo do you want your plots to go out (kpc)?
circ_method = 'Abadi_W26' # Which circularity method(s) do you want?
pform = 1   # If pform=1, script will assume that snapshots are located inside of
            # snapshot folders. If pform=2, it will assume that snapshots are
            # all located on the same level.
hid = 1 # What is the ID of the halo we're centering on at z=0? Almost always 1 for MMs
dbkey = ''  # Is there a unique identifier for this simulation in your tangos db? If you only have one sim
              # that starts with cursim, you can set dbkey=''

ishalo = '4096_1' # What is the host ID for the "in situ" halo?
clim = 0.55 # What's the lower limit that you are using for circ of disk stars?

sim = db.get_simulation('%'+cursim+'%'+dbkey+'%')

# Read in data from allhalostardata and circularity files
with h5py.File(datapath+cursim+'_allhalostardata.h5','r') as f:
    hostids = f['host_IDs'].asstr()[:]
    partids = f['particle_IDs'][:]
    pct = f['particle_creation_times'][:]
with h5py.File(opath+cursim+'_circ.h5','r') as f:
    circ = f[circ_method+'_circ'][:]

hnum,tslist = sim[-1][1].calculate_for_progenitors('finder_id()','step_path()')

for hid,steppath in zip(hnum,tslist):
    stval = int(steppath[-6:])
    if pform == 1:
        simloc = simpath+cursim+spec+'/'+cursim+spec+'.'+str(stval).zfill(6)+'/'+cursim+spec+'.'+str(stval).zfill(6)
    elif pform ==2:
        simloc = simpath+cursim+spec+'/'+cursim+spec+'.'+str(stval).zfill(6)
    else:
        print ('Error: Path format not understood')
        sys.exit()
    s = pynbody.load(simloc)
    h = s.halos()
    s.physical_units()
    pynbody.analysis.halo.center(h[int(hid)])

    allstars = s.s[s.s['tform']>0]

    ctr = 0
    for t_low,t_up in zip(tllim,tulim): # for each pair of time limits...

        # grab halo stars that have formed
        relstars_h = partids[np.where((circ<clim) & (pct>t_low) & (pct<t_up) & (hostids==ishalo))]
        relinds_h = np.isin(allstars['iord'],relstars_h)

        # grab disk stars that have formed
        relstars_d = partids[np.where((circ>clim) & (pct>t_low) & (pct<t_up) & (hostids==ishalo))]
        relinds_d = np.isin(allstars['iord'],relstars_d)
        
        massarr_h = allstars['mass'][relinds_h]
        stars_r_h = allstars['r'][relinds_h]
        massarr_d = allstars['mass'][relinds_d]
        stars_r_d = allstars['r'][relinds_d]

        if len(stars_r_d)>0 or len(stars_r_h)>0: # if any stars in our range exist at this snapshot...
            # create histogram showing radial distribution
            rbins = np.arange(0,plotlim,0.02)
            ret_tot = stats.binned_statistic(stars_r_h, massarr_h,'sum', bins=rbins)
            dmass_h = ret_tot.statistic
            ret_tot = stats.binned_statistic(stars_r_d, massarr_d,'sum', bins=rbins)
            dmass_d = ret_tot.statistic
            plt.plot(rbins[1:],dmass_h/max(dmass_h),'r-',label='t$_\mathrm{form}$='+str(t_low)+'-'+str(t_up)+' Gyr, halo')
            plt.plot(rbins[1:],dmass_d/max(dmass_d),'k-',label='t$_\mathrm{form}$='+str(t_low)+'-'+str(t_up)+' Gyr, disk')
            plt.xlim(0,plotlim)
            plt.ylim(0,1.01)
            plt.xlabel('r (kpc)', fontsize=fsize)
            plt.ylabel('Normalized Mass Profile', fontsize=fsize)
            plt.legend(fontsize=fsize)
            plt.savefig(opath+cursim+'_'+str(ctr)+'_'+str(stval).zfill(4)+'.png',bbox_inches='tight',dpi=150)
            plt.close()
        ctr += 1