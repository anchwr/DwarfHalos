'''
Plot the orbital circularities of the star particles in each
contributor at z=0

Usage:   python CircularityByContributor_rz.py <sim>
Example: python CircularityByContributor_rz.py r634 

Outputs: plots showing orbital circularity of star particles in a 
         given contributor to the simulation. By default, centers 
         on halo <hid> and goes out to <halolim> kpc; produces plots
         for <circ_method> as long as there is corresponding data 
         (i.e., a field called <circ_method>_circ) in the <sim>_circ.hdf5 file.
'''

import matplotlib as mpl
import matplotlib.pyplot as plt
import pynbody
import numpy as np
import pandas as pd
import datashader as dsh
import seaborn as sns
import h5py
from datashader.mpl_ext import dsshow
import matplotlib.gridspec as gridspec
import sys
from matplotlib.transforms import Bbox

fsize = 14

plt.style.use('default')
plt.rcParams['xtick.labelsize'] = fsize
plt.rcParams['ytick.labelsize'] = fsize

if len(sys.argv) != 2:
    print ('Usage: python CircularityEvolution_rz.py <sim>')
    sys.exit()
else:
    cursim = str(sys.argv[1])

opath = '/Users/Anna/Research/Outputs/dwarf_stellar_halos/'+str(cursim)+'/' # Where do you want outputs from this script to be written?
spec = '.romulus25.3072g1HsbBH' # what's the rest of the rootname for the simulation?
datapath = '/Users/Anna/Research/Outputs/dwarf_stellar_halos/ahsdfiles/' # Where does your allhalostardata file live?
simpath = '/Volumes/Abhorsen/Data/RomZooms/' # Where does your simulation live?
halolim = 50 # How far from this halo do you want your plots to go out (kpc)?
circ_method = 'Abadi_W26' # Which circularity method(s) do you want?
pform = 1   # If pform=1, script will assume that snapshots are located inside of
            # snapshot folders. If pform=2, it will assume that snapshots are
            # all located on the same level.
hid = 1 # What is the ID of the halo we're centering on? Almost always 1 for MMs
st = 4096

# grab relevant data from files
with h5py.File(opath+cursim+'_allhalostardata.h5','r') as f:
    hostids = f['host_IDs'].asstr()[:]
    partids = f['particle_IDs'][:]
    pct = f['particle_creation_times'][:]
with h5py.File(opath+cursim+'_circ.h5','r') as f:
    circ = f[circ_method+'_circ'][:]
uIDs = np.unique(hostids)

age_color_map = sns.blend_palette(("black", "#16263B", "#386094", "#4575b4", "#4daf4a","#FFD24D", "darkorange"), as_cmap=True)

# load simulation and center halo
if pform == 1:
    simloc = simpath+cursim+spec+'/'+cursim+spec+'.'+str(st).zfill(6)+'/'+cursim+spec+'.'+str(st).zfill(6)
elif pform ==2:
    simloc = simpath+cursim+spec+'/'+cursim+spec+'.'+str(st).zfill(6)
else:
    print ('Error: Path format not understood')
    sys.exit()

s = pynbody.load(simloc)
h = s.halos()
s.physical_units()
pynbody.analysis.halo.center(h[hid])

# grab star data (no black holes)
allstars = s.s[s.s['tform'].in_units('Gyr')>0]

# re-order to match hdf5 file if necessary
if not np.array_equal(partids,allstars['iord']):
    allstarinds = allstars['iord']
    index = np.argsort(allstarinds)
    sorted_allstars = allstarinds[index]
    sorted_index = np.searchsorted(sorted_allstars,partids)
    pindex = np.take(index,sorted_index,mode="clip")
    mask = allstarinds[pindex] != partids
    res = np.ma.array(pindex,mask=mask)
    allstars = allstars[np.ma.compressed(res)]

stars_r = allstars['r']
stars_age = allstars['age'].in_units('Gyr')

# for each of our contributors...
for i in uIDs:
    relstars = np.where(hostids==i)[0] # which stars formed in this contributor?

    # Make radius-circularity-age figure(s)
    df = pd.DataFrame({'radius':stars_r[relstars], 'circularity':circ[relstars], 'age':stars_age[relstars]/s.properties['time'].in_units('Gyr')})
    fig = plt.figure(figsize=(8,8),dpi=120)

    ax1 = fig.add_axes([0.1, 0.1, 0.85, 0.85])

    artist = dsshow(df,dsh.Point('radius','circularity'),dsh.mean('age'),norm='linear',cmap=age_color_map,x_range=(0,halolim),y_range=(-2,2), vmin=0,vmax=1,aspect='auto',ax=ax1)
    artist.bbox_df = Bbox([[np.nanmin(df['radius']), np.nanmin(df['circularity'])], [np.nanmax(df['radius']), np.nanmax(df['circularity'])]])
    ax1.set_xlim(0,halolim)
    if circ_method == 'Stinson' or circ_method == 'Stinson_W24':
        ax1.set_ylim(-2,2)
    elif circ_method == 'Abadi' or circ_method == 'Abadi_W26':
        ax1.set_ylim(-1.25,1.25)
    ax1.set_xlabel('Galactocentric Distance (kpc)',fontsize=20)
    ax1.set_ylabel('Circularity',fontsize=20)

    ax2 = fig.add_axes([0.58, 0.2, 0.33, 0.04])
    norm = mpl.colors.Normalize(vmin=0, vmax=1)
    cb1 = mpl.colorbar.ColorbarBase(ax2,cmap=age_color_map,norm=norm,ticks=np.arange(0,1.1,0.2),orientation='horizontal',label='Age/Current Time')
    cb1.ax.xaxis.get_label().set_fontsize(20)

    ax2.spines["top"].set_color('white')
    ax2.spines["bottom"].set_color('white')
    ax2.spines["left"].set_color('white')
    ax2.spines["right"].set_color('white')
    ax2.set_yticklabels([])
    ax2.set_yticks([])
    plt.savefig(opath+cursim+'_circularity_'+circ_method+'_'+i+'.png',bbox_inches='tight')
    plt.close()


