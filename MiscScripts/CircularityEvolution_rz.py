'''
Plot the orbital circularity of star particles at each snapshot
relative to the angular momentum vector of the central halo at z=0.

Usage:   python CircularityEvolution_rz.py <sim>
Example: python CircularityEvolution_rz.py r634 

Outputs: plots showing orbital circularity of star particles at each
         snapshot in the tangos db. By default, centers on the main 
         progenitor of the halo and goes out to <halolim> kpc

Note that Stinson_W24 circularity is used
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
import tangos as db
import math
import matplotlib.gridspec as gridspec
import sys
from scipy.interpolate import make_interp_spline

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
simpath = '/Volumes/Abhorsen/Data/RomZooms/' # Where does your simulation live?
halolim = 50 # How far from this halo do you want your plots to go out (kpc)?
dbkey = ''  # Is there a unique identifier for this simulation in your tangos db? If you only have one sim
              # that starts with cursim, you can set dbkey=''

with h5py.File(opath+cursim+'_circ.h5','r') as f:
    norm_L = f['angmom_vec'][:]

def round_to_n(val,n):
    '''
    Round val to n significant figures
    '''
    if val == 0:
        return 0.0
    else:
        return round(val,int(n-math.ceil(math.log10(abs(val)))))

age_color_map = sns.blend_palette(("black", "#16263B", "#386094", "#4575b4", "#4daf4a","#FFD24D", "darkorange"), as_cmap=True)

# grab snapshot paths and amiga.grp IDs for this halo's main progenitors
sim = db.get_simulation('%'+cursim+'%'+dbkey+'%')
st,hnum,tm = sim[-1][1].calculate_for_progenitors('step_path()','finder_id()','t()') 

for simloc,hid,curtime in zip(st,hnum,tm):
    print ('Running '+simloc[-6:])
    s = pynbody.load(simpath+simloc)
    h = s.halos()
    s.physical_units()

    pynbody.analysis.halo.center(h[hid]) # this centers both spatially and in terms of velocity; we are moving the entire snapshot


    disk_rot_arr = pynbody.analysis.angmom.calc_faceon_matrix(norm_L)
    if pynbody.__version__<2.0:
        pynbody.transformation.transform(s,disk_rot_arr)
    else:
        pynbody.transformation.Rotation.rotate(s,disk_rot_arr) # rotate so that our disk is face-on (again moving everything)
    allstars = s.s[s.s['tform']>0] # no black holes

    stars_r = np.linalg.norm(allstars['pos'],axis=1)
    # Calculate z-component of angular momentum
    jz = allstars['pos'][:,0]*allstars['vel'][:,1] - allstars['pos'][:,1]*allstars['vel'][:,0]
    # Calculate angular momentum of particle in perfect circular orbit at that cylindrical radius
    radii = np.logspace(np.log10(min(stars_r)),np.log10(halolim+30),2000)
    p = pynbody.analysis.profile.Profile(h[hid],bins=radii)
    totmass_enc = p['mass_enc']
    Menc_profile = make_interp_spline(np.concatenate(([0],radii[1:])), np.concatenate(([0],totmass_enc)))
    Menc = Menc_profile(stars_r)
    grav_pot = 6.67408*10**-8*Menc*(1.988*10**33)/(stars_r*3.086*10**21)
    vc = np.array([np.sqrt(gr)*10**-5 for gr in grav_pot])
    jc = vc*stars_r
    # Calculate circularity
    circ = jz/jc

    df = pd.DataFrame({'radius':stars_r, 'circularity':np.array(circ), 'age':allstars['age'].in_units('Gyr')/curtime})
    fig = plt.figure(figsize=(8,8),dpi=120)

    ax1 = fig.add_axes([0.1, 0.1, 0.85, 0.85])

    artist = dsshow(df,dsh.Point('radius','circularity'),dsh.mean('age'),norm='linear',cmap=age_color_map,x_range=(0,halolim),y_range=(-2,2), vmin=0,vmax=1,aspect='auto',ax=ax1)
    ax1.set_xlim(0,50)
    ax1.set_ylim(-2,2)
    ax1.set_xlabel('Galactocentric Distance (kpc)',fontsize=20)
    ax1.set_ylabel('Circularity',fontsize=20)
    ax1.text(0.8,0.95, 't='+str(round_to_n(float(curtime),3))+' Gyr',fontsize=13,transform=plt.gca().transAxes)


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
    plt.savefig(opath+cursim+'_circularity_SW24_'+simloc[-6:]+'.png',bbox_inches='tight')
    plt.close()