'''
Makes a map of the surface brightness out to <halo_lim> around halo
<hid>, binning light into pixels of <pixel_width> pc x <pixel_width> pc.
Currently assumes halo <hid> is the same halo used as reference for
CircularityCalculation_rz.py and therefore rotates halo to be side-on
using the angular momentum vector written out by that script, but this
is fairly easily changed if you want to visualize a different halo or 
alter the angle.

Usage:   python SBMap_rz.py <sim> optional:<filter>
Example: python SBMap_rz.py r634 Roman_F129

If you have already run SaveLums_rz.py to generate luminosities in 
various bands using FSPS, you can read these in and use them here.
If not, you can try using pynbody's built-in SSP tables. 
'''

import pynbody
import numpy as np
import h5py
import pickle
from scipy import stats
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colorbar import Colorbar
import sys

plt.style.use('default')
fsize=16
plt.rcParams['xtick.labelsize'] = fsize
plt.rcParams['ytick.labelsize'] = fsize

fil = 'Roman_F129' # default filter

if len(sys.argv) < 2:
    print ('Usage: python SBMap_rz.py <sim> <opt:filter>')
    print ('Default filter is Roman_F129')
    sys.exit()
else:
    cursim = str(sys.argv[1])
    if len(sys.argv) == 2:
        fil = str(sys.argv[2])
    else:
        print ('Usage: python SBMap_rz.py <sim> <opt:filter>')
        print ('Default filter is Roman_F129')
        sys.exit()       

opath = '/Users/Anna/Research/Outputs/M33analogs/MM/'+str(cursim)+'/' # Where do you want outputs from this script to be written?
datapath = '/Users/Anna/Research/Outputs/M33analogs/MM/ahsdfiles/' # Where does your allhalostardata file live?
simpath = '/Volumes/Abhorsen/Data/RomZooms/' # Where does your simulation live?
pform = 1   # If pform=1, script will assume that snapshots are located inside of
            # snapshot folders. If pform=2, it will assume that snapshots are
            # all located on the same level.
spec = '.romulus25.3072g1HsbBH' # what's the rest of the rootname for the simulation?
st = 4096 # snapshot number - assumed to be same that you ran SaveLums for if you are reading lums in
hid = 1 # What is the ID of the halo we're centering on? Almost always 1 for MMs
halo_lim = 100
pixel_width = 1500

readlum = True # Will you be reading the luminosities of individual star particles in from a file generated with FSPS?
zpfile = 'FSPS_SolABMags.pkl' # file containing zero points for different bands
Lsol = 3.828*10**33 # luminosity of Sun in ergs/s
sblow = 39
sbhigh = 23

if readlum:
    with open(zpfile,'rb') as f:
        BandSun = pickle.load(f)
    with h5py.File(opath+cursim+'_luminosities.h5','r') as f:
        lum = f[fil][:]

with h5py.File(opath+cursim+'_circ.h5','r') as f:
    partids = f['particle_IDs'][:]
    norm_L = f['angmom_vec'][:]

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

# rotate disk to side-on
disk_rot_arr = pynbody.analysis.angmom.calc_faceon_matrix(norm_L)
pynbody.transformation.Rotation.rotate(s,disk_rot_arr) # rotate so that our disk is face-on (again moving everything)
s.rotate_x(90)

# grab star data (no black holes)
allstars = s.s[s.s['tform'].in_units('Gyr')>0]

# re-order to match hdf5 file if necessary
if np.array_equal(partids,allstars['iord']):
    pos = allstars['pos']
    if not readlum:
        lum = allstars[fil+'_mag']
else:
    allstarinds = allstars['iord']
    index = np.argsort(allstarinds)
    sorted_allstars = allstarinds[index]
    sorted_index = np.searchsorted(sorted_allstars,partids)
    pindex = np.take(index,sorted_index,mode="clip")
    mask = allstarinds[pindex] != partids
    res = np.ma.array(pindex,mask=mask)
    allstars = allstars[np.ma.compressed(res)]
    pos = allstars['pos']
    if not readlum:
        mag = allstars[fil+'_mag']
        lum = 10**(-1*mag/2.5) # Note this is unnormalized - fine for SB calc

# Make surface brightness map
xypts = np.zeros((len(lum),2)) # divide region around halo into pixels
xypts[:,0] = pos[:,0].in_units('kpc')
xypts[:,1] = pos[:,1].in_units('kpc')
boxlen = pixel_width/1000. # convert pixel_width to kpc
xr = [-1*halo_lim,halo_lim]
yr = [-1*halo_lim,halo_lim]
xbins = np.arange(-1*halo_lim,halo_lim,boxlen)
ybins = np.arange(-1*halo_lim,halo_lim,boxlen)
nxbins = len(xbins)
nybins = len(ybins)
# sum luminosity in each bin
retstat = stats.binned_statistic_2d(x=xypts[:,0], y=xypts[:,1], values=lum, statistic='sum', bins=(xbins,ybins), expand_binnumbers=True)
lumarr = np.copy(retstat.statistic)
lumarr[np.isnan(lumarr)] = 0.001 # deal with empty bins
lumarr[lumarr==0] = 0.001
X,Y = np.meshgrid(retstat.x_edge,retstat.y_edge)
xbinwidth = retstat.x_edge[1]-retstat.x_edge[0]
ybinwidth = retstat.y_edge[1]-retstat.y_edge[0]
# convert to SB for each pixel (based on pynbody's conversion)
pixarea = pixel_width**2 # pixel area in pc^2
sqarcsec_in_bin = pixarea / (2.3504430539466191*10**-9) # convert to arcsec^2
if readlum:
    SB = -2.5 * np.log10((lumarr/Lsol) / sqarcsec_in_bin)+BandSun[fil]
else:
    SB = -2.5 * np.log10(lumarr/sqarcsec_in_bin)

# create figure
f3 = plt.figure(figsize=(11,9))
outer_grid = gridspec.GridSpec(1, 2, wspace=0.05, hspace=0.0, width_ratios=[25,1])

inner_grid = gridspec.GridSpecFromSubplotSpec(1,1,subplot_spec=outer_grid[0], wspace=0.00, hspace=0.09)
cbar_grid = gridspec.GridSpecFromSubplotSpec(1,1,subplot_spec=outer_grid[1], wspace=0.00, hspace=0.09)

ax11 = plt.Subplot(f3,inner_grid[0])
boundaries = np.arange(sbhigh, sblow, 1)
cmap_inferno = plt.cm.get_cmap('inferno_r',len(boundaries)+3)
colors = list(cmap_inferno(np.arange(len(boundaries))))
cmap = mpl.colors.ListedColormap(colors, "")
cmap.set_over('black')
cmap.set_under('white')
ncplot = ax11.pcolormesh(X,Y,SB,cmap=cmap,norm=mpl.colors.BoundaryNorm(boundaries, ncolors=len(boundaries), clip=False))
ax11.set_box_aspect(1)
ax11.set_xlabel('[kpc]',fontsize=fsize)
ax11.set_ylabel('[kpc]',fontsize=fsize)
f3.add_subplot(ax11)

ax21 = plt.Subplot(f3,cbar_grid[0])
cb = Colorbar(ax=ax21, mappable=ncplot, orientation='vertical',extend="both")
ax21.set_yticks(ticks=np.arange(sbhigh,sblow,2))
cb.set_label(label='Surface Brightness (mag/arcsec$^2$)',size=fsize+4)
f3.add_subplot(ax21)
plt.savefig(opath+cursim+'_SB_'+fil+'.png',dpi=300)






