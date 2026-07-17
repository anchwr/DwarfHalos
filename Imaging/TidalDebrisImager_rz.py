'''
Makes plots of tidal debris from individual hosts colored by luminosity,
circularity, [Fe/H], and [O/Fe]. Also creates plots for all in situ 
material and all ex situ material.

Usage:   python TidalDebrisImager_rz.py <sim> optional:<filter>
Example: python TidalDebrisImager_rz.py r634 Roman_F129

If you have already run SaveLums_rz.py to generate luminosities in 
various bands using FSPS, you can read these in and use them here.
If not, you can try using pynbody's built-in SSP tables. However,
it looks like zero points haven't been included for a lot of the 
non-AB bands, so the conversion from magnitudes to luminosities may fail.
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
import pickle

fsize = 14

plt.style.use('default')
plt.rcParams['xtick.labelsize'] = fsize
plt.rcParams['ytick.labelsize'] = fsize

fil = 'Roman_F129' # default filter

if len(sys.argv) < 2:
    print ('Usage: python TidalDebrisImager_rz.py <sim> <opt:filter>')
    print ('Default filter is Roman_F129')
    sys.exit()
else:
    cursim = str(sys.argv[1])
    if len(sys.argv) == 2:
        fil = str(sys.argv[2])
    else:
        print ('Usage: python TidalDebrisImager_rz.py <sim> <opt:filter>')
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
ishalo = '4096_1' # What is the allhalostardata host ID of the central halo?

circtype = 'Abadi_W26_circ' # Which circularity do you want to plot
nstarlim = 100 # tidal debris must have at least nstarlim stars to be plotted in individual tidal debris plots
maxwid = 100 # maximum distance from the center of the main halo that you want to visualize

readlum = True # Will you be reading the luminosities of individual star particles in from a file generated with FSPS?
lumpath = opath+'r'+str(cursim)+'_luminosities.h5' # if so, put the path to the file here

FeSol = 0.0016 # What metallicity should be used for the Sun?

lmin = float(10**35) # what's the lowest luminosity you'd like to show (Lsol)?
lmax = float(5*10**38) # what's the highest luminosity you'd like to show (Lsol)?
mmin = -3 # what's the lowest Fe/H you'd like to show?
mmax = -1 # what's the highest Fe/H you'd like to show?
amin = 0.8 # what's the lowest O/Fe you'd like to show?
amax = 1.5 # what's the highest O/Fe you'd like to show?
cmin = 0 # what's the lowest circularity you'd like to show?
cmax = 1 # what's the highest circularity you'd like to show?

if readlum:
    with h5py.File(lumpath,'r') as f:
        lum = f[fil][:]

# grab relevant data from files
with h5py.File(datapath+cursim+'_allhalostardata.h5','r') as f:
    hostids = f['host_IDs'].asstr()[:]
    partids = f['particle_IDs'][:]
with h5py.File(opath+cursim+'_circ.h5','r') as f:
    circ = f[circtype][:]
    norm_L = f['angmom_vec'][:]
uIDs = np.unique(hostids)

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
if not np.array_equal(partids,allstars['iord']):
    allstarinds = allstars['iord']
    index = np.argsort(allstarinds)
    sorted_allstars = allstarinds[index]
    sorted_index = np.searchsorted(sorted_allstars,partids)
    pindex = np.take(index,sorted_index,mode="clip")
    mask = allstarinds[pindex] != partids
    res = np.ma.array(pindex,mask=mask)
    allstars = allstars[np.ma.compressed(res)]

if not readlum:
    mag = allstars[fil+'_mag']
    norm = pynbody.analysis.luminosity.get_current_ssp_table().get_spectral_density_normalization(fil)
    lum = 10**(-1*mag/2.5)*norm 

mets = allstars['FeMassFrac']/FeSol
met2 = allstars['OxMassFrac']/allstars['FeMassFrac']
pos = allstars['pos']

# for each remnant with more than nstarlim star particles, make tidal debris images
for i in uIDs:
    relparts = partids[hostids==i] # grab the star particles that belong to this host
    if len(relparts)>nstarlim:
        xypts = np.zeros((len(lum),2))
        starmask = np.isin(allstars['iord'],relparts)
        xvals = pos[starmask][:,0]
        yvals = pos[starmask][:,1]
        zvals = pos[starmask][:,2]
        xwid = max([abs(x) for x in xvals])
        ywid = max([abs(x) for x in yvals])
        zwid = max([abs(x) for x in zvals])
        wid = 2*min(max([xwid,ywid,zwid]),maxwid)

        # Make your tick marks at least somewhat reasonable
        if wid<5:
            st = 1
        elif wid<20:
            st = 3
        elif wid<60:
            st = 10
        elif wid<120:
            st = 20
        elif wid<200:
            st = 40
        elif wid<300:
            st = 50
        else:
            st = 150
        tlist = np.arange(-1*math.floor(wid/2./st)*st,math.floor(wid/2./st)*st+1,st)

        df = pd.DataFrame({})
        df['x'] = xvals
        df['y'] = yvals
        df['z'] = zvals
        df['lum_pd'] = lum[starmask]
        df['met_pd'] = np.log10(mets[starmask])
        df['alpha_pd'] = np.log10(met2[starmask])
        df['circ_pd'] = circ[starmask]

        # Make a figure that shows luminosity and circularity
        f3 = plt.figure(figsize=(10,6))
        outer_grid = gridspec.GridSpec(1, 2, wspace=0.1, hspace=0.0, width_ratios=[20,1])
        inner_grid = gridspec.GridSpecFromSubplotSpec(2,3,subplot_spec=outer_grid[0], wspace=0.15, hspace=0.15)

        for ctr in range(0,3):
            ax1 = plt.Subplot(f3,inner_grid[0,ctr])
            ax1.set_facecolor('black')
            if ctr == 0:
                artist00 = dsshow(df, dsh.Point("x", "y"), dsh.sum('lum_pd'), aspect='equal', norm='log',vmin=lmin, vmax=lmax,cmap='Greys_r',x_range=(-1*(wid/2.),(wid/2.)), y_range=(-1*(wid/2.),(wid/2.)),ax=ax1)
                ax1.set_yticks(tlist)
                ax1.set_xticks(tlist)
                ax1.set_title('z',fontsize=fsize+2)
                ax1.set_ylabel('[kpc]',fontsize=fsize)
            elif ctr == 1:
                artist01 = dsshow(df, dsh.Point("y", "z"), dsh.sum('lum_pd'), aspect='equal', norm='log',vmin=lmin, vmax=lmax,cmap='Greys_r',x_range=(-1*(wid/2.),(wid/2.)), y_range=(-1*(wid/2.),(wid/2.)),ax=ax1)
                ax1.set_yticks([])
                ax1.set_xticks(tlist)
                ax1.set_title('x',fontsize=fsize+2)
                ax1.set_xlabel('[kpc]',fontsize=fsize)
            else:
                artist02 = dsshow(df, dsh.Point("x", "z"), dsh.sum('lum_pd'), aspect='equal', norm='log',vmin=lmin, vmax=lmax,cmap='Greys_r',x_range=(-1*(wid/2.),(wid/2.)), y_range=(-1*(wid/2.),(wid/2.)),ax=ax1)
                ax1.set_yticks([])
                ax1.set_title('y',fontsize=fsize+2)
                ax1.set_xticks(tlist)
            f3.add_subplot(ax1)

        cbar_grid = gridspec.GridSpecFromSubplotSpec(2,1,subplot_spec=outer_grid[1], wspace=0.0, hspace=0.08)
        ax2 = plt.Subplot(f3,cbar_grid[0])
        cmap = plt.cm.Greys_r
        norm = mpl.colors.Normalize(vmin=np.log10(lmin), vmax=np.log10(lmax))
        cb1 = mpl.colorbar.ColorbarBase(ax2,cmap=cmap,norm=norm,ticks=np.arange(math.floor(np.log10(lmin)),math.ceil(np.log10(lmax))+1.,1),orientation='vertical',label='log$_{10}$(L/erg s$^{-1}$)')
        f3.add_subplot(ax2)

        for ctr in range(0,3):
            ax1 = plt.Subplot(f3,inner_grid[1,ctr])
            ax1.set_facecolor('lightgrey')
            if ctr == 0:
                artist00 = dsshow(df, dsh.Point("x", "y"), dsh.mean('circ_pd'), aspect='equal', norm='linear',vmin=cmin, vmax=cmax,cmap='seismic_r',x_range=(-1*(wid/2.),(wid/2.)), y_range=(-1*(wid/2.),(wid/2.)),ax=ax1)
                ax1.set_yticks(tlist)
                ax1.set_xticks(tlist)
                ax1.set_title('z',fontsize=fsize+2)
                ax1.set_ylabel('[kpc]',fontsize=fsize)
            elif ctr == 1:
                artist01 = dsshow(df, dsh.Point("y", "z"), dsh.mean('circ_pd'), aspect='equal', norm='linear',vmin=cmin, vmax=cmax,cmap='seismic_r',x_range=(-1*(wid/2.),(wid/2.)), y_range=(-1*(wid/2.),(wid/2.)),ax=ax1)
                ax1.set_yticks([])
                ax1.set_xticks(tlist)
                ax1.set_title('x',fontsize=fsize+2)
                ax1.set_xlabel('[kpc]',fontsize=fsize)
            else:
                artist02 = dsshow(df, dsh.Point("x", "z"), dsh.mean('circ_pd'), aspect='equal', norm='linear',vmin=cmin, vmax=cmax,cmap='seismic_r',x_range=(-1*(wid/2.),(wid/2.)), y_range=(-1*(wid/2.),(wid/2.)),ax=ax1)
                ax1.set_yticks([])
                ax1.set_title('y',fontsize=fsize+2)
                ax1.set_xticks(tlist)
            f3.add_subplot(ax1)

        cbar_grid = gridspec.GridSpecFromSubplotSpec(2,1,subplot_spec=outer_grid[1], wspace=0.0, hspace=0.08)
        ax2 = plt.Subplot(f3,cbar_grid[1])
        cmap = plt.cm.seismic_r
        norm = mpl.colors.Normalize(vmin=cmin, vmax=cmax)
        cb1 = mpl.colorbar.ColorbarBase(ax2,cmap=cmap,norm=norm,ticks=np.arange(cmin,cmax+0.01,0.2),orientation='vertical',label='Circularity')
        f3.add_subplot(ax2)

        plt.savefig(opath+cursim+'_'+i+'_lumcirc.png',bbox_inches='tight',dpi=150)

        # Make a figure that shows [Fe/H] and [O/Fe]
        f3 = plt.figure(figsize=(10,6))
        outer_grid = gridspec.GridSpec(1, 2, wspace=0.1, hspace=0.0, width_ratios=[20,1])
        inner_grid = gridspec.GridSpecFromSubplotSpec(2,3,subplot_spec=outer_grid[0], wspace=0.15, hspace=0.15)

        for ctr in range(0,3):
            ax1 = plt.Subplot(f3,inner_grid[0,ctr])
            ax1.set_facecolor('lightgrey')
            if ctr == 0:
                artist00 = dsshow(df, dsh.Point("x", "y"), dsh.mean('met_pd'), aspect='equal', norm='linear',vmin=mmin, vmax=mmax,cmap='viridis',x_range=(-1*(wid/2.),(wid/2.)), y_range=(-1*(wid/2.),(wid/2.)),ax=ax1)
                ax1.set_yticks(tlist)
                ax1.set_xticks(tlist)
                ax1.set_title('z',fontsize=fsize+2)
                ax1.set_ylabel('[kpc]',fontsize=fsize)
            elif ctr == 1:
                artist01 = dsshow(df, dsh.Point("y", "z"), dsh.mean('met_pd'), aspect='equal', norm='linear',vmin=mmin, vmax=mmax,cmap='viridis',x_range=(-1*(wid/2.),(wid/2.)), y_range=(-1*(wid/2.),(wid/2.)),ax=ax1)
                ax1.set_yticks([])
                ax1.set_xticks(tlist)
                ax1.set_title('x',fontsize=fsize+2)
                ax1.set_xlabel('[kpc]',fontsize=fsize)
            else:
                artist02 = dsshow(df, dsh.Point("x", "z"), dsh.mean('met_pd'), aspect='equal', norm='linear',vmin=mmin, vmax=mmax,cmap='viridis',x_range=(-1*(wid/2.),(wid/2.)), y_range=(-1*(wid/2.),(wid/2.)),ax=ax1)
                ax1.set_yticks([])
                ax1.set_title('y',fontsize=fsize+2)
                ax1.set_xticks(tlist)
            f3.add_subplot(ax1)

        cbar_grid = gridspec.GridSpecFromSubplotSpec(2,1,subplot_spec=outer_grid[1], wspace=0.0, hspace=0.08)
        ax2 = plt.Subplot(f3,cbar_grid[0])
        cmap = plt.cm.viridis
        norm = mpl.colors.Normalize(vmin=mmin, vmax=mmax)
        cb1 = mpl.colorbar.ColorbarBase(ax2,cmap=cmap,norm=norm,ticks=np.arange(mmin,mmax+0.1,1),orientation='vertical',label='[Fe/H]')
        f3.add_subplot(ax2)

        for ctr in range(0,3):
            ax1 = plt.Subplot(f3,inner_grid[1,ctr])
            ax1.set_facecolor('lightgrey')
            if ctr == 0:
                artist00 = dsshow(df, dsh.Point("x", "y"), dsh.mean('alpha_pd'), aspect='equal', norm='linear',vmin=amin, vmax=amax,cmap='viridis',x_range=(-1*(wid/2.),(wid/2.)), y_range=(-1*(wid/2.),(wid/2.)),ax=ax1)
                ax1.set_yticks(tlist)
                ax1.set_xticks(tlist)
                ax1.set_title('z',fontsize=fsize+2)
                ax1.set_ylabel('[kpc]',fontsize=fsize)
            elif ctr == 1:
                artist01 = dsshow(df, dsh.Point("y", "z"), dsh.mean('alpha_pd'), aspect='equal', norm='linear',vmin=amin, vmax=amax,cmap='viridis',x_range=(-1*(wid/2.),(wid/2.)), y_range=(-1*(wid/2.),(wid/2.)),ax=ax1)
                ax1.set_yticks([])
                ax1.set_xticks(tlist)
                ax1.set_title('x',fontsize=fsize+2)
                ax1.set_xlabel('[kpc]',fontsize=fsize)
            else:
                artist02 = dsshow(df, dsh.Point("x", "z"), dsh.mean('alpha_pd'), aspect='equal', norm='linear',vmin=amin, vmax=amax,cmap='viridis',x_range=(-1*(wid/2.),(wid/2.)), y_range=(-1*(wid/2.),(wid/2.)),ax=ax1)
                ax1.set_yticks([])
                ax1.set_title('y',fontsize=fsize+2)
                ax1.set_xticks(tlist)
            f3.add_subplot(ax1)

        cbar_grid = gridspec.GridSpecFromSubplotSpec(2,1,subplot_spec=outer_grid[1], wspace=0.0, hspace=0.08)
        ax2 = plt.Subplot(f3,cbar_grid[1])
        cmap = plt.cm.viridis
        norm = mpl.colors.Normalize(vmin=amin, vmax=amax)
        cb1 = mpl.colorbar.ColorbarBase(ax2,cmap=cmap,norm=norm,ticks=np.arange(amin,amax+0.01,1),orientation='vertical',label='[O/Fe]')
        f3.add_subplot(ax2)

        plt.savefig(opath+cursim+'_'+i+'_fealpha.png',bbox_inches='tight',dpi=150)

# Now make these same plots for all in situ material and ex situ material
isAll = np.where(hostids==ishalo)
exAll = np.where(hostids!=ishalo)
isdf = pd.DataFrame({})
isdf['x'] = pos[isAll][:,0]
isdf['y'] = pos[isAll][:,1]
isdf['z'] = pos[isAll][:,2]
isdf['lum_pd'] = lum[isAll]
isdf['alpha_pd'] = np.log10(met2[isAll])
isdf['met_pd'] = np.log10(mets[isAll])

exdf = pd.DataFrame({})
exdf['x'] = pos[exAll][:,0]
exdf['y'] = pos[exAll][:,1]
exdf['z'] = pos[exAll][:,2]
exdf['lum_pd'] = lum[exAll]
exdf['alpha_pd'] = np.log10(met2[exAll])
exdf['met_pd'] = np.log10(mets[exAll])

wid = 2*min(maxwid,np.linalg.norm(exdf['x'],exdf['y'],exdf['z']))

# Make your tick marks at least somewhat reasonable
if wid<5:
    st = 1
elif wid<20:
    st = 3
elif wid<60:
    st = 10
elif wid<120:
    st = 20
elif wid<200:
    st = 40
elif wid<300:
    st = 50
else:
    st = 150
tlist = np.arange(-1*math.floor(wid/2./st)*st,math.floor(wid/2./st)*st+1,st)

# start with luminosity
f3 = plt.figure(figsize=(8,5))
outer_grid = gridspec.GridSpec(1, 2, wspace=0.1, hspace=0.0, width_ratios=[20,1])
inner_grid = gridspec.GridSpecFromSubplotSpec(2,2,subplot_spec=outer_grid[0], wspace=0.15, hspace=0.15)

ax1 = plt.Subplot(f3,inner_grid[0,0])
ax1.set_facecolor('black')
artist00 = dsshow(isdf, dsh.Point("x", "y"), dsh.sum('lum_pd'), aspect='equal', norm='log',vmin=lmin, vmax=lmax,cmap='Greys_r',x_range=(-1*(wid/2.),(wid/2.)), y_range=(-1*(wid/2.),(wid/2.)),ax=ax1)
ax1.set_title('In Situ',fontsize=fsize+2)
ax1.set_ylabel('[kpc]',fontsize=fsize)
f3.add_subplot(ax1)

ax1 = plt.Subplot(f3,inner_grid[0,1])
ax1.set_facecolor('black')
artist01 = dsshow(exdf, dsh.Point("x", "y"), dsh.sum('lum_pd'), aspect='equal', norm='log',vmin=lmin, vmax=lmax,cmap='Greys_r',x_range=(-1*(wid/2.),(wid/2.)), y_range=(-1*(wid/2.),(wid/2.)),ax=ax1)
ax1.set_title('Ex Situ',fontsize=fsize+2)
ax1.set_xlabel('[kpc]',fontsize=fsize)
f3.add_subplot(ax1)
    
cbar_grid = gridspec.GridSpecFromSubplotSpec(1,1,subplot_spec=outer_grid[1], wspace=0.0, hspace=0.08)
ax2 = plt.Subplot(f3,cbar_grid[0])
cmap = plt.cm.Greys_r
norm = mpl.colors.Normalize(vmin=np.log10(lmin), vmax=np.log10(lmax))
cb1 = mpl.colorbar.ColorbarBase(ax2,cmap=cmap,norm=norm,ticks=np.arange(math.floor(np.log10(lmin)),math.ceil(np.log10(lmax))+1.,0.5),orientation='vertical')
cb1.set_label(label='log$_{10}$(L/erg s$^{-1}$)',size=fsize)
f3.add_subplot(ax2)

ax1 = plt.Subplot(f3,inner_grid[1,0])
ax1.set_facecolor('black')
artist00 = dsshow(isdf, dsh.Point("x", "z"), dsh.sum('lum_pd'), aspect='equal', norm='log',vmin=lmin, vmax=lmax,cmap='Greys_r',x_range=(-1*(wid/2.),(wid/2.)), y_range=(-1*(wid/2.),(wid/2.)),ax=ax1)
ax1.set_ylabel('[kpc]',fontsize=fsize)
ax1.set_xlabel('[kpc]',fontsize=fsize)
f3.add_subplot(ax1)

ax1 = plt.Subplot(f3,inner_grid[0,1])
ax1.set_facecolor('black')
artist01 = dsshow(exdf, dsh.Point("x", "z"), dsh.sum('lum_pd'), aspect='equal', norm='log',vmin=lmin, vmax=lmax,cmap='Greys_r',x_range=(-1*(wid/2.),(wid/2.)), y_range=(-1*(wid/2.),(wid/2.)),ax=ax1)
ax1.set_title('Ex Situ',fontsize=fsize+2)
ax1.set_xlabel('[kpc]',fontsize=fsize)
f3.add_subplot(ax1)

plt.savefig(opath+cursim+'_'+'IsExAll_lum.png',bbox_inches='tight',dpi=200)

# circularity
f3 = plt.figure(figsize=(8,5))
outer_grid = gridspec.GridSpec(1, 2, wspace=0.1, hspace=0.0, width_ratios=[20,1])
inner_grid = gridspec.GridSpecFromSubplotSpec(2,2,subplot_spec=outer_grid[0], wspace=0.15, hspace=0.15)

ax1 = plt.Subplot(f3,inner_grid[0,0])
ax1.set_facecolor('lightgrey')
artist00 = dsshow(isdf, dsh.Point("x", "y"), dsh.mean('circ_pd'), aspect='equal', norm='linear',vmin=cmin, vmax=cmax,cmap='seismic_r',x_range=(-1*(wid/2.),(wid/2.)), y_range=(-1*(wid/2.),(wid/2.)),ax=ax1)
ax1.set_title('In Situ',fontsize=fsize+2)
ax1.set_ylabel('[kpc]',fontsize=fsize)
f3.add_subplot(ax1)

ax1 = plt.Subplot(f3,inner_grid[0,1])
ax1.set_facecolor('lightgrey')
artist01 = dsshow(exdf, dsh.Point("x", "y"), dsh.mean('circ_pd'), aspect='equal', norm='linear',vmin=cmin, vmax=cmax,cmap='seismic_r',x_range=(-1*(wid/2.),(wid/2.)), y_range=(-1*(wid/2.),(wid/2.)),ax=ax1)
ax1.set_title('Ex Situ',fontsize=fsize+2)
ax1.set_xlabel('[kpc]',fontsize=fsize)
f3.add_subplot(ax1)
    
cbar_grid = gridspec.GridSpecFromSubplotSpec(1,1,subplot_spec=outer_grid[1], wspace=0.0, hspace=0.08)
ax2 = plt.Subplot(f3,cbar_grid[0])
cmap = plt.cm.seismic_r
norm = mpl.colors.Normalize(vmin=cmin, vmax=cmax)
cb1 = mpl.colorbar.ColorbarBase(ax2,cmap=cmap,norm=norm,ticks=np.arange(cmin,cmax+0.01,0.1),orientation='vertical')
cb1.set_label(label='Circularity',size=fsize)
f3.add_subplot(ax2)

ax1 = plt.Subplot(f3,inner_grid[1,0])
ax1.set_facecolor('lightgrey')
artist00 = dsshow(isdf, dsh.Point("x", "z"), dsh.mean('circ_pd'), aspect='equal', norm='linear',vmin=cmin, vmax=cmax,cmap='seismic_r',x_range=(-1*(wid/2.),(wid/2.)), y_range=(-1*(wid/2.),(wid/2.)),ax=ax1)
ax1.set_ylabel('[kpc]',fontsize=fsize)
ax1.set_xlabel('[kpc]',fontsize=fsize)
f3.add_subplot(ax1)

ax1 = plt.Subplot(f3,inner_grid[0,1])
ax1.set_facecolor('lightgrey')
artist01 = dsshow(exdf, dsh.Point("x", "z"), dsh.mean('circ_pd'), aspect='equal', norm='linear',vmin=cmin, vmax=cmax,cmap='seismic_r',x_range=(-1*(wid/2.),(wid/2.)), y_range=(-1*(wid/2.),(wid/2.)),ax=ax1)
ax1.set_title('Ex Situ',fontsize=fsize+2)
ax1.set_xlabel('[kpc]',fontsize=fsize)
f3.add_subplot(ax1)

plt.savefig(opath+cursim+'_'+'IsExAll_circ.png',bbox_inches='tight',dpi=200)

# [Fe/H]
f3 = plt.figure(figsize=(8,5))
outer_grid = gridspec.GridSpec(1, 2, wspace=0.1, hspace=0.0, width_ratios=[20,1])
inner_grid = gridspec.GridSpecFromSubplotSpec(2,2,subplot_spec=outer_grid[0], wspace=0.15, hspace=0.15)

ax1 = plt.Subplot(f3,inner_grid[0,0])
ax1.set_facecolor('lightgrey')
artist00 = dsshow(isdf, dsh.Point("x", "y"), dsh.mean('met_pd'), aspect='equal', norm='linear',vmin=mmin, vmax=mmax,cmap='viridis',x_range=(-1*(wid/2.),(wid/2.)), y_range=(-1*(wid/2.),(wid/2.)),ax=ax1)
ax1.set_title('In Situ',fontsize=fsize+2)
ax1.set_ylabel('[kpc]',fontsize=fsize)
f3.add_subplot(ax1)

ax1 = plt.Subplot(f3,inner_grid[0,1])
ax1.set_facecolor('lightgrey')
artist01 = dsshow(exdf, dsh.Point("x", "y"), dsh.mean('met_pd'), aspect='equal', norm='linear',vmin=mmin, vmax=mmax,cmap='viridis',x_range=(-1*(wid/2.),(wid/2.)), y_range=(-1*(wid/2.),(wid/2.)),ax=ax1)
ax1.set_title('Ex Situ',fontsize=fsize+2)
ax1.set_xlabel('[kpc]',fontsize=fsize)
f3.add_subplot(ax1)
    
cbar_grid = gridspec.GridSpecFromSubplotSpec(1,1,subplot_spec=outer_grid[1], wspace=0.0, hspace=0.08)
ax2 = plt.Subplot(f3,cbar_grid[0])
cmap = plt.cm.viridis
norm = mpl.colors.Normalize(vmin=mmin, vmax=mmax)
cb1 = mpl.colorbar.ColorbarBase(ax2,cmap=cmap,norm=norm,ticks=np.arange(mmin,mmax+0.01,0.5),orientation='vertical')
cb1.set_label(label='[Fe/H]',size=fsize)
f3.add_subplot(ax2)

ax1 = plt.Subplot(f3,inner_grid[1,0])
ax1.set_facecolor('lightgrey')
artist00 = dsshow(isdf, dsh.Point("x", "z"), dsh.mean('met_pd'), aspect='equal', norm='linear',vmin=mmin, vmax=mmax,cmap='viridis',x_range=(-1*(wid/2.),(wid/2.)), y_range=(-1*(wid/2.),(wid/2.)),ax=ax1)
ax1.set_ylabel('[kpc]',fontsize=fsize)
ax1.set_xlabel('[kpc]',fontsize=fsize)
f3.add_subplot(ax1)

ax1 = plt.Subplot(f3,inner_grid[0,1])
ax1.set_facecolor('lightgrey')
artist01 = dsshow(exdf, dsh.Point("x", "z"), dsh.mean('met_pd'), aspect='equal', norm='linear',vmin=mmin, vmax=mmax,cmap='viridis',x_range=(-1*(wid/2.),(wid/2.)), y_range=(-1*(wid/2.),(wid/2.)),ax=ax1)
ax1.set_title('Ex Situ',fontsize=fsize+2)
ax1.set_xlabel('[kpc]',fontsize=fsize)
f3.add_subplot(ax1)

plt.savefig(opath+cursim+'_'+'IsExAll_FeH.png',bbox_inches='tight',dpi=200)

# [O/Fe]
f3 = plt.figure(figsize=(8,5))
outer_grid = gridspec.GridSpec(1, 2, wspace=0.1, hspace=0.0, width_ratios=[20,1])
inner_grid = gridspec.GridSpecFromSubplotSpec(2,2,subplot_spec=outer_grid[0], wspace=0.15, hspace=0.15)

ax1 = plt.Subplot(f3,inner_grid[0,0])
ax1.set_facecolor('lightgrey')
artist00 = dsshow(isdf, dsh.Point("x", "y"), dsh.mean('alpha_pd'), aspect='equal', norm='linear',vmin=amin, vmax=amax,cmap='viridis',x_range=(-1*(wid/2.),(wid/2.)), y_range=(-1*(wid/2.),(wid/2.)),ax=ax1)
ax1.set_title('In Situ',fontsize=fsize+2)
ax1.set_ylabel('[kpc]',fontsize=fsize)
f3.add_subplot(ax1)

ax1 = plt.Subplot(f3,inner_grid[0,1])
ax1.set_facecolor('lightgrey')
artist01 = dsshow(exdf, dsh.Point("x", "y"), dsh.mean('alpha_pd'), aspect='equal', norm='linear',vmin=amin, vmax=amax,cmap='viridis',x_range=(-1*(wid/2.),(wid/2.)), y_range=(-1*(wid/2.),(wid/2.)),ax=ax1)
ax1.set_title('Ex Situ',fontsize=fsize+2)
ax1.set_xlabel('[kpc]',fontsize=fsize)
f3.add_subplot(ax1)
    
cbar_grid = gridspec.GridSpecFromSubplotSpec(1,1,subplot_spec=outer_grid[1], wspace=0.0, hspace=0.08)
ax2 = plt.Subplot(f3,cbar_grid[0])
cmap = plt.cm.viridis
norm = mpl.colors.Normalize(vmin=amin, vmax=amax)
cb1 = mpl.colorbar.ColorbarBase(ax2,cmap=cmap,norm=norm,ticks=np.arange(amin,amax+0.01,0.5),orientation='vertical')
cb1.set_label(label='[O/Fe]',size=fsize)
f3.add_subplot(ax2)

ax1 = plt.Subplot(f3,inner_grid[1,0])
ax1.set_facecolor('lightgrey')
artist00 = dsshow(isdf, dsh.Point("x", "z"), dsh.mean('alpha_pd'), aspect='equal', norm='linear',vmin=amin, vmax=amax,cmap='viridis',x_range=(-1*(wid/2.),(wid/2.)), y_range=(-1*(wid/2.),(wid/2.)),ax=ax1)
ax1.set_ylabel('[kpc]',fontsize=fsize)
ax1.set_xlabel('[kpc]',fontsize=fsize)
f3.add_subplot(ax1)

ax1 = plt.Subplot(f3,inner_grid[0,1])
ax1.set_facecolor('lightgrey')
artist01 = dsshow(exdf, dsh.Point("x", "z"), dsh.mean('alpha_pd'), aspect='equal', norm='linear',vmin=amin, vmax=amax,cmap='viridis',x_range=(-1*(wid/2.),(wid/2.)), y_range=(-1*(wid/2.),(wid/2.)),ax=ax1)
ax1.set_title('Ex Situ',fontsize=fsize+2)
ax1.set_xlabel('[kpc]',fontsize=fsize)
f3.add_subplot(ax1)

plt.savefig(opath+cursim+'_'+'IsExAll_OFe.png',bbox_inches='tight',dpi=200)