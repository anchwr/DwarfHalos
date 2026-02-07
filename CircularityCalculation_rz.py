'''
Step 7 of stellar halo pipeline
Calculate and save the orbital circularity of each star particle

Usage:   python CircularityCalculation_rz.py <sim>
Example: python CircularityCalculation_rz.py r634 

Outputs: plots showing orbital circularity of stars as a function of 
         radius and shaded by stellar ages
         plots showing face-on and edge-on images of gas density and 
         stars for the chosen AM method
         <sim>_allhalostardata_circ.h5, which is a version of 
         the allhalostardata hdf5 file with the circularity of
         each star particle saved

There are several combinations of options you can use for this and 
what works best may depend on the galaxy in question. To calculate
the angular momentum vector, you can either use the AM of the stars 
with ages < <ysage> Myr or the gas with T< <gastemp> K within a radius 
of <disklim>. To calculate circularity, you can either use the method 
from Stinson+10 ('Stinson' or 'Stinson_W24') or the method from Abadi+03 
('Abadi'). You can select as many methods at a time as you want.
'Stinson' will give you pynbody's implementation of Stinson+10, 
while 'Stinson_W24' will give you the simpler version I used in Wright+24.
Ideally, you want a disk to be very clear as a distribution of young 
stars at small radii with a circularity somewhere around 1 so that it's 
easy to later make clear cuts for stellar halo stars (e.g., for well-behaved 
FOGGIE galaxies, I used r<~30 kpc and Stinson circ=0.65-1.35 for the disk, and 
everything else was either bulge (r>5 kpc) or stellar halo (r=5-350 kpc) -
see Fig 1 of Wright+24). However, not all galaxies have nice thin co-planar disks. 
Hopefully the combination of images produced by this script will help you 
figure out whether any of these parameters makes sense for your galaxy. 
Once you're happy with the AM method you've chosen, you can set 
<savecirc> to True to save the circularity of each star particle.
'''

import numpy as np
import matplotlib.pyplot as plt
import pynbody
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import datashader as dsh
import seaborn as sns
import h5py
import sys
from datashader.mpl_ext import dsshow
import tangos as db
from pynbody.analysis import Profile
from pynbody import array,units
import math
from scipy.interpolate import make_interp_spline


fsize = 16

plt.rcParams['xtick.labelsize'] = fsize
plt.rcParams['ytick.labelsize'] = fsize

if len(sys.argv) != 2:
    print ('Usage: python CircularityCalculation.py <sim>')
    sys.exit()
else:
    cursim = str(sys.argv[1])

am_method = 'young_stars' # 'young_stars': use stars with ages < ysage Myr to ID AM vector
                          # 'gas': use gas with T<gastemp K to ID AM vector
halolim = 50 # how far from the center of the galaxy should your plot go?
hid = 1 # What is the amiga.grp ID of the halo we're centering on? Almost always 1 for MMs
disklim = 10 # How far out should we check for young stars or cool gas for AM calc (in kpc)?
ysage = 25 # We will use stars with age<ysage Myr to calculate AM if am_method='young_stars'
gastemp = 1e3 # temperature below which to use gas for AM calc if am_method='gas'
savecirc = True # Are you ready to save the circularity values to your hdf5 file?
circ_method = ['Stinson_W24','Abadi_W26'] # options are ['Stinson','Stinson_W24','Abadi','Abadi_W26'] - note order
machine = 'mogget'

if machine=='mogget':
    opath = '/Users/Anna/Research/Outputs/M33analogs/MM/'+cursim+'/' # Where should outputs be saved?
    datapath = '/Users/Anna/Research/Outputs/M33analogs/MM/'+cursim+'/' # Where does your allhalostardata hdf5 file live?
    simdir = '/Volumes/Abhorsen/Data/RomZooms/' # Where does your simulation live?
elif machine=='emu':
    opath = '/home/awright/dwarf_stellar_halos/'+cursim+'/'
    datapath = '/home/awright/dwarf_stellar_halos/'+cursim+'/'
    simdir = '/data/REPOSITORY/romulus_zooms/'

age_color_map = sns.blend_palette(("black", "#16263B", "#386094", "#4575b4", "#4daf4a","#FFD24D", "darkorange"), as_cmap=True)
grav_const = 4.3*10**-6 # grav const in kpc*km^2/Msol*s^2

keyadd = 'W26'
if ysage != 25 and am_method=='young_stars':
    keyadd += '_ys'+str(ysage)
if disklim != 10:
    keyadd += '_dl'+str(disklim)
if gastemp != 1e3 and am_method=='gas':
    keyadd += '_gt'+str(gastemp)

def Stinson_W24(allstars,stars_r):
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
    return jz/jc

def Abadi_W26(allstars,stars_r,makeplot=False):
    # if makeplot=True, makes and saves a plot of the corrected E-j_z of the star particles
    # and the maximum E-j_z curve based on the midplane potential (cf Abadi+03)

    radii = np.logspace(-1,np.log10(halolim+30),500)

    # Combination of pynbody's midplane_rot_curve and midplane_potential functions
    # However, note we use minimum potential from sampled points, as this will be better for
    # estimating the energy of a circular orbit in the midplane. Run r442 for a good demonstration
    # of this
    rs = [pos for r in radii for pos in [(r, 0, 0), (r/np.sqrt(2),r/np.sqrt(2),0), (0, r, 0), (-r/np.sqrt(2),r/np.sqrt(2),0), (-r, 0, 0), (-r/np.sqrt(2),-r/np.sqrt(2),0), (0, -r, 0), (r/np.sqrt(2),-r/np.sqrt(2),0)]]
    m_by_r, m_by_r2 = pynbody.gravity.direct(h[hid], np.array(rs, dtype=h[hid]['pos'].dtype), eps=s['eps'][0])
    v_out = (m_by_r2.units * h[hid]['pos'].units) ** (1, 2)
    phi_out = units.G * h[hid]['mass'].units / h[hid]['pos'].units
    potential = units.G * m_by_r * h[hid]['mass'].units / h[hid]['pos'].units

    vels = []
    pots = []
    ptmin = []
    ptmax = []
    i = 0
    for r in radii:
        r_acc_r = []
        pot = []
        for pos in [(r, 0, 0), (r/np.sqrt(2),r/np.sqrt(2),0), (0, r, 0), (-r/np.sqrt(2),r/np.sqrt(2),0), (-r, 0, 0), (-r/np.sqrt(2),-r/np.sqrt(2),0), (0, -r, 0), (r/np.sqrt(2),-r/np.sqrt(2),0)]:
            r_acc_r.append(np.dot(-m_by_r2[i, :], pos))
            pot.append(potential[i])
            i = i + 1
        vel2 = np.mean(r_acc_r)
        if vel2 > 0:
            vel = math.sqrt(vel2)
        else:
            vel = 0
        pots.append(np.mean(pot))
        vels.append(vel)
        ptmin.append(np.nanmin(pot))
        ptmax.append(np.nanmax(pot))
    vc = array.SimArray(vels, units=v_out)
    phi = array.SimArray(pots, units=phi_out)
    phimin = array.SimArray(ptmin, units=phi_out)
    phimax = array.SimArray(ptmax, units=phi_out)

    # Calculate circular energy 
    ec = np.array([0.5*(v*np.sqrt(grav_const))**2 + p*grav_const for v,p in zip(vc,phi)])
    ecmin = np.array([0.5*(v*np.sqrt(grav_const))**2 + p*grav_const for v,p in zip(vc,phimax)])
    ecmax = np.array([0.5*(v*np.sqrt(grav_const))**2 + p*grav_const for v,p in zip(vc,phimin)])
    jc = np.array([r*v*np.sqrt(grav_const) for r,v in zip(radii,vc)])
    jcirc = make_interp_spline(ec,jc)
    ecirc = make_interp_spline(jc,ec)
    jcirc_min = make_interp_spline(ecmin,jc)
    jcirc_max = make_interp_spline(ecmax,jc)

    # Correct particle energies to most-bound particle
    offset = min(h[hid].s['te'])-ecirc(0)

    # plot to make sure you're actually picking out the maximum j_z at a given energy
    if makeplot==True:
        plt.figure(figsize=(5,5))
        plt.plot((allstars['te']-offset)/(10**5),allstars['jz']/(10**3),marker='.',color='gray',markersize=2,alpha=0.3,linestyle='none',label='Particle Values')
        plt.plot(ecmin/(10**5),jcirc_min(ecmin)/(10**3),'r--',label='Max midplane $\Phi$')
        plt.plot(ec/(10**5),jcirc(ec)/(10**3),'r-',label='Mean midplane $\Phi$')
        plt.plot(ecmax/(10**5),jcirc_max(ecmax)/(10**3),'r:',label='Min midplane $\Phi$')
        plt.xlabel('E (10$^5$ km$^2}$/s$^2$)', fontsize=15)
        plt.ylabel('J$_z$ (10$^3$ kpc km/s)', fontsize=15)
        # comment the below limits out if you want to see where truly obvious ex situ material sits in this plane
        plt.ylim(min(h[hid].s['jz'])/(10**3),max(h[hid].s['jz'])/(10**3))
        plt.xlim((min(h[hid].s['te']-offset)/(10**5))-0.01,(max(h[hid].s['te']-offset)/(10**5))+0.01)
        plt.legend()
        plt.savefig(opath+cursim+'_Ejz_'+am_method+'_Adadi_W26_'+keyadd+'.png',bbox_inches='tight',dpi=150)
        plt.close()

    circ = [jz/c for jz,c in zip(allstars['jz'],jcirc_max(allstars['te']-offset))]

    return circ

s = pynbody.load(simdir+cursim+'.romulus25.3072g1HsbBH/'+cursim+'.romulus25.3072g1HsbBH.004096/'+cursim+'.romulus25.3072g1HsbBH.004096')
h = s.halos()
s.physical_units()

# get AM vector
pynbody.analysis.halo.center(h[hid]) # this centers both spatially and in terms of velocity; we are moving the entire snapshot
if am_method == 'gas':
    amsp = h[hid].g[pynbody.filt.Sphere(disklim)][pynbody.filt.LowPass('temp',str(gastemp)+' K')]
else:
    amsp = h[hid].s[pynbody.filt.Sphere(disklim)][pynbody.filt.LowPass('age',str(ysage)+' Myr')]
print ('Particles being used to calculate AM vector: ',len(amsp['iord']))
L = pynbody.analysis.angmom.ang_mom_vec(amsp)
norm_L = L/np.sqrt((L**2).sum())
print ('AM vector:',norm_L)

disk_rot_arr = pynbody.analysis.angmom.calc_faceon_matrix(norm_L)
pynbody.transformation.Rotation.rotate(s,disk_rot_arr) # rotate so that our disk is face-on (again moving everything)

allstars = s.s[s.s['tform']>0] # no black holes

circ = []
# Calculate circularity
stars_r = np.linalg.norm(allstars['pos'],axis=1)
if 'Stinson' in circ_method:
    pynbody.analysis.morphology.estimate_jcirc_from_rotation_curve(allstars,particles_per_bin=1000)
    circ.append(allstars['j'][:,2]/allstars['j_circ'])
if 'Stinson_W24' in circ_method:
    jzbyjc = Stinson_W24(allstars,stars_r)
    circ.append(jzbyjc)
if 'Abadi' in circ_method: 
    pynbody.analysis.morphology.estimate_jcirc_from_energy(allstars,particles_per_bin=100,quantile=0.95)
    circ.append(allstars['j'][:,2]/allstars['j_circ'])
if 'Abadi_W26' in circ_method:
    jzbyjc = Abadi_W26(allstars,stars_r,makeplot=True)
    circ.append(jzbyjc)

# Make radius-circularity-age figure
sim = db.get_simulation('%'+cursim+'%')
for ctr,cm in enumerate(circ):
    df = pd.DataFrame({'radius':stars_r, 'circularity':cm, 'age':(sim[-1].time_gyr-allstars['tform'].in_units('Gyr'))/sim[-1].time_gyr})
    fig = plt.figure(figsize=(8,8),dpi=120)

    ax1 = fig.add_axes([0.1, 0.1, 0.85, 0.85])

    artist = dsshow(df,dsh.Point('radius','circularity'),dsh.mean('age'),norm='linear',cmap=age_color_map,x_range=(0,halolim),y_range=(-2,2), vmin=0,vmax=1,aspect='auto',ax=ax1)
    ax1.set_xlim(0,halolim)
    if circ_method[ctr] == 'Stinson' or circ_method[ctr]=='Stinson_W24':
        ax1.set_ylim(-2,2)
    elif circ_method[ctr] == 'Abadi' or circ_method[ctr]=='Abadi_W26':
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
    plt.savefig(opath+cursim+'_circularity_'+am_method+'_'+circ_method[ctr]+keyadd+'.png',bbox_inches='tight')
    plt.close()

# Make images of gas and stars so you can sanity check your choice of AM
ml = -5 # log min gas density in g/cm^2
mh = -1 # log max gas density in g/cm^2
w = 20 # image width in kpc
f3 = plt.figure(figsize=(9,10))
pynbody.plot.stars.render(h[hid].s,dynamic_range=3,width=2*w)
# pynbody.plot.stars.render(h[hid].s,starsize=res,mag_range=(19,27),clear=False,width=40) 
plt.xlabel('kpc',fontsize=20)
plt.ylabel('kpc',fontsize=20)
plt.savefig(opath+cursim+'_stars_faceon_'+am_method+keyadd+'.png',bbox_inches='tight')
plt.close()

f3 = plt.figure(figsize=(9,10))
cmap = plt.cm.cubehelix
norm = mpl.colors.Normalize(vmin=ml, vmax=mh)
pynbody.plot.sph.image(h[hid].g,units="g cm**-2",width = 2*w,cmap='cubehelix',vmin=10**ml,vmax=10**mh,show_cbar=False)
plt.xlabel('kpc',fontsize=20)
plt.ylabel('kpc',fontsize=20)
plt.savefig(opath+cursim+'_gas_faceon_'+am_method+keyadd+'.png',bbox_inches='tight')
plt.close()

h[hid].rotate_x(90)
f3 = plt.figure(figsize=(9,10))
pynbody.plot.stars.render(h[hid].s,dynamic_range=3,width=2*w)
plt.xlabel('kpc',fontsize=20)
plt.ylabel('kpc',fontsize=20)
plt.savefig(opath+cursim+'_stars_sideon_'+am_method+keyadd+'.png',bbox_inches='tight')
plt.close()

f3 = plt.figure(figsize=(9,10))
cmap = plt.cm.cubehelix
norm = mpl.colors.Normalize(vmin=ml, vmax=mh)
pynbody.plot.sph.image(h[hid].g,units="g cm**-2",width = 2*w,cmap='cubehelix',vmin=10**ml,vmax=10**mh,show_cbar=False)
plt.xlabel('kpc',fontsize=20)
plt.ylabel('kpc',fontsize=20)
plt.savefig(opath+cursim+'_gas_sideon_'+am_method+keyadd+'.png',bbox_inches='tight')
plt.close()

# If you're happy with the AM vector you've calculated, 
# write out your data (but make sure to read in the old data first)
if savecirc == True:
    ofile = datapath+cursim+'_allhalostardata_circ.h5'
    print ('Saving circularity to '+ofile)
    with h5py.File(datapath+cursim+'_allhalostardata.h5','r') as f:
        hostids = f['host_IDs'][:]
        partids = f['particle_IDs'][:]
        pct = f['particle_creation_times'][:]
        ph = f['particle_hosts'][:]
        pp = f['particle_positions'][:]
        ts = f['timestep_location'][:]
    
    if np.array_equal(partids,allstars['iord']): # if these are the same, we don't need to do anything else
        with h5py.File(ofile,'w') as f:
            f.create_dataset('particle_IDs', data=partids)
            f.create_dataset('particle_positions', data=pp)
            f.create_dataset('particle_creation_times', data=pct)
            f.create_dataset('timestep_location', data=ts)
            f.create_dataset('particle_hosts', data=ph)
            f.create_dataset('host_IDs', data=hostids, dtype="S10")
            for ctr,cm in enumerate(circ):
                f.create_dataset(circ_method[ctr]+'_circ', data=np.array(cm))
    else: # If they're not, re-order sim data so that the stars are in the same order as the hdf5 file 
        if len(allstars['iord'])!=len(partids):
            print ('WARNING: You have '+str(len(partids))+' stars in your allhalostardata file and '+str(len(allstars['iord']))+' stars in your simulation')
        
        allstarinds = allstars['iord']
        index = np.argsort(allstarinds)
        sorted_allstars = allstarinds[index]
        sorted_index = np.searchsorted(sorted_allstars,partids)
        pindex = np.take(index,sorted_index,mode="clip")
        mask = allstarinds[pindex] != partids
        res = np.ma.array(pindex,mask=mask)

        allstars_inhdf5 = allstars[np.ma.compressed(res)]

        assert(np.array_equal(partids,allstars_inhdf5['iord'])) # these had better be the same

        with h5py.File(ofile,'w') as f:
            f.create_dataset('particle_IDs', data=partids)
            f.create_dataset('particle_positions', data=pp)
            f.create_dataset('particle_creation_times', data=pct)
            f.create_dataset('timestep_location', data=ts)
            f.create_dataset('particle_hosts', data=ph)
            f.create_dataset('host_IDs', data=hostids, dtype="S10")
            for ctr,cm in enumerate(circ):
                f.create_dataset(circ_method[ctr]+'_circ', data=np.array(cm)[np.ma.compressed(res)])

'''
Created on Mar 4, 2024

@author: anna
'''