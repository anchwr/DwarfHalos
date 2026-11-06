'''
Script to link together pairs of halos in a tangos database
If the halos are not in adjacent snapshots, a chain of phantoms
will be created in the intervening snapshots to link them. These
phantoms will have negative finder_id values, starting with -1 in
a given snapshot. Note that, with the current implementation,
new links will take precedence over existing ones.

Usage: python LinkMaker_rz.py <inputfile> <opt:SanityCheck>
Example: python LinkMaker_rz.py RomZoomLinks.txt False

Output: None (prints log to screen and modifies active db in place)

Takes up to two arguments. The first should be a file containing
the lists of halo pairs. The format of this file should be as follows:

Sim	Halo1		Halo2
431	1824_2		2016_2
492	0223_13		0345_30
492	0273_5		0345_8

where the first column is the number of the simulation (e.g., 431) and
the second and the third columns should be the first halo to be linked
(chronologically) and the second halo to be linked, respectively. The halo
IDs are formatted following the convention used in the stellar halos pipeline
i.e., SnapshotName_TangosIndex. So, for instance, 1824_2 would be the second halo
in the tangos database at snapshot 001824.
The second argument is a boolean (True/False) indicating whether you want the script to 
print out the fraction of DM particles that the two halos have in common and then
ask you before linking them. The default is to perform the sanity check, but if you
pass 'False', the script will link all halo pairs listed in your file without 
requiring any further input from the user.
'''

import tangos as db
import numpy as np
from tangos.core.halo import PhantomHalo
import sys
import math
import pynbody

simpath = '/data/REPOSITORY/romulus_zooms/' # where do your simulations live? Only relevant if SanityCheck=True
spec = '.romulus25.3072g1HsbBH' # what's the rest of the rootname for the simulation?
dbkey = ''  # Is there a unique identifier for this simulation in your tangos db? If you only have one sim
            # that starts with cursim, you can set dbkey=''
pform = 1 # If pform=1, script will assume that snapshots are located inside of
            # snapshot folders. If pform=2, it will assume that snapshots are
            # all located on the same level.
AHF = False # Did you construct your tangos db with AHF (rather than amiga)?
SanityCheck = True # Default value (can pass false to CL). Do you want script to print out fraction of DM particles
                   # halos have in common and doublecheck with you before linking them?
wgt = 1 # default linking weight

def str2bool(torf):
    if torf == 'True' or torf == 'true':
        resp = True
    else:
        resp = False
    return resp

if len(sys.argv) < 2:
    print ('Usage: python LinkMaker_rz.py <inputfile> <opt:SanityCheck>')
    sys.exit()
else:
    if len(sys.argv) <= 3:
        ifile = str(sys.argv[1])
    else:
        print ('Usage: python LinkMaker_rz.py <inputfile> <opt:SanityCheck>')
        sys.exit()
    if len(sys.argv) == 3:
        SanityCheck = str2bool(sys.argv[2])

def snapshot_to_db_index(sim,snap):
    '''
    Calculates the tangos database index of snapshot <snap>
    Returns the index if that snap is in the database. Otherwise,
    prints an error and exits
    '''
    dbind = -1
    for ctr,ts in enumerate(sim.timesteps):
        if str(snap).zfill(6) in str(ts):
            dbind = ctr
    if dbind>=0:
        return dbind
    else:
        print ('Snapshot '+str(snap)+' does not seem to exist in the current db.')
        exit()

def round_to_n(val,n):
    '''
    Round val to n significant figures
    '''
    if val == 0:
        return 0.0
    else:
        return round(val,int(n-math.ceil(math.log10(abs(val)))))


def HaloComparison(sim,halo1,halo2):
    '''
    Calculate and return the fraction of <halo2>'s DM particles that are 
    also in <halo1> and the fraction of <halo1>'s DM particles that are 
    also in <halo2> in <sim>.
    '''
    snnum = int(halo1.split('_')[0])
    hnum = int(halo1.split('_')[1])
    snind = int(snapshot_to_db_index(sim,snnum))
    fid = sim[snind][hnum].finder_id # grab amiga.grp ID
    if pform == 1:
        simloc = simpath+'r'+str(cursim)+spec+'/'+sim[snind].extension
    else:
        simloc = simpath+sim[snind].extension
    s = pynbody.load(simloc)
    h = s.halos()
    if AHF and isinstance(h,pynbody.halo.number_array.AmigaGrpCatalogue):
        fid += 1
    ptcls1 = h[fid].dm['iord'] # grab DM iords

    snnum = int(halo2.split('_')[0])
    hnum = int(halo2.split('_')[1])
    snind = int(snapshot_to_db_index(sim,snnum))
    if pform == 1:
        simloc = simpath+'r'+str(cursim)+spec+'/'+sim[snind].extension
    else:
        simloc = simpath+sim[snind].extension
    fid = sim[snind][hnum].finder_id # grab amiga.grp ID
    s = pynbody.load(simloc)
    if AHF and isinstance(h,pynbody.halo.number_array.AmigaGrpCatalogue):
        fid += 1
    h = s.halos()
    ptcls2 = h[fid].dm['iord'] # grab DM iords

    com1 = len(ptcls1[np.isin(ptcls1,ptcls2)])/len(ptcls1)
    com2 = len(ptcls2[np.isin(ptcls2,ptcls1)])/len(ptcls2)

    return com1,com2

# Read in input file. Option for third column to indicate whether 
# new links should take precedence over existing links is coded but
# currently unused
cs = []
h1arr = []
h2arr = []
obe = []
rf = open(ifile,'r')
rfl = rf.readline()
rfl = rf.readline()
while rfl != '':
    rs = str(rfl).split()
    cs.append(int(rs[0]))
    h1arr.append(rs[1])
    h2arr.append(rs[2])
    if len(rs)>3:
        obe.append(True)
    else:
        obe.append(False)
    rfl = rf.readline()
rf.close()

for h1,h2,cursim,ob in zip(h1arr,h2arr,cs,obe): # for each pair of halos
    sim = db.get_simulation('%r'+str(cursim)+'%'+dbkey+'%')
    curi = h1.split('_')[0]
    curh = h1.split('_')[1]
    curt = snapshot_to_db_index(sim,curi)
    futi = h2.split('_')[0]
    futh = h2.split('_')[1]
    futt = snapshot_to_db_index(sim,futi)

    if SanityCheck: # if we want the sanity check, calculate shared DM fractions and check with user
        com1,com2 = HaloComparison(sim,h1,h2)
        print (h1+' contains '+str(100*round_to_n(com2,4))+'% of '+h2+'\'s particles')
        print (h2+' contains '+str(100*round_to_n(com1,4))+'% of '+h1+'\'s particles')
        dolink = input('Link them? y/n ')
    else:
        dolink = 'y'
    # if ob:
    #     wgt = 1
    # else:
    #     wgt = 1
    if dolink == 'y': # if we want to link
        myhalo = sim[int(curt)][int(curh)]
        if futt <= curt: # check halos are actually in the right order
            raise ValueError(h1+' comes after or at the same time as '+h2+'. Please re-arrange and try again.')
        elif futt-curt > 1:
            curt += 1
            while int(futt-curt) > 0: # if your halos aren't in adjacent steps, iteratively create phantoms between them 
                exph = sim[int(curt)].phantoms.all() # grab a list of all phantoms in this snapshot
                phnum = len(exph)+1 # how many phantoms are there?
                print('Adding Phantom '+str(phnum)+' to step '+str(curt)+' of r'+str(cursim))
                # Generate a finder ID for your phantom that's different from that of other phantoms
                phids = np.array([p.finder_id for p in exph])
                if len(phids[phids<0])>0:
                    phid = int(min(phids[phids<0])-1) # if there are other phantoms, just subtract 1 from the minimum
                else:
                    phid = -1 # if there are no phantoms, start with -1
                # create your phantom and link it to the relevant object in the previous snapshot
                phantom = PhantomHalo(sim[int(curt)],phnum,phid)
                db.core.get_default_session().add(phantom)
                link1 = db.core.HaloLink(myhalo, phantom, db.core.get_or_create_dictionary_item(db.core.get_default_session(), 'manual_link'),weight=wgt)
                link2 = db.core.HaloLink(phantom, myhalo, db.core.get_or_create_dictionary_item(db.core.get_default_session(), 'manual_link'),weight=wgt)
                db.core.get_default_session().add_all([link1, link2])
                db.core.get_default_session().commit()
                myhalo = phantom
                curt += 1 
        # create your final link to <halo2>
        newhalo = sim[int(futt)][int(futh)] 
        link1 = db.core.HaloLink(myhalo, newhalo, db.core.get_or_create_dictionary_item(db.core.get_default_session(), 'manual_link'),weight=wgt)
        link2 = db.core.HaloLink(newhalo, myhalo, db.core.get_or_create_dictionary_item(db.core.get_default_session(), 'manual_link'),weight=wgt)
        db.core.get_default_session().add_all([link1, link2])
        db.core.get_default_session().commit()
        print (h1+' and '+h2+' of r'+str(cursim)+' linked!\n' )
    else:
        print ('Not linking!\n')