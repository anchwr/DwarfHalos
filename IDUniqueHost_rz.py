'''
Step 4 of stellar halo pipeline
Creates a unique ID for each host that forms a star. The format of
this ID is <last snapshot where host was IDed>_<index at this snapshot>.
So, if a host was halo 5 at snapshot 3552 and then merged with halo 1 
before the next snapshot, its unique ID will be 3552_5. Stars that form
in its main progenitors will also be associated with this ID. These IDs
are written out to a file with a similar format to <sim>_halostarhosts.txt.

Output: <sim>_uniquehalostarhosts.txt

Usage:   python IDUniqueHost_rz.py <sim>
Example: python IDUniqueHost_rz.py r634 

Note that this is currently set up for MMs, but should be easily adapted 
by e.g., changing the paths or adding a path CL argument. It is also
designed to accommodate the phantoms that rockstar generates when it
temporarily loses track of a halo, which slows it down quite a bit. 
If you're only ever going to be using it with other types of merger
trees, it can be simplified.
'''

import numpy as np
import tangos as db
from astropy.table import Table
import collections
import sys

if len(sys.argv) != 2:
    print ('Usage: python writeouthosts_rz.py <sim>')
    sys.exit()
else:
    cursim = str(sys.argv[1])

sim = db.get_simulation(cursim+'%')

d = collections.defaultdict(list)

hsfile = '/Users/Anna/Research/Outputs/M33Analogs/MM/'+cursim+'/'+cursim+'_halostarhosts.txt'
ofile = '/Users/Anna/Research/Outputs/M33Analogs/MM/'+cursim+'/'+cursim+'_uniquehalostarhosts.txt'

def checkmatch_p(step,halo,hid,disp):
    '''
    Check that <halo> (at snapshot number <step>) is the main progenitor of <hid> <disp> snapshots ago
    Returns True or False 
    '''
    proj = sim[int(step)][int(halo)].calculate_for_progenitors('finder_id()')[0]
    try:
        match = (proj[disp]==hid)
    except:
        match = False
    return match
    
def checkmatch_d(step,halo,hid,disp):
    '''
    Check that <halo> (at snapshot number <step>) is the main descendant of <hid> <disp> snapshots in the future
    Returns True or False 
    '''
    proj = sim[int(step)][int(halo)].calculate_for_descendants('finder_id()')[0]
    try:
        match = (proj[disp]==hid)
    except:
        match = False
    return match

def trackforward(step,halo):
    '''
    Track <halo> (at snapshot number <step>) forward in time
    Returns latest snapshot number and halo number when <halo> was self-consistently identified
    '''
    # Make a list of the descendants of <halo> (at <step>)
    # Can simplify considerably once earlier/later works with phantoms
    desc,fid,phid = sim[int(step)][int(halo)].calculate_for_descendants('halo_number()','finder_id()','type()')

    # Trim phantoms off the end of the list
    eol = phid[-1]
    while eol>0:
        fid = fid[:-1]
        desc = desc[:-1]
        phid = phid[:-1]
        eol = phid[-1]

    # make sure that this list is self-consistent - i.e., the last halo in this list identifies
    # <halo> as its main progenitor
    nd = len(desc)-1
    match = checkmatch_p(step+nd,desc[nd],fid[0],nd)
    stat = int(match)

    # If the final halo in the list doesn't identify <halo> as its main progenitor, identify
    # the last halo in the list that does. This halo provides the unique ID
    last_t = 0
    if stat == 0:
        ma_desc = desc[phid<1] # avoid phantoms
        refarr = np.cumsum(phid<1)
        sf = len(ma_desc)
        s0 = 0
        while (s0 <= sf):
            ci = (s0+sf)//2
            ci_trans = np.argmax(refarr>ci)
            match = checkmatch_p(step+ci_trans,desc[ci_trans],fid[0],ci_trans)
            if match==True:
                s0 = ci+1
                last_t = ci_trans
            else:
                sf = ci-1
    else:
        last_t = nd
    return step+last_t,int(desc[last_t])

# read in your list of hosts that had new stars at each snapshot
# and initialize your list of unique IDs
tslist = []
hostlist = []
nslist = []
unilist = [] # list that will contain strings of unique IDs
rf = open(hsfile,'r')
rlist = rf.readline()
badlist = []
badlist_nums = []
while rlist != '':
    rs = str(rlist).split()
    tslist.append(int(rs[0]))
    loch = []
    locn = []
    for p in rs[1:]:
        ps = p.split(',')
        loch.append(int(ps[0]))
        if len(ps)>1:
            locn.append(float(ps[1]))
    if int(rs[0]) in badlist:
        loch.append(-1)
        ind = np.where(np.array(badlist)==int(rs[0]))[0][0]
        print (ind)
        locn.append(badlist_nums[ind])
    hostlist.append(np.array(loch))
    if len(ps)>1:
        nslist.append(np.array(locn))
    unilist.append(np.array(['        ' for x in loch]))
    # currently assumes 8 characters is fine, so max timestep is 9999, max halo is 999
    # easy to change if you need to
    rlist = rf.readline()
rf.close()

# for each host that has a new star in each snapshot, generate the unique ID
# if the host is part of a chain that we've already checked for consistency, just
# note that we've already found the unique ID for this host
fulltslist = sim.timesteps
fulltslist = np.array([int(str(tstr.extension).split('.')[-1]) for tstr in fulltslist])
kt = 0
for i in range(0,len(fulltslist)):
    if fulltslist[i] in tslist: # for each snapshot
        print ('------',fulltslist[i])
        for ctr, hid in enumerate(hostlist[kt]): # for each host in that snapshot
            if hid<=0: # if this host is just "amiga.grp didn't find one"...
                unilist[kt][ctr] = str(i).zfill(4)+'_0' # unique ID is just <snapshot index>_0 for now
            else: # Otherwise...
                pstr = str(i).zfill(4)+','+str(hid)
                print ('Current: '+str(fulltslist[i]).zfill(4)+'_'+str(hid))
                if d[pstr] != []: # If we've already found a self-consistent chain containing this host
                                  # don't repeat search
                    keystr = d[pstr]
                    print ('Found key: ',keystr)
                else: # Otherwise, construct the self-consistent chain and store the unique ID
                    unis,unih = trackforward(i,hid)
                    keystr = str(int(sim[int(unis)].extension.split('.')[-1])).zfill(4)+'_'+str(unih)
                    print ('Unique: ',keystr) # store the unique ID
                    proj,fid,phid = sim[int(unis)][unih].calculate_for_progenitors('halo_number()','finder_id()','type()')
                    curfid = fid[0]
                    pstr = str(unis).zfill(4)+','+str(proj[0])
                    d[pstr] = keystr 
                    for j in range(1,unis-i): # store the list of local IDs that correspond to this unique ID
                        if phid[j]<1: # avoid phantoms
                            try:
                                lh = sim[int(unis-j)][int(proj[j])].calculate('later('+str(j)+').finder_id()')
                                if lh == curfid:
                                    pstr = str(unis-j).zfill(4)+','+str(proj[j])
                                    d[pstr] = keystr
                            except:
                                match = checkmatch_d(unis-j,int(proj[j]),curfid,j)
                                if match:
                                    pstr = str(unis-j).zfill(4)+','+str(proj[j])
                                    d[pstr] = keystr
                unilist[kt][ctr] = keystr
        kt = kt+1

# Write out your list of unique IDs
wf = open(ofile,'w')
for i in range(0,len(tslist)):
    tstr = str(tslist[i])
    for ctr in range(0,len(hostlist[i])):
        tstr = tstr+'\t'+str(unilist[i][ctr])+','+str(hostlist[i][ctr])
        if len(ps)>1:
            tstr = tstr+','+str(nslist[i][ctr])
    tstr = tstr+'\n'
    wf.write(tstr)
wf.close()

'''
Created on Aug 20, 2021

@author: anna
'''
