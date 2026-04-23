#!/bin/bash
echo $1 # single command line argument is the name of the simulation - whatever comes before the first period (e.g., r431, cptmarvel)

simpath='/data/REPOSITORY/romulus_zooms/' # What's the root directory for these simulations?
outpath='/home/awright/dwarf_stellar_halos/'$1'/' # Where do you want your outputs to go? Not a good idea to change the final folder from $1
nproc=4 # How many processes do you want to run LocAtCreation on?
idkey='amiga.grp' # What ID keyword should be used to access halo IDs in pynbody (e.g., 'amiga.grp')?

# You'll only need to change these if you're using something other than the Massive Dwarfs
finstep='004096' # what's the final output of this simulation?
spec=$1'.romulus25.3072g1HsbBH' # what's the rest of the rootname for the simulation?

if [ ! -f $outpath ]; then # if output directory doesn't exist, make it
    echo 'creating '$outpath
    mkdir $outpath
fi

# Run steps 1-5
echo 'Step 1: grabbing star particle formation information'
python GrabTF_rz.py $simpath$spec'/'$spec'.'$finstep'/'$spec'.'$finstep $outpath

echo 'Step 2: finding locations of star particles at birth'
python LocAtCreation_pool_rz.py $simpath$spec'/' $outpath $nproc $idkey

echo 'Step 3: writing out new star hosts at each snapshot'
python writeouthosts_rz.py $outpath

echo 'Step 4: identifying a unique ID for each halo that hosts a new star'
python IDUniqueHost_rz.py $outpath

echo 'Step 5: writing out data'
python StoreUniqueHostID_rz.py $outpath

echo 'Done!'
