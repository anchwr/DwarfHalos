#!/bin/bash
echo $1 # single command line argument is the name of the simulation - whatever comes before the first period (e.g., r431, cptmarvel)

simpath='/data/REPOSITORY/romulus_zooms/' # What's the root directory for these simulations?
outpath='/home/awright/dwarf_stellar_halos/'$1'/' # Where do you want your outputs to go? Not a good idea to change the final folder from $1
scriptpath='/home/awright/DwarfHalos/' # Where do the scripts live?
nproc=4 # How many processes do you want to run LocAtCreation on?
idkey='amiga.grp' # What ID keyword should be used to access halo IDs in pynbody (e.g., 'amiga.grp')?
pathformat=1 # What does your file hierarchy look like? If pathformat=1, we will assume snapshots are located in snapshot folders
               # If pathformat=2, we will assume snapshots are all located on the same level
AHF=False # Were your tangos dbs built using AHF rather than amiga? This alters the indexing slightly

# You'll only need to change these if you're using something other than the Massive Dwarfs
finstep='004096' # what's the final output of this simulation?
spec=$1'.romulus25.3072g1HsbBH' # what's the rest of the rootname for the simulation?

if [ ! -f $outpath ]; then # if output directory doesn't exist, make it
    echo 'creating '$outpath
    mkdir $outpath
fi

if [ $pathformat -eq 1 ]; then
    snappath=$simpath$spec'/'$spec'.'$finstep'/'$spec'.'$finstep
elif [ $pathformat -eq 2 ]; then
    snappath=$simpath$spec'/'$spec'.'$finstep
else
    echo 'pathformat value not understood'
    exit 1
fi

cd $scriptpath

# Run steps 1-5
echo 'Step 1: grabbing star particle formation information'
python GrabTF_rz.py $snappath $outpath || { echo "Fatal error occurred" >&2; exit 1; }

echo 'Step 2: finding locations of star particles at birth'
python LocAtCreation_pool_rz.py $simpath$spec'/' $outpath $nproc $idkey $pathformat $AHF || { echo "Fatal error occurred" >&2; exit 1; }

echo 'Step 3: writing out new star hosts at each snapshot'
python writeouthosts_rz.py $outpath || { echo "Fatal error occurred" >&2; exit 1; }

echo 'Step 4: identifying a unique ID for each halo that hosts a new star'
python IDUniqueHost_rz.py $outpath $spec || { echo "Fatal error occurred" >&2; exit 1; }

echo 'Step 5: writing out data'
python StoreUniqueHostID_rz.py $outpath || { echo "Fatal error occurred" >&2; exit 1; }

echo 'Done!'
