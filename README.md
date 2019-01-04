# Sample-scripts
##insertion
This script adds molecules in random positions above or below a membrane using
Poisson disk sampling (https://en.wikipedia.org/wiki/Supersampling#Poisson_disc) to make sure points are not close to each other. 
All files concerining this cript are in the 'insertion' folder 
###Prerequesites
Requires mdanalysis and cython among other dependencies which can be intalled as follows:
```
pip install mdanalysis
pip install cython
```
###Installation
Following this the cython portion of the program can be installed as follows
```
python setup.py build_ext --inplace
```
###Example
The example script can be run as follows:
```
python above_membrane.py molecule.gro membrane.gro -d -2.5 -ms "resname POPE POPG CDL2" -n 6 -o test.gro
```
A description of the options availble can be found using
'''
python above_membrane.py -h
'''

##umbrella sampling
This script parses the output of umbrella sampling files in gromacs and runs Weighted Histogram Analysis Method using either the method included within gromacs or the 
Grossfiled implementation (see http://membrane.urmc.rochester.edu/?page_id=126). Note that this script is only a python wrapper for both implementions.
All files concerining this script are in the 'umbrella' folder. 
###Example
The example script can be run as follows:
```
python parse_pull_files.py -pxn umb_prod -p test_data/ -op test -s 1:41:0:2000 -skip 2 -n_jobs 6 --zprof0 4.8 --keep
```
A description of the options availble can be found using
'''
python parse_pull_files.py -h
'''

