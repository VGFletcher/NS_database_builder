#This code takes in a series of trajectory files from a pymatnest nested sampling run and creates a
#database of configurations that are equally spaced in iteration number, within a given temperature
#range
#pymatnest: https://github.com/libAtoms/pymatnest/tree/master

#This code was authored by @V.G.Fletcher
#UK Ministry of Defence Copr. Crown owned copyright 2024/AWE

import sys, time, glob
import numpy as np
import ase
from ase.io import read, write
from mpi4py import MPI
import argparse

def db_builder(traj_files, db_name, db_size, comm, rank, size, prun_lo_lim=None, prun_up_lim=None, verbose=False):
    """
    """

    ts = time.time()

    num_files = len(traj_files)

    #Basic error checks
    if (rank == 0):
        if (num_files == 0):
            print("No trajectory files found")
            comm.Abort()

        if ((num_files % size) != 0):
            print('Number of files cannot be evenly divided')
            comm.Abort()
    comm.barrier()

    #Divide work between threads
    files_per_thread = round(num_files/size)
    starting_int = files_per_thread * rank

    #print('rank', rank, 'starting int', starting_int)
    
    if ((rank == 0) and (verbose==True)):
        t1 = time.time()
        print('Setup and Error Checking took', t1-ts, 's')

    #Read in the data and prune based on the temperature, if given temperature ranges
    temps = []
    file_it = []
    struc_it = []
    all_strucs = []
    for i in range(files_per_thread):
        it = starting_int + i
        strucs = ase.io.read(traj_files[it], index=':', parallel=False)
        all_strucs.append(strucs)
        for j,struc in enumerate(strucs):
            temps.append(struc.info['temp'])
            file_it.append(i)
            struc_it.append(j)
            
    if (prun_up_lim != None):
        max_T = prun_up_lim
    else:
        max_T = np.max(temps)
        
    if (prun_lo_lim != None):
        min_T = prun_lo_lim
    else:
        min_T = np.min(temps)

    VOIs = np.linspace(max_T,min_T,db_size)
    #VOIs = np.flip(np.sort(np.random.uniform(min_T,max_T,db_size)))
    local_db = []
    local_T = []
    for VOI in VOIs:
        match = np.argmin(np.abs(np.array(temps) - VOI))
        struc = all_strucs[file_it[match]][struc_it[match]]
        local_db.append(struc)
        local_T.append(struc.info['temp'])
        del all_strucs[file_it[match]][struc_it[match]]
        del temps[match]

    del all_strucs

    if (rank != 0):
        all_VOIs = np.array([[None], [None]])
    if (rank == 0):
        all_VOIs = np.empty((size, db_size))

    all_VOIs[:,:] = comm.gather(local_T, root=0)
    
    global_db_i = []
    if rank == 0:
        global_db_i = np.argmin(all_VOIs, axis=0)
        print(np.min(all_VOIs, axis=0))
        
        db_file = open(db_name, 'w')
        db_file.close()

    if rank ==0:
        print(global_db_i)
    global_db_i = comm.bcast(global_db_i, root=0)
    
    comm.barrier()
    for i,j in enumerate(global_db_i):
        if (j == rank):
            struc = local_db[i]
            ase.io.write('./{}'.format(db_name), struc, format='extxyz', parallel=False, append=True)
        comm.barrier()

    return

#Set up MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

#Parse args
parser = argparse.ArgumentParser(description='Create a database from nested sampling output')

parser.add_argument('-i', '--traj_regex', action='store', help="Regex to identify the trajectory files to search", type=str, required=True)
parser.add_argument('-o', '--db_name', action='store', help="Name of the output database", type=str, required=True)
parser.add_argument('-s', '--db_size', action='store', help="how many configs to put in the database", type=int, required=True)

parser.add_argument('-lt', '--low_temp', action='store', help="lower temperature limit to restrict search", type=int, default=None)
parser.add_argument('-ut', '--up_temp', action='store', help="upper temperature limit to restrict search", type=int, default=None)
parser.add_argument('-V', '--verb', action='store_true', help="Verbosity of search")

args = parser.parse_args()

traj_regex = args.traj_regex
db_name    = args.db_name
db_size    = args.db_size

low_t_lim = args.low_temp
up_t_lim  = args.up_temp
verbose   = args.verb

#Get file names and broadcast, since glob may not return the order reliably
if (rank == 0):
    print("\nUsing regex", traj_regex, "to form a database of", db_size, "conigurations called", db_name, "considering only configurations estimated to be bewteen", low_t_lim,"K and", up_t_lim,"K\n")
    traj_files = glob.glob(traj_regex)
else:
    traj_files = []

traj_files = comm.bcast(traj_files, root=0)

#Call sorter
db_builder(traj_files, db_name, db_size, comm, rank, size, prun_lo_lim=low_t_lim, prun_up_lim=up_t_lim, verbose=verbose)
