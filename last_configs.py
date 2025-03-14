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
    data = []
    upper_its = []
    lower_its = []
    struc_store = {}
    for i in range(files_per_thread):
        it = starting_int + i

        if (verbose==True):
            print('rank', rank, 'accessing', traj_files[it])
            
        try:
            all_strucs = ase.io.read(traj_files[it], index=':', parallel=False)
        except:
            print("Failed to read {}".format(traj_files[it]))
            comm.Abort()

        pruned_loop_len = len(all_strucs)

        for j in range(pruned_loop_len):
            struc = all_strucs[j]

            itera = struc.info['iter']
                
            data.append([it, j, itera])
            struc_store[it] = all_strucs

        if (rank == 0) and verbose:
            t2 = time.time()
            print('Extraction took', t2-t1, 's')

    #Locate the closest values to the requested values for each thread
    data = np.array(data)

    DOI = data[:,2]
    DOI_cut = DOI
    
    VOI_ind = np.argsort(DOI_cut)[:db_size]

    VOI_vals = DOI[VOI_ind]

    if ((rank == 0) and (verbose==True)):
        t1 = t2
        t2 = time.time()
        print('Locating VOIs on root thread took', t2-t1, 's')

    #Send the found values to the root thread
    if (rank != 0):
        all_VOIs = np.array([[None], [None]])

    if (rank == 0):
        all_VOIs = np.empty((size, db_size))

    all_VOIs[:,:] = comm.gather(VOI_vals, root=0)
    #print('rank', rank, 'vois', all_VOIs)

    if ((rank == 0) and (verbose==True)):
        t1 = t2
        t2 = time.time()
        print('Gathering Data to root took', t2-t1, 's')

    #Root thread now finds the global optimal values
    if (rank == 0):
        index_vals = []
        actual_vals = []
        for i in range(db_size):
            ind = np.argmin(all_VOIs[:,i])
            actual_vals.append(all_VOIs[:,i][ind])
            index_vals.append(ind)

        unique_vals = len(np.unique(actual_vals))
        if (unique_vals != db_size):
            print("\nWARNING: NOT ENOUGH UNIQUE CONFIGS FOR GIVEN DATABASE SIZE")
            print(f"ONLY {unique_vals} OF {db_size} CONFIGS FOUND WERE UNIQUE AND HAVE BEEN WRITTEN TO {db_name}\n")

        db_file = open(db_name, 'w')

    if ((rank == 0) and (verbose==True)):
        t1 = t2
        t2 = time.time()
        print('Finding Global VOIs took', t2-t1, 's')
        print('Global VOIs are:', actual_vals)

    #Root thread broadcasts which thread has the global value for each iteration of the database size
    #This thread then writes its configuration to the database
    THREAD_LOC = -1
    #print('rank', rank, 'thread loc', THREAD_LOC)
    history = np.ones(db_size)*-1
    for i in range(db_size):
        if (rank == 0):
            THREAD_LOC = index_vals[i]

        comm.barrier()
        THREAD_LOC = comm.bcast(THREAD_LOC, root=0)
        if (THREAD_LOC == rank):
            data_location = VOI_ind[i]
            it = int(data[data_location][0])
            loc = int(data[data_location][1])
            
            itera = int(data[data_location][2])
            history[i] = itera
            first_time = sum(history == itera) == 1
            if first_time:
                struc = struc_store[it][loc]
                ase.io.write('./{}'.format(db_name), struc, format='extxyz', parallel=False, append=True)

        else:
            continue

    if ((rank == 0) and (verbose==True)):
        t1 = t2
        t2 = time.time()
        print('Writing relevant structures to .db file took', t2-t1, 's')

    #Root thread closes the file and MPI finalise is called automatically in python
    if (rank == 0):
        db_file.close()
        print('Sorting took', time.time()-ts, 'looking through', num_files, 'files, to find', db_size, 'data points' )

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
