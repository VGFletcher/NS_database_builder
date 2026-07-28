#This code takes in the trajectory ouput files from a multithreaded pymatnest nested sampling run,
#sorts them by iteration number, and saves them to a single file
#pymatnest: https://github.com/libAtoms/pymatnest/tree/master
#!!CODE SORTS BY NS_ENERGY NOT ITERATION NUMBER!!
#Any reference to iteration number is now outdated 

#This code was authored by @V.G.Fletcher
#UK Ministry of Defence Copr. Crown owned copyright 2024/AWE

import time, glob
import numpy as np
import ase
from ase.io import read, write
from mpi4py import MPI
import argparse

def traj_concat(traj_files, concat_name, comm, rank, size, verbose=False):
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
    
    if ((rank == 0) and (verbose==True)):
        t1 = time.time()
        print('Setup and Error Checking took', t1-ts, 's')

    #Read in the data and prune based on the temperature, if given temperature ranges
    data = []
    struc_store = []
    for i in range(files_per_thread):
        it = starting_int + i

        if verbose:
            print('rank', rank, 'accessing', traj_files[it])
            
        try:
            all_strucs = ase.io.read(traj_files[it], index=':', parallel=False)
        except:
            print("Failed to read {}".format(traj_files[it]))
            comm.Abort()

        loop_len = len(all_strucs)
        
        for j in range(loop_len):
            struc = all_strucs[j]

            itera = struc.info['ns_energy']
                
            data.append([it, j, itera])
        struc_store += all_strucs

    if verbose and (rank == 0):
        t2 = time.time()
        print('Extraction took', t2-t1, 's')

    #Sort the arrays based on iteration number
    data       = np.array(data)
    iterations = data[:,2]
    sort_ind   = np.argsort(iterations)[::-1]

    data = data[sort_ind]
    strucs_sorted = [struc_store[i] for i in sort_ind]

    #clear memory
    del struc_store
    
    num_configs = len(iterations)

    tot_configs = comm.allreduce(num_configs, op=MPI.SUM)
    
    if (rank == 0):
        concat_file = open(concat_name, 'w')
        if verbose:
            t3 = time.time()
            print("Individual sorting took {} s".format(t3-t2))
        
    THREAD_LOC = -1
    for i in range(tot_configs):
        #Get the lowest iteration number for each thread
        try:
            lowest_iteration = data[0][2]
        except:
            #If all iterations have been selected set this to infinity
            lowest_iteration = -np.inf

        #Gather all the lowest iteration numbers to root
        lowest_its = comm.gather(lowest_iteration, root=0)

        #Use root to get the global lowest
        if (rank == 0):
            THREAD_LOC = np.argsort(lowest_its)[::-1][0]
        else:
            THREAD_LOC = -1

        #Broadcast the global lowest
        comm.barrier()
        THREAD_LOC = comm.bcast(THREAD_LOC, root=0)
        if (THREAD_LOC == rank):
            #The thread with the global lowest is allowed to write
            ase.io.write('./{}'.format(concat_name), strucs_sorted[0], format='extxyz', parallel=False, append=True)
            #Reset its data arrays to exclude the saved config
            data = data[1:]
            strucs_sorted = strucs_sorted[1:]
        else:
            continue

    #Root thread closes the file and MPI finalise is called automatically in python
    if (rank == 0):
        concat_file.close()
        print('Concatenating took', time.time()-ts, 'to concat', tot_configs, 'configs from', num_files, 'files' )

    return

#Set up MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

#Parse args
parser = argparse.ArgumentParser(description='Create a database from nested sampling output')

parser.add_argument('-i', '--traj_regex', action='store', help="Regex to identify the trajectory files to search", type=str, required=True)
parser.add_argument('-o', '--concat_name', action='store', help="Name of the output file", type=str, required=True)

parser.add_argument('-V', '--verb', action='store_true', help="Verbosity of search")

args = parser.parse_args()

traj_regex = args.traj_regex
concat_name = args.concat_name

verbose   = args.verb

#Get file names and broadcast, since glob may not return the order reliably
if (rank == 0):
    print("\nUsing regex", traj_regex, "to form a concatenated file\n")
    traj_files = glob.glob(traj_regex)
else:
    traj_files = []

traj_files = comm.bcast(traj_files, root=0)

#Call concatenator
traj_concat(traj_files, concat_name, comm, rank, size, verbose=verbose)
