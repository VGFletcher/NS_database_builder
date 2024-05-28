#uncertainty database builder code authored by @V.G.Fletcher
#The purpose is to take the trajectories outputted from a nested sampling run and create a single file that contains...
#...a specific number of the most uncertain configurations

import time, glob
import numpy as np
import ase
from ase.io import read, write
from mpi4py import MPI
import argparse

def db_builder(traj_files, db_name, db_size, comm, rank, size, temp_low_lim=None, temp_up_lim=None, verbose=False):
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

    #Read in the data
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
        if (temp_up_lim != None):
            for j in range(loop_len):
                struc = all_strucs[0]
                try:
                    temps = struc.info['temp']
                    remove = temps > temp_up_lim
                except:
                    print("Could not extract temperature value so configuration was removed")
                    remove = True
                    
                if remove:
                    del all_strucs[0]
                else:
                    break
                
        loop_len = len(all_strucs)
        if (temp_low_lim != None):
            for j in range(loop_len):
                struc[-1]
                try:
                    temps = struc.info['temp']
                    remove = temps < temp_low_lim
                except:
                    print("Could not extract temperature value so configuration was removed")
                    remove = True
                if remove:
                    del all_strucs[-1]
                else:
                    break
    
        loop_len = len(all_strucs)
        for j in range(loop_len):
            struc = all_strucs[j]
            try:
                w_sigma = struc.info['w_sigma']
            except:
                w_sigma = -np.inf
                
            data.append([it, j, w_sigma])
        struc_store += all_strucs

    if verbose and (rank == 0):
        t2 = time.time()
        print('Extraction took', t2-t1, 's')

    #Sort the arrays based on weighted uncertainty
    data     = np.array(data)
    w_sigmas = data[:,2]
    sort_ind = np.argsort(w_sigmas)

    data = data[sort_ind]
    strucs_sorted = [struc_store[i] for i in sort_ind]

    #clear memory
    del struc_store
    
    if (rank == 0):
        db_file = open(db_name, 'w')
        if verbose:
            t3 = time.time()
            print("Individual sorting took {} s".format(t3-t2))
        
    THREAD_LOC = -1
    for i in range(db_size):
        #Get the highest sigma value for each thread
        try:
            highest_sigma = data[-1][2]
        except:
            #If all sigma have been selected, set this to infinity
            highest_sigma = -np.inf

        #Gather all the highest sigma values to root
        high_sigmas = comm.gather(highest_sigma, root=0)

        #Use root to get the global highest
        if (rank == 0):
            THREAD_LOC = np.argsort(high_sigmas)[-1]
        else:
            THREAD_LOC = -1

        #Broadcast the global highest
        comm.barrier()
        THREAD_LOC = comm.bcast(THREAD_LOC, root=0)
        if (THREAD_LOC == rank):
            #The thread with the global highest is allowed to write
            ase.io.write('./{}'.format(db_name), strucs_sorted[-1], format='extxyz', parallel=False, append=True)
            #Reset its data arrays to exclude the saved config
            data = data[:-1]
            strucs_sorted = strucs_sorted[:-1]
        else:
            continue

    #Root thread closes the file and MPI finalise is called automatically in python
    if (rank == 0):
        db_file.close()
        print('Sorting took', time.time()-ts, 'to find the', db_size, 'most uncertain configs from', num_files, 'files' )

    return

#Set up MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

#Parse args
parser = argparse.ArgumentParser(description='Find the most uncertain configs from a nested sampling output')

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

temp_low_lim = args.low_temp
temp_up_lim = args.up_temp

verbose   = args.verb

#Get file names and broadcast, since glob may not return the order reliably
if (rank == 0):
    print("\nUsing regex", traj_regex, "to form a database of the most uncertain", db_size, "conigurations called", db_name, "\n")
    traj_files = glob.glob(traj_regex)
else:
    traj_files = []

traj_files = comm.bcast(traj_files, root=0)

#Call concatenator
db_builder(traj_files, db_name, db_size, comm, rank, size, temp_low_lim, temp_up_lim, verbose)
