#Temperature calculator authored by @V.G.Fletcher
#The purpose is to take the data file produced by ns_analyse and use this to predict the temperature...
#...of the sampled configurations

import time, glob, os
import numpy as np
import pandas as pd
from scipy import interpolate
import ase
from ase.io import read, write
from mpi4py import MPI
import argparse

def calc_temps(dat_file, traj_files, comm, rank, size, verbose=False):
    """
    This function calculates the temperature of each configuration using the nested sampling analysis file.
    It then saves this value to the .extxyz file so that it can be accessed in one file.
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

    #Try to open data file from ns analyse
    try:
        dat = np.array(pd.read_csv(dat_file, delim_whitespace=True,skiprows=[0]))
        #Extract temperature and enthalpy columns
        T = dat[:,0]
        U = dat[:,3]
    except:
        print("Could not read {}".format(dat_file))
        comm.Abort()

    #Divide work between threads
    files_per_thread = round(num_files/size)
    starting_int = files_per_thread * rank

    #Put values into interpolation function to convert any enthalpy value to a temperature
    U_T_func = interpolate.interp1d(U, T, fill_value="extrapolate")

    if ((rank == 0) and (verbose==True)):
        t1 = time.time()
        print('Setup and Error Checking took', t1-ts, 's')

    for fs in range(files_per_thread):
        if verbose:
            t2 = time.time()
            
        it = starting_int + fs

        #Try to open the trajectory file
        try:
            if verbose:
                print("Rank", rank, "Reading", traj_files[it])
            all_strucs = ase.io.read(traj_files[it], index=':', parallel=False)
        except:
            print("Error reading traj file {}".format(traj_files[it]))
            comm.Abort()

        #Loop through all structures
        len_all_strucs = len(all_strucs)

        if verbose:
            print("Rank", rank, "working on", len_all_strucs, "configurations")
            
        for i in range(len_all_strucs):
            struc = all_strucs[0]

            #Extract enthalpy
            ns_energy = struc.info['ns_energy']

            #Predict temperature
            struc.info['temp'] = U_T_func(ns_energy)

            #Save structure
            ase.io.write(traj_files[it] + ".tmp", struc, format="extxyz", append=True, parallel=False)

            #remove from memory
            del struc
            del all_strucs[0]

        #Overwrite old trajectory file with successfully produced new traj file
        os.rename(traj_files[it],          traj_files[it] + '.old')
        os.rename(traj_files[it] + '.tmp', traj_files[it])
        os.remove(traj_files[it] + '.old')

        if verbose:
            t3 = time.time()
            print("Rank", rank, "finished with", traj_files[it], "in", t3-t2, "s")

    if (rank == 0) and verbose:
        te = time.time()
        print('Total time taken', te-ts, 's')

    return

#Set up MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

#Parse args
parser = argparse.ArgumentParser(description='Calculate the temperature of configurations from a nested sampling output')

parser.add_argument('-i', '--traj_regex', action='store', help="Regex to identify the trajectory files to search", type=str, required=True)
parser.add_argument('-d', '--dat_file', action='store', help="Name of the ns_analyse results file", type=str, required=True)
parser.add_argument('-V', '--verb', action='store_true', help="Verbosity of calculation")

args = parser.parse_args()

traj_regex = args.traj_regex
dat_file   = args.dat_file
verbose   = args.verb

#Get file names and broadcast, since glob may not return the order reliably
if (rank == 0):
    print("\nUsing results from", dat_file, "to predict the temperture of configurations in files with regex", traj_regex)
    traj_files = glob.glob(traj_regex)
else:
    traj_files = []

traj_files = comm.bcast(traj_files, root=0)

#Call calculator
calc_temps(dat_file, traj_files, comm, rank, size, verbose=verbose)
