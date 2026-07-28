#This code takes in a series of trajectory files (ideally from a pymatnest nested sampling run)
#and calculates the XRD pattern using dans_diffraction package
#Setting have been chosen specifically for a study of magnesium and will need to be changed manually
#by a new user
#pymatnest: https://github.com/libAtoms/pymatnest/tree/master
#dans_diffraction: https://pypi.org/project/Dans-Diffraction/

#This code was authored by @V.G.Fletcher
#UK Ministry of Defence Copr. Crown owned copyright 2024/AWE

import time, os
import numpy as np
import ase
from ase.io import read, write
import ase.geometry.analysis as geom_ana

from mpi4py import MPI
import argparse

def calc_rdf(concat_name, results_prefix, els, cutoff, bins, comm, rank, size, verbose=False):
    """
    """

    ts = time.time()

    #check file can be opened
    try:
        all_strucs = ase.io.read(concat_name, index=':', parallel=True)
    except:
        print("Failed to read {}".format(concat_name))
        comm.Abort()
    comm.barrier()

    num_of_configs = len(all_strucs)

    #Divide work between threads
    configs_per_thread = round(num_of_configs/size)
    starting_int = configs_per_thread * rank

    #After even division, give any remaining jobs to the last thread
    if rank == (size-1):
        extra_files = num_of_configs - (configs_per_thread * size)
        configs_per_thread += extra_files
    if verbose:
        print('rank', rank, 'has', configs_per_thread, 'configs')
        t1 = time.time()
        if rank == 0:
            print('Setup and Error Checking took', t1-ts, 's')

    #Begin main loop
    thread_rdf = open(f"{results_prefix}.{rank}.npy", 'w')
    intensities = []
    for i in range(configs_per_thread):
        it = starting_int + i
        struc = all_strucs[it]
        print(rank, it)

        struc.set_cell(struc.get_cell().minkowski_reduce()[0])
        struc.wrap()
        struc*=[3,3,3]

        rdf, x = geom_ana.get_rdf(struc, cutoff, bins, elements=(els[0],els[1]))

        intensities.append(rdf)
    np.savetxt(thread_rdf, np.reshape(intensities, (configs_per_thread,len(rdf))))

    if verbose:
        t2 = time.time()
        print(f"Rank {rank} finished in {t2-t1} s")

    thread_rdf.close()
    comm.barrier()

    #Use root thread to concat files
    if rank == 0:
        master_rdf = open(f"{results_prefix}.npy", 'w')
        np.savetxt(master_rdf, np.reshape(x, (1,len(x))))
        for i in range(size):
            thread_rdf_dat = np.loadtxt(f"{results_prefix}.{i}.npy")
            np.savetxt(master_rdf, thread_rdf_dat)
            print(f"concatenated rank {i} data")
            os.remove(f"{results_prefix}.{i}.npy")
        master_rdf.close()
    comm.barrier()

    #Root thread closes the file and MPI finalise is called automatically in python
    if (rank == 0):
        print('Total time taken was', time.time()-ts)

    return

#Set up MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

#Parse args
parser = argparse.ArgumentParser(description='Create a database from nested sampling output')

parser.add_argument('-i', '--concat_name', action='store', help="Name of the file with all trajectories in iteration order", type=str, required=True)
parser.add_argument('-o', '--res_prefix', action='store', help="Prefix of files to save xrd data to", type=str, required=True)
parser.add_argument('-r', '--cutoff', action='store', help="RDF cutoff", type=float, required=True)
parser.add_argument('-e', '--el_el_int', action='store', help="Two elements for interaction", type=str, required=True)
parser.add_argument('-b', '--bins', action='store', help="N bins def. 200", type=int, default=200)

parser.add_argument('-V', '--verb', action='store_true', help="Verbosity of search")

args = parser.parse_args()

concat_name = args.concat_name
results_prefix = args.res_prefix
cutoff = args.cutoff
bins = args.bins
e_e_i = args.el_el_int
els = [int(el) for el in e_e_i.split()]

verbose   = args.verb

#Call xrd calculator
calc_rdf(concat_name, results_prefix, els, cutoff, bins, comm, rank, size, verbose=verbose)
