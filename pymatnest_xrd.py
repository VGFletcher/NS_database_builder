#xrd calculator for pymatnest authored by @V.G.Fletcher
#The purpose is to take the trajectories outputted from a nested sampling run and create a single file that contains...
#...the xrd data of every configuration in iteration order

import time, os
import numpy as np
import ase
from ase.io import read, write
import Dans_Diffraction as dif
from mpi4py import MPI
import argparse

def calc_xrd(concat_name, results_prefix, comm, rank, size, verbose=False):
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
    thread_xrd = open(f"{results_prefix}.{rank}.npy", 'w')
    intensities = []
    for i in range(configs_per_thread):
        it = starting_int + i
        struc = all_strucs[it]

        #Extract configuration parameters
        cell = struc.cell.cellpar()
        types = struc.get_chemical_symbols()
        positions = struc.get_scaled_positions()

        #Create crystal structure for Dans Diffraction
        xtl = dif.Crystal()
        xtl.new_cell(cell)
        xtl.new_atoms(u=positions[:,0],v=positions[:,1],w=positions[:,2],type=types)

        #Setup and perform x-ray spectra
        xtl.Scatter.setup_scatter(scattering_type='x-ray', wavelength_a=0.398, min_twotheta=10, max_twotheta=30)
        twotheta, inten, reflections = xtl.Scatter.powder(units='twotheta', pixels=200)

        #Scale the results
        scale_f = np.max(inten)
        inten = inten/scale_f

        intensities.append(inten)
    np.savetxt(thread_xrd, np.reshape(intensities, (configs_per_thread,len(inten))))

    if verbose:
        t2 = time.time()
        print(f"Rank {rank} finished in {t2-t1} s")

    thread_xrd.close()
    comm.barrier()

    #Use root thread to concat files
    if rank == 0:
        master_xrd = open(f"{results_prefix}.npy", 'w')
        np.savetxt(master_xrd, np.reshape(twotheta, (1,len(twotheta))))
        for i in range(size):
            thread_xrd_dat = np.loadtxt(f"{results_prefix}.{i}.npy")
            np.savetxt(master_xrd, thread_xrd_dat)
            print(f"concatenated rank {i} data")
            os.remove(f"{results_prefix}.{i}.npy")
        master_xrd.close()
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

parser.add_argument('-V', '--verb', action='store_true', help="Verbosity of search")

args = parser.parse_args()

concat_name = args.concat_name
results_prefix = args.res_prefix

verbose   = args.verb

#Call xrd calculator
calc_xrd(concat_name, results_prefix, comm, rank, size, verbose=verbose)
