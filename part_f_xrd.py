#xrd calculator for pymatnest authored by @V.G.Fletcher
#The purpose is to take the trajectories outputted from a nested sampling run and create a single file that contains...
#...the xrd data of every configuration in iteration order

import time, os
import numpy as np
import ase
from ase.io import read
from scipy import constants as sc
from mpi4py import MPI
import argparse

Kb = sc.physical_constants['Boltzmann constant in eV/K'][0]

def read_energies(file_name):
    strucs = ase.io.read(file_name, index=':')
    energies = []
    iterations = []
    for struc in strucs:
        energies.append(struc.info['ns_energy'])
        iterations.append(struc.info['iter'])
    
    return np.array(iterations), np.array(energies)

def log_weights(n_Es, K):
    i_vals = np.arange(0, n_Es, 1)
    
    log_X = np.zeros_like(i_vals) + np.log(K/(K+1))
    log_Xn = np.cumsum(log_X)
    
    log_w = log_Xn - np.log(K+1)
    return log_w

def Z_vals(B, log_weights, energies):
    log_z = log_weights - (energies * B)
    shift = np.max(log_z)
    
    Z = np.exp(log_z-shift)
    sum_z = np.sum(Z)
    return Z, sum_z
    
#Set up MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

#Parse args
parser = argparse.ArgumentParser(description='Create temperature weighted XRD data from NS')

parser.add_argument('-i', '--traj_file', action='store', help="Name of the concat. traj file", type=str, required=True)
parser.add_argument('-k', '--n_walkers', action='store', help="Number of walkers used in the sampling", type=int, required=True)
parser.add_argument('-xrd', '--xrd_data', action='store', help="Name of the file with the xrd data", type=str, required=True)

parser.add_argument('-ti', '--starting_temp', action='store', help="Starting temp for analysis", type=float, required=True)
parser.add_argument('-tf', '--final_temp', action='store', help="Final temp for analysis", type=float, required=True)
parser.add_argument('-dt', '--temp_step', action='store', help="Temperature step for analysis", type=float, required=True)

parser.add_argument('-o', '--res_prefix', action='store', help="Prefix of files to save xrd data to", type=str, required=True)

parser.add_argument('-V', '--verb', action='store_true', help="Verbosity of search")

args = parser.parse_args()

traj_file = args.traj_file
K = args.n_walkers
xrd_file = args.xrd_data

start_t = args.starting_temp
final_t = args.final_temp
delta_t = args.temp_step

res_prefix = args.res_prefix

verbose   = args.verb

t_vals = np.arange(start_t, final_t, delta_t)

num_t_vals = len(t_vals)

#Divide work between threads
t_per_thread = round(num_t_vals/size)
starting_t = t_per_thread * rank

#After even division, give any remaining jobs to the last thread
if rank == (size-1):
    extra_t = num_t_vals - (t_per_thread * size)
    t_per_thread += extra_t

if verbose:
    print(f"Rank {rank} starting from {start_t} K to {final_t} K with step {delta_t} K")

#Read the energies file
iterations, energies = read_energies(traj_file)
n_Es = iterations[-1] + 1
if verbose and rank==0:
    print('Read energies file')

#Calculate the temperature independent weights
log_w = log_weights(n_Es, K)[iterations]
if verbose and rank==0:
    print('Calculated t independent weights')
    
#Import the xrd data
xrd_dat = np.loadtxt(xrd_file)
if verbose and rank==0:
    print('Loaded XRD data')

res_file = open(f"{res_prefix}.{rank}.npy", 'w')
#Loop through the thread allcoated temperatures
for i in range(t_per_thread):
    it = starting_t + i
    T = t_vals[it]
    print(T)
    
    #Calcualte the Z values for each temp
    B = 1.0/(Kb*T)
    Z, sum_Z = Z_vals(B, log_w, energies)
    
    weighted_xrd = np.mean(xrd_dat[1:] * np.reshape(Z, (len(Z),1)), axis=0)
    np.savetxt(res_file, np.reshape(weighted_xrd, (1, len(weighted_xrd))))
res_file.close()
comm.barrier()

#Use root thread to concat files
if rank == 0:
    master_xrd = open(f"{res_prefix}.npy", 'w')
    np.savetxt(master_xrd, np.reshape(xrd_dat[0], (1,len(xrd_dat[0]))))
    for i in range(size):
        thread_xrd_dat = np.loadtxt(f"{res_prefix}.{i}.npy")
        np.savetxt(master_xrd, thread_xrd_dat)
        print(f"concatenated rank {i} data")
        os.remove(f"{res_prefix}.{i}.npy")
    master_xrd.close()
comm.barrier()
    
