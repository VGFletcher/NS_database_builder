#xrd calculator for pymatnest authored by @V.G.Fletcher
#The purpose is to take the trajectories outputted from a nested sampling run and create a single file that contains...
#...the xrd data of every configuration in iteration order

import time, os
import numpy as np
import pandas as pd
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
    temps = []
    for struc in strucs:
        enth = struc.info['ns_energy'] #this is PE + KE + PV

        temps.append(struc.info['temp'])

        energies.append(enth)
        iterations.append(struc.info['iter'])
    
    return np.array(iterations), np.array(energies), np.array(temps)

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
    return Z, sum_z, shift

def read_ops(file_prefix):
    qw6 = np.array(pd.read_csv(f'{file_prefix}.qw6', delim_whitespace=True, skiprows=8, skipfooter=2, engine='python', names=['Q6', 'W6']))
    qw4 = np.array(pd.read_csv(f'{file_prefix}.qw4', delim_whitespace=True, skiprows=8, skipfooter=2, engine='python', names=['Q4', 'W4']))

    Q6 = qw6[:,0]
    W6 = qw6[:,1]
    Q4 = qw4[:,0]
    W4 = qw4[:,1]
    return Q6, W6, Q4, W4

#Set up MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

#Parse args
parser = argparse.ArgumentParser(description='Create temperature weighted XRD data from NS')

parser.add_argument('-i', '--traj_file', action='store', help="Name of the concat. traj file", type=str, required=True)
parser.add_argument('-k', '--n_walkers', action='store', help="Number of walkers used in the sampling", type=int, required=True)
parser.add_argument('-op', '--qw6_qw4_data', action='store', help="Prefix of the files with the order parameter data", type=str, required=True)
parser.add_argument('-tt', '--trans_temp', action='store', help="Transition temperature of liq-sol", type=float, required=True)

parser.add_argument('-ti', '--starting_temp', action='store', help="Starting temp for analysis", type=float, required=True)
parser.add_argument('-tf', '--final_temp', action='store', help="Final temp for analysis", type=float, required=True)
parser.add_argument('-dt', '--temp_step', action='store', help="Temperature step for analysis", type=float, required=True)

parser.add_argument('-o', '--res_prefix', action='store', help="Prefix of files to save xrd data to", type=str, required=True)

parser.add_argument('-V', '--verb', action='store_true', help="Verbosity of search")

args = parser.parse_args()

traj_file = args.traj_file
K = args.n_walkers
op_prefix = args.qw6_qw4_data
trans_t = args.trans_temp

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
iterations, energies, temps = read_energies(traj_file)
n_Es = iterations[-1] + 1
if verbose and rank==0:
    print('Read energies file')

#Calculate the temperature independent weights
log_w = log_weights(n_Es, K)[iterations]
if verbose and rank==0:
    print('Calculated t independent weights')
    
#Create T/F arrays for properties of interest
tmask = temps < trans_t

Q6, W6, Q4, W4 = read_ops(op_prefix)

hcp_m1 = (Q6 < 0.49745) * (Q6 > 0.4189)
hcp_m2 = W6 < 0.0

bcc_m1 = W6 > 0.0

hcp = tmask * hcp_m1 * hcp_m2
bcc = tmask * bcc_m1

res_file = open(f"{res_prefix}.{rank}.npy", 'w')
#Loop through the thread allcoated temperatures
for i in range(t_per_thread):
    it = starting_t + i
    T = t_vals[it]
    print(T)
    
    #Calcualte the Z values for each temp
    B = 1.0/(Kb*T)
    Z, sum_z, shift = Z_vals(B, log_w, energies)

    #sum_z = np.sum(Z[tmask])

    hcp_lz = -Kb*T*(np.log(np.sum(Z[hcp])) + shift)
    bcc_lz = -Kb*T*(np.log(np.sum(Z[bcc])) + shift)
    
    np.savetxt(res_file, np.reshape([T, hcp_lz, bcc_lz], (1,3)))
res_file.close()
comm.barrier()

#Use root thread to concat files
if rank == 0:
    master_dat = open(f"{res_prefix}.npy", 'w')
    for i in range(size):
        thread_dat = np.loadtxt(f"{res_prefix}.{i}.npy")
        np.savetxt(master_dat, thread_dat)
        print(f"concatenated rank {i} data")
        os.remove(f"{res_prefix}.{i}.npy")
    master_dat.close()
comm.barrier()
    
