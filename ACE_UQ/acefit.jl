#This code is an example use of the acefit.jl code package for fitting a magnesium ACE potential
#acefit.jl: https://github.com/ACEsuit/ACEfit.jl

#This example was authored by @V.G.Fletcher
#UK Ministry of Defence Copr. Crown owned copyright 2024/AWE

import Pkg

##########################
#####First time setup#####
##########################
#Pkg.Registry.add("General")
#Pkg.Registry.add(Pkg.RegistrySpec(url="https://github.com/ACEsuit/ACEregistry"))
#Pkg.add("ACEpotentials")
#Pkg.add("Suppressor")
#Pkg.add("JLD2")
##########################
##########################

############################
#####IMPORTING_PACKAGES#####
############################
using LinearAlgebra
using ACEpotentials
using Suppressor
using Random
using JLD2
############################
############################

########################
#####FIT_PARAMETERS#####
########################
order = parse(Int64, ARGS[1])
degree = parse(Int64, ARGS[2])
test_ratio = parse(Float64, ARGS[3])

dataset_name = "master.edb.extxyz"
output_name = "o$(order)_d$(degree)"
model_name = "o$(order)_d$(degree)"

erefs = Dict("Mg" => -1688.821130128)
elements = [:Mg]
rcut = 8.2

e_weight = 9.0
f_weight = 1.0
v_weight = 1.0
blr_tol = 1e-3
n_members = 10

repul_rest = false
repul_weight = 0.0

#0.5 is too extreme, 0.2 is reasonable
#higher numbers reduce importance of high temp configs
alpha = 0.4
#####################
#####################

data_keys = (energy_key = "dft_energy", force_key = "dft_forces", virial_key = "dft_virial")

#rcut = cutoff of interactions
#order = body order of interactions, creat n+1 body potential
#totaldegree = polynomial degree of fitting
#Eref= energy of isolated atoms
model = acemodel(elements = elements,
                     rcut = rcut,
                    order = order,
              totaldegree = degree,
                     Eref = erefs)

#prior = smoothness_prior(model)
solver = ACEfit.BLR(tol=blr_tol, committee_size=n_members, factorization=:svd)

#####################
#####################

##########################
#####TEST_TRAIN_SPLIT#####
##########################
dataset = read_extxyz(dataset_name)
ds_size = length(dataset)
test_size = round(Int, test_ratio * ds_size)
train_size = ds_size - test_size

println("TOTAL_DATASET_HAS_", ds_size, "_CONFIGURATIONS")
println("TRAINING_SET_HAS_", train_size, "_CONFIGURATIONS")
println("TEST_SET_HAS_", test_size, "_CONFIGURATIONS\n")

shuffle!(dataset)
test_db = dataset[1:test_size]
train_db = dataset[test_size+1:ds_size]

##########################
##########################

##########################
#####Creating Weights#####
##########################
pressure_sets = Dict()
for (i, at) in enumerate(train_db)
    press = at.data["ns_P"].data
    at.data["config_type"].data = "con_" * string(i)
    pressure_sets[press] = [[],[]]
end

#Calculate the DFT enthalpy from the DFT potential energy
for at in train_db
    press = at.data["ns_P"].data
    i = at.data["config_type"].data
    
    pe = at.data["dft_energy"].data
    cell_v = det(cell(at))
    pv = cell_v * press
    H = pe + pv
    push!(pressure_sets[press][1], i)
    push!(pressure_sets[press][2], H)
end

#Rescale all the sets to their lowest enthalpy configuration
for p_set in keys(pressure_sets)
    enes = pressure_sets[p_set][2]
    E0 = enes[argmin(enes)]
    scaled_ene = enes .- E0
    pressure_sets[p_set][2] = scaled_ene
end

#Calculate the scaling factors through sf = exp(-alpha * (E-E0))
for p_set in keys(pressure_sets)
    enes = pressure_sets[p_set][2]
    sfs = exp.(-alpha .* enes)
    pressure_sets[p_set][2] = sfs
end

#Create a dictionary of the rescaled weights
weights = Dict()
for p_set in keys(pressure_sets)
    for (i,w) in zip(pressure_sets[p_set][1],pressure_sets[p_set][2])
        weights[i] = Dict("E" => e_weight * w, "F" => f_weight * w, "V" => v_weight * w)
    end
end

##########################
##########################

###########################
#####FITTING_PROCEDURE#####
###########################

println("BEGINNING_ACE_FIT_WITH_", length(model.basis), "_BASIS_FUNCTIONS")

@suppress begin
	acefit!(model, train_db; solver=solver, data_keys..., weights=weights, repulsion_restraint=repul_rest, restraint_weight=repul_weight);
end

println("FINISHED_ACE_FIT\n")

###########################
###########################

###########################
#####TESTING_PROCEDURE#####
###########################

println("BEGINNING_TRAINING_ERRORS")
ACEpotentials.linear_errors(train_db, model; data_keys...);
println("FINISHED_TRAINING_ERRORS\n")

if (length(test_db) > 0)
   println("BEGINNING_TEST_ERRORS")
   ACEpotentials.linear_errors(test_db, model; data_keys...);
   println("FINISHED_TEST_ERRORS\n")
end

export2lammps(output_name * ".yace", model)
save_object(model_name * ".jld2", model)
save_potential(output_name * ".json", model)

###########################
###########################