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
#Pkg.add("ArgParse")
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
using ArgParse
############################
############################

sett = ArgParseSettings()
@add_arg_table sett begin
    "-o"
        help = "Order of the ACE potential"
        arg_type = Int64
        required = true
    "-d"
        help = "Degree of the ACE potential"
        arg_type = Int64
        required = true
    "-r"
        help = "Test Ratio"
        arg_type = Float64
        required = false
        default = 0.0
    "-a"
        help = "Exponential weighting scheme alpha value"
        arg_type = Float64
        required = false
        default = nothing
    "-w"
        help = "Angular Resolution, lower increases resolution"
        arg_type = Float64
        required = false
        default = 1.5
end
parsed_args = parse_args(ARGS, sett)

########################
#####FIT_PARAMETERS#####
########################

order      = parsed_args["o"]
degree     = parsed_args["d"]
test_ratio = parsed_args["r"]
alpha      = parsed_args["a"]
wL         = parsed_args["w"]

dataset_name = "master.edb.extxyz"
output_name = "o$(order)_d$(degree)"
model_name = "o$(order)_d$(degree)"

erefs = Dict("Ti" => -1587.531990, "Al" => -107.344591, "V" => -1946.496760)
elements = [:Ti,:Al,:V]
rcut = 5.0

e_weight = 9.0
f_weight = 1.0
v_weight = 1.0
blr_tol = 1e-3
n_members = 10

repul_rest = false
repul_weight = 0.0

#####################
#####################

data_keys = (energy_key = "dft_energy", force_key = "dft_forces", virial_key = "dft_virial")

erefs_sym = Dict()
for key in keys(erefs)
    erefs_sym[Symbol(key)] = erefs[key]
end

#rcut = cutoff of interactions
#order = body order of interactions, creat n+1 body potential
#totaldegree = polynomial degree of fitting
#Eref= energy of isolated atoms
model = acemodel(elements = elements,
                     rcut = rcut,
                    order = order,
              totaldegree = degree,
                     Eref = erefs,
		     wL=wL)

prior = smoothness_prior(model; p=1)
solver = ACEfit.BLR(tol=blr_tol, committee_size=n_members, factorization=:svd)

#####################
#####################

##########################
#####TEST_TRAIN_SPLIT#####
##########################
dataset = read_extxyz(dataset_name)
ds_size_all = length(dataset)

test_db = Array{Atoms, 1}()
train_db = Array{Atoms, 1}()

if test_ratio != 0
   sorted_dataset = Dict()
   for at in dataset
       pset = at.data["config_type"].data
       sorted_dataset[pset] = Array{Atoms, 1}()
   end

   for at in dataset
       pset = at.data["config_type"].data
       push!(sorted_dataset[pset], at)
   end

   for pset in keys(sorted_dataset)
       m = match(r"_hole$", pset)
       if m != nothing
       	  hset = sorted_dataset[pset]
      	  global train_db = cat(train_db, hset, dims=1)
      	  delete!(sorted_dataset, pset)
       end	 
   end
  
   for pset in keys(sorted_dataset)
       #shuffle!(sorted_dataset[pset])
       ds_size = length(sorted_dataset[pset])
       
       p_tset_len = (test_ratio * ds_size)
       test_size = round(Int, p_tset_len)
       step = div(ds_size, test_size)

       mask = falses(ds_size)
       mask[1:step:end] .= true #Test samples, equally spaced in dataset
       inv_mask = .!mask #train samples

       p_teset = sorted_dataset[pset][mask]
       p_trset = sorted_dataset[pset][inv_mask]

       global test_db = cat(test_db, p_teset, dims=1)
       global train_db = cat(train_db, p_trset, dims=1)
   end

   for at in test_db
       temp = at.data["temp"].data
       at.data["config_type"].data = "T_" * string(temp)
   end 

else
   train_db = dataset
end

te_size = length(test_db)
tr_size = length(train_db)

println("TOTAL_DATASET_HAS_", ds_size_all, "_CONFIGURATIONS")
println("TRAINING_SET_HAS_", tr_size, "_CONFIGURATIONS")
println("TEST_SET_HAS_", te_size, "_CONFIGURATIONS\n")

# shuffle!(dataset)
# test_db = dataset[1:test_size]
# train_db = dataset[test_size+1:ds_size]

##########################
##########################

##########################
#####Creating Weights#####
##########################
if alpha != nothing

pressure_sets = Dict()
for (i, at) in enumerate(train_db)
    #press = at.data["ns_P"].data
    #at.data["config_type"].data = "con_" * string(i)
    press = at.data["config_type"].data
    pressure_sets[press] = [[],[]]
end

#Calculate the DFT enthalpy from the DFT potential energy
for at in train_db
    press = at.data["ns_P"].data
    i = at.data["config_type"].data

    symbs = chemical_symbols(at)
    base_ene = sum(getindex.(Ref(erefs_sym), symbs))
    
    pe = at.data["dft_energy"].data - base_ene
    
    cell_v = det(cell(at))
    pv = cell_v * press
    H = (pe + pv)/length(at)
    push!(pressure_sets[i][1], i)
    push!(pressure_sets[i][2], H)
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

else
    weights = Dict( "default" => Dict("E" => e_weight, "F" => f_weight, "V" => v_weight))
end
##########################
##########################

###########################
#####FITTING_PROCEDURE#####
###########################

println("BEGINNING_ACE_FIT_WITH_", length(model.basis), "_BASIS_FUNCTIONS")

#@suppress begin
	acefit!(model, train_db; solver=solver, prior=prior, data_keys..., weights=weights, repulsion_restraint=repul_rest, restraint_weight=repul_weight);
#end

println("FINISHED_ACE_FIT\n")

export2lammps(output_name * ".yace", model)
save_object(model_name * ".jld2", model)
save_potential(output_name * ".json", model)

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

###########################
###########################