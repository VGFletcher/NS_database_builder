#This code is an example use of the acefit.jl code package to evaluate the uncertainty of a database
#of configurations based on a fitted ACE potential committee
#acefit.jl: https://github.com/ACEsuit/ACEfit.jl

#This code was authored by @V.G.Fletcher
#UK Ministry of Defence Copr. Crown owned copyright 2024/AWE

using Distributed

@everywhere import Pkg

##########################
#####First time setup#####
##########################
#Pkg.Registry.add("General")
#Pkg.Registry.add(Pkg.RegistrySpec(url="https://github.com/ACEsuit/ACEregistry"))
#Pkg.add("ACEpotentials")
#Pkg.add("JLD2")
#Pkg.add("Glob")
#Pkg.add("ArgParse")
##########################
##########################

############################
#####IMPORTING_PACKAGES#####
############################
@everywhere using ACEpotentials
@everywhere using JLD2
@everywhere using Statistics
using Glob
using ArgParse
############################
############################

###########################
#####IMPORT_PARAMETERS#####
###########################
s = ArgParseSettings()
@add_arg_table s begin
    "--traj_regex", "-i"
    help = "The regex used to identify all traj files in extxyz format"
    arg_type = AbstractString
    required = true
    
    "--ace_model", "-m"
    help = "The trained ACE model in .jld2 format"
    arg_type = AbstractString
    required = true
end

parsed_args = parse_args(ARGS, s)

files = Glob.glob(parsed_args["traj_regex"])
model = load_object(parsed_args["ace_model"])

poten = model.potential
###########################
###########################

#######################
#####PARALLEL_LOOP#####
#######################
@sync @distributed for f in files
   
    println("Loading $f")
    dataset = read_extxyz(f)

    num_configs = length(dataset)
    println("Loaded $f with $num_configs configs")

    eval_configs = Array{Atoms{Float64}, 1}()
    for j in range(start=1, stop=num_configs, step=1)

    	config = deepcopy(dataset[1])
	
	E, E_co = ACE1.co_energy(poten, config)
        sigma = sqrt(mean((E_co .- E).^2))

	dataset[1]["sigma"] = sigma
	dataset[1]["ACE_energy"] = E

	push!(eval_configs, dataset[1])
	deleteat!(dataset, 1)
	
	if (j%1000) == 0
	   GC.gc()
	end
    end
    
    write_extxyz(f * ".tmp", eval_configs)

    mv(f, f * ".old")
    mv(f * ".tmp", f)
    rm(f * ".old")
    
    dataset = nothing
    eval_configs = nothing

    GC.gc()
end
#######################
#######################