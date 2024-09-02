#This code evaluates the temperature weighted uncertainty (sigma_T) of a database of configurations
#with given sigma values where sigma_T = 1 - exp(-sigma/(Kb*T))

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
#Pkg.add("Glob")
#Pkg.add("ArgParse")
##########################
##########################

############################
#####IMPORTING_PACKAGES#####
############################
@everywhere using ACEpotentials
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
end

parsed_args = parse_args(ARGS, s)

files = Glob.glob(parsed_args["traj_regex"])
###########################
###########################

#######################
#####PARALLEL_LOOP#####
#######################
@sync @distributed for f in files
    Kb = 8.61733326 * 10^-5 #eV/K
   
    println("Loading $f")
    dataset = read_extxyz(f)

    num_configs = length(dataset)
    println("Loaded $f with $num_configs configs")

    eval_configs = Array{Atoms{Float64}, 1}()
    for j in range(start=1, stop=num_configs, step=1)

    	config = deepcopy(dataset[1])

	sigma = config.data["sigma"].data

	temp = config.data["temp"].data
	if temp != "-inf" && temp != "inf" && !isnan(temp)
	   w_sigma = 1 - exp(-sigma/(Kb*temp))
	   dataset[1]["w_sigma"] = w_sigma
	end
	
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