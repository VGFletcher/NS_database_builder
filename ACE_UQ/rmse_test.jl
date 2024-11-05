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

f = files[1]

poten = model.potential
###########################
###########################

println("Loading $f")
test_db = read_extxyz(f)

num_configs = length(test_db)
println("Loaded $f with $num_configs configs")

data_keys = (energy_key = "dft_energy", force_key = "dft_forces", virial_key = "dft_virial")

ACEpotentials.linear_errors(test_db, model; data_keys...);