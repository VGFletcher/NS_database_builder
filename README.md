### This repository consists of code so that you can form databases of configurations from the output of nested sampling runs.
Code authored by V.G.Fletcher <br /> 
UK Ministry of Defence © Crown owned copyright 2024/AWE

<hr>

# database_builder.py
Assuming you have a collection of trajectory files from a nested sampling run, you can run this script to search all of the trajectory files and create a database, of a given size, consisting of configurations that are as close to equally spaced in iteration number as possible. <br /> 
The possibility of getting exactly equal spacing is subject to your temperature restrictions, the configuration output frequency from the nested sampling run, and the requested database size.

> [!NOTE]
> ### Required Python modules
> - ASE
> - MPI4Py
> - Numpy
>   
> It is important that the number of files to search can be equally divided by the thread number. Due to the amount of data in a nested sampling output file, an unbalanced search scheme has not been implemented.


## Running the code
To use this code you provide: 
- [ -i ] A regular expression to identify all the trajectory files to search
- [ -s ] The size of the database
- [ -o ] The name of the database

Optionally, if you have calculated the temperature of the configurations, you can restrict the search to a given temperature range by providing:
- [ -lt ] A minimum temperature value
- [ -ut ] A maximum temperature value

## Ideal examples would be: <br />
To generate a database of 500 configurations in a file called "new_db.extxyz", by searching for files that match the expression "\*.traj.\*.extxyz" of which I expect there to be 20*n files: <br />
`mpirun -np 20 python3 database_builder.py -i "*.traj.*.extxyz" -s 500 -o "new_db.extxyz"`<br />

And to do the same again but restrict the search to only configurations that have temperature between 100 and 10,000 k<br />
`mpirun -np 20 python3 database_builder.py -i "*.traj.*.extxyz" -s 500 -o "new_db.extxyz" -lt 100 -ut 10000`

<hr />

# temperature_calculator.py
Assuming you have run ns_analyse after a completed nested sampling run, and want to predict the temperature of the outputted configurations, you can use this function to do that.<br /> 
Note that the predictions are just predictions and so the accuracy is subject to the noise of a nested sampling run.<br />
Also keep in mind that the temperature is only reliable within the temperature range used when you ran ns_analyse. Temperatures outside of the ns_analyse range will be predicted through linear extrapolation and so you may notice -np.inf or +np.inf at very low enthalpy or high enthalpy values respectively. To get an actual temperature value you need only to expand the temperature range of your ns_analyse call.<br />

> [!NOTE]
> ### Required Python modules
> - ASE
> - MPI4Py
> - Numpy
> - Scipy
> - Pandas
> 
> It is important that the number of trajectory files to search can be equally divided by the thread number. Due to the amount of data in a nested sampling output file, an unbalanced search scheme has not been implemented.

To use this code you provide: 
- [ -i ] A regular expression to identify all the trajectory files to search
- [ -d ] A file with the data outputted by ns_analyse

## Ideal examples would be: <br />
To predict the temperature of configurations in files matching the expression "\*.traj.\*.extxyz", of which I expect there to be 24*n files, using the analysis data from ns_analyse stored in analysis_data.dat: <br />
`mpirun -np 24 python3 temperature_calculator.py -i "*.traj.*.extxyz" -d "analysis_data.dat"`<br />

<hr />

# pymatnest_traj_concat.py
Assuming you have run nested sampling and have not outputted every structure but now you want look at structure specific properties like the XRD. You can use this function to concatenate all the nested sampling trajectories into one file that is in iteration order.<br /> 
Note that the zeroth iteration in Pymatnest is not outputted so the first iteration is n where n is the output frequency from your nested sampling run.<br />

> [!NOTE]
> ### Required Python modules
> - ASE
> - MPI4Py
> - Numpy
> 
> It is important that the number of trajectory files to search can be equally divided by the thread number. Due to the amount of data in a nested sampling output file, an unbalanced sorting scheme has not been implemented.

To use this code you provide: 
- [ -i ] A regular expression to identify all the trajectory files to search
- [ -o ] The name of the outputted concatenated traj files

## Ideal examples would be: <br />
To combine the configurations in files matching the expression "\*.traj.\*.extxyz", of which I expect there to be 24*n files, and put these into a file called combined_files.extxyz: <br />
`mpirun -np 24 python3 pymatnest_traj_concat.py -i "*.traj.*.extxyz" -o "combined_files.extxyz"`<br />

<hr />

# pymatnest_xrd.py
Assuming you have run nested sampling, and combined all the files using pymatnest_traj_concat.py, you can use this function to calculate the XRD of each configuration to eventually calculate the temperature weighted XRD.<br /> 

> [!NOTE]
> ### Required Python modules
> - ASE
> - MPI4Py
> - Numpy
> - [Dans_Diffraction](https://github.com/DanPorter/Dans_Diffraction)
> 
> It is important that the traj files be concatenated and iteration ordered to make any further calculations easier.
> ***The first row of the outputted array will be the twotheta array

To use this code you provide: 
- [ -i ] The name of the concatenated traj files in iteration order
- [ -o ] The prefix of the file to store the outputted XRD arrays***

## Ideal examples would be: <br />
To calculate the XRD of all configurations stored in the concatenated file "combined_files.extxyz" and store the arrays in XRD_data.npy, having access to a 40 core node: <br />
`mpirun -np 40 python3 pymatnest_xrd.py -i "combined_files.extxyz" -o "XRD_data"`<br />

<hr />

# part_f_xrd.py
Assuming you have run nested sampling, and combined all the files using pymatnest_traj_concat.py, and calculated the XRD of every configuration, you can use this function to calculate the temperature dependent XRD to analyse the crystal structures at each temperature.<br /> 

> [!NOTE]
> ### Required Python modules
> - ASE
> - MPI4Py
> - Numpy
> - Scipy
> 
> It is important that the traj files be concatenated and iteration ordered
> ***The first row of the outputted array will be the twotheta array

To use this code you provide: 
- [ -i ] The name of the concatenated traj files in iteration order
- [ -k ] The number of walkers used in the nested sampling
- [ -xrd ] The name of the file with the the xrd arrays
- [ -ti ] The initial temperature to weight from
- [ -tf ] The final temperature to weight to
- [ -dt ] The step size of the temperature range defined by -ti and -tf
- [ -o ] The prefix of the file to store the outputted temperature weighted XRD arrays***

## Ideal examples would be: <br />
To calculate the temperature weighted XRD from a nested sampling run conducted using 1200 walkers, every 2K from 200K to 400K, after calculating and storing all XRDs in a file called XRD_data.npy, and outputting these arrays into a file called t_weighted_XRD.npy, having access to a 40 core node: <br />
`mpirun -np 40 python3 part_f_xrd.py -i "combined_files.extxyz" -k 1200 -xrd XRD_data.npy -ti 200 -tf 400 -dt 2 -o "t_weighted_XRD"`<br />

<hr />

# acetest.jl
Assuming you have run nested sampling, and have an existing ACE potential with a committee of potentials that you want to use to calculate the uncertainty of the nested sampling data.<br /> 

> [!NOTE]
> ### Required Julia modules
> - [ACEpotentials](https://github.com/ACEsuit/ACEregistry)
> - JLD2
> - Glob
> - ArgParse

To use this code you provide: 
- [ -i ] A regular expression to identify all the trajectory files to evaluate
- [ -m ] The ACE model, stored as a .jld2 format

## Ideal examples would be: <br />
To predict the uncertainty of configurations in files matching the expression "\*.traj.\*.extxyz", using the ACE model called ace_model.jld2, having access to a 40 core node: <br />
`julia -p 40 --project=. ./acetest.jl -i "*.traj.*.extxyz" -m "ace_model.jld2"`<br />

<hr />

# acetest_T.jl
Assuming you have run acetest.jl and calculated the predicted temperature of the configurations you can use this function to calculate the temperature weighted uncertainty.<br /> 

> [!NOTE]
> ### Required Julia modules
> - [ACEpotentials](https://github.com/ACEsuit/ACEregistry)
> - Glob
> - ArgParse

To use this code you provide: 
- [ -i ] A regular expression to identify all the trajectory files to evaluate

## Ideal examples would be: <br />
To predict the temperature weighted uncertainty of configurations in files matching the expression "\*.traj.\*.extxyz", having access to a 40 core node: <br />
`julia -p 40 --project=. ./acetest_T.jl -i "*.traj.*.extxyz"`<br />

<hr />

# db_builder_sigma.py
Assuming you have calculated the temperature weighted uncertainties of your trajectory files, you can use this function to pick the top N configs within a set temperature range.<br /> 

> [!NOTE]
> ### Required Python modules
> - ASE
> - MPI4Py
> - Numpy
> 
> It is important that the number of files to search can be equally divided by the thread number. Due to the amount of data in a nested sampling output file, an unbalanced search scheme has not been implemented.

To use this code you provide: 
- [ -i ] Regex to identify the trajectory files to look through
- [ -o ] The name of the file to output the selected configs to
- [ -s ] The number of configs to select

Optionally, if you have calculated the temperature of the configurations, you can restrict the search to a given temperature range by providing:
- [ -lt ] The lower temperature limit
- [ -ut ] The upper temperture limit

## Ideal examples would be: <br />
To generate a database of 200 configurations in a file called "uq_db.extxyz", by searching for files that match the expression "\*.traj.\*.extxyz" of which I expect there to be 24*n files: <br />
`mpirun -np 24 python3 db_builder_sigma.py -i "*.traj.*.extxyz" -s 200 -o "uq_db.extxyz"`<br />

And to do the same again but restrict the search to only configurations that have temperature between 100 and 5,000 k<br />
`mpirun -np 24 python3 database_builder.py -i "*.traj.*.extxyz" -s 200 -o "uq_db.extxyz" -lt 100 -ut 5000`

<hr />

# acefit.jl
Assuming you have a database, this is a simple example of the ACE fitting Julia code.<br /> 

> [!NOTE]
> ### Required Julia modules
> - [ACEpotentials](https://github.com/ACEsuit/ACEregistry)
> - JLD2
> - Suppressor
>
> Due to the number of parameters and complexity of the program you must modify most of the parameters in the file for your specific system. This should be used as a bare-bones example of the ACE fitting procedure.

To use this code you provide: 
- Arg1: An integer value for the ACE order
- Arg2: An integer value for the ACE degree
- Arg3: A floating point number for the percentage of test configurations

## Ideal examples would be: <br />
To fit a model with order 3 degree 7 with 20% of the database being for testing and having access to a 40 core node: <br />
`julia -p 40 --project=. ./acefit.jl 3 7 0.2`<br />
