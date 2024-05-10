### This repository consists of code so that you can form databases of configurations from the output of nested sampling runs.

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
