# Running on different temperatures

We will explain how to run the simulation on different temperatures step by step.
The project is placed inside: `/home/nirz/Mice_project`.

## step 1 - generating the dump.0 files: 

Inside the folder `./md_simulations` we have the folder `PRL-2017-PairEntropy-master` which is the original folder of pablo.
Our data is placed inside `./md_simulations/Aluminum/liquid_pure`, where one can see different folders for each temperature.
You can see that inside `plumed.dat` and `plumed.start.dat` we commented out the metadynamics operations.
Inside `in.setup` we changed the printing into `dump.0` file frequency.
Moreover, inside `start.lmp` we run until 20 minutes which generating around 350mb - 500mb of data.
To run the simulation we are using the command:
`mpirun -np 10 /home/nirz/lmp2021/src/lmp_g++_openmpi -in start.lmp`

## step 2: moving the dump.0 files to the right folder

Our program is reading the `dump.0` files from the folder: `./data/dump_files/Aluminum/pure_liquid/`. Therefore, we will have to copy the files to the right folder by using: `cp ./md_simulations/Aluminum/liquid_pure/900/dump.0 ./data/dump_files/Aluminum/pure_liquid/900/` 

## step 3: loading the data from dump.0 to data.h5 format

In order to save the data in a format we are familiar with, we will use the command: `python ./src/bin/load_data.py --T=900`.
This command will read `dump.0` file from `./data/dump_files/Aluminum/pure_liquid/900/` and save to the same folder `names.h5` and `data.h5`.

## step 4: generating the tensors of shape(10,10,10) with resolution of 20

We will generate the data for the neural net, therefore we will use the `lattice_saver.py` where we will declare the resolution (20) and also the shape of the boxes (10,10,10). We will run this code for each temperature by the command: `python ./src/mice/main/lattice_saver.py --T=900`.
This command will save inside `./data/dump_files/Aluminum/pure_liquid/900/20/10_10_10/data.h5`the tensors for training.

## step 5: running the neural net

Now everything is ready and we have to run the neural net on our data.
We will run `python ./src/mice/main/sweep_func.py --T=900` which will start to learn the mi for the number of boxes: 20 and the (10,10,10)