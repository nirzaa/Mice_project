export LD_LIBRARY_PATH=/home/nirz/lmp2021/lib/plumed/plumed-2.7.1/src/lib
#cd /home/nirz/lmp2021/lib/plumed/plumed-2.7.1/
#source sourceme.sh
nohup /home/nirz/lmp2021/lib/plumed/plumed2/bin/plumed --no-mpi sum_hills --hills ../HILLS --stride 1000 --mintozero --bin 200,200 &
