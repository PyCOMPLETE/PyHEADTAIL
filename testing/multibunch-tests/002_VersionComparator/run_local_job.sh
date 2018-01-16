#!/bin/bash
# A shell which is used to launch PyHEADTAIL simulations in a MPI environment 
# from a Jupyter Notebook

nnodes=$1
job_id=$2
copy_path=$3

# Path to the installed environment
# (same as ENVPATH in the file install_mpi_environment.sh)
ENVIRONMENT=/home/jani/environments/mpi

source $ENVIRONMENT/virtualenvs/py2.7/bin/activate
which python
$ENVIRONMENT/openmpi/bin/mpirun -np $nnodes python main.py $job_id
sleep 1s
echo "Done"
