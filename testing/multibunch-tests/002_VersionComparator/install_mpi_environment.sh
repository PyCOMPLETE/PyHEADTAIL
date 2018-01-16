#!/bin/bash
#
# This script installs a python environment for multibunch PyHEADTAIL simulations.
# The script has been tested with CentOS 7 and Scientific Linux CERN 6 and based on the guide from:
# github.com/PyCOMPLETE/PyECLOUD/wiki/Setup-python-%28including-mpi4py%29-without-admin-rights
#
# @author J. Komppula, July 2017


# path where the environment will be installed
ENVPATH=/home/jani/environments/mpi

cd $ENVPATH

wget https://www.python.org/ftp/python/2.7.11/Python-2.7.11.tar.xz
mkdir python_src
mv Python-2.7.11.tar.xz python_src/
cd python_src
tar xvfJ Python-2.7.11.tar.xz
cd Python-2.7.11
mkdir $ENVPATH/python27
./configure --prefix=$ENVPATH/python27 --with-zlib
make
make install



cd $ENVPATH
cd python_src
curl -O https://pypi.python.org/packages/source/v/virtualenv/virtualenv-15.0.0.tar.gz
tar -zxvf virtualenv-15.0.0.tar.gz
cd virtualenv-15.0.0
$ENVPATH/python27/bin/python setup.py install
cd ..

cd $ENVPATH
mkdir virtualenvs
cd virtualenvs
$ENVPATH/python27/bin/virtualenv py2.7 --python=$ENVPATH/python27/bin/python

cd py2.7/bin
source ./activate

pip install numpy
pip install scipy
pip install cython
pip install ipython
pip install matplotlib


cd $ENVPATH/python_src
wget https://www.open-mpi.org/software/ompi/v1.10/downloads/openmpi-1.10.2.tar.bz2
tar jxf openmpi-1.10.2.tar.bz2
cd openmpi-1.10.2
./configure --prefix=$ENVPATH/openmpi
make all install

env MPICC=$ENVPATH/openmpi/bin/mpicc pip install mpi4py

cd $ENVPATH/python_src
wget https://support.hdfgroup.org/ftp/HDF5/current/src/hdf5-1.10.1.tar.bz2
tar jxf hdf5-1.10.1.tar.bz2
cd hdf5-1.10.1
CC=$ENVPATH/openmpi/bin/mpicc ./configure --enable-parallel --enable-shared
make
make install

cd $ENVPATH/python_src
wget https://pypi.python.org/packages/11/6b/32cee6f59e7a03ab7c60bb250caff63e2d20c33ebca47cf8c28f6a2d085c/h5py-2.7.0.tar.gz
tar xvzf h5py-2.7.0.tar.gz
cd h5py-2.7.0
CC=$ENVPATH/openmpi/bin/mpicc HDF5_MPI="ON" HDF5_DIR=$ENVPATH/python_src/hdf5-1.10.1/hdf5 pip install --no-binary=h5py h5py

