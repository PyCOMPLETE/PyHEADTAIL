from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
mpi_size = comm.Get_size()

# work_size = 127 # arbitrary prime number

work_size = 100
work = np.zeros(work_size)

base = work_size / mpi_size
leftover = work_size % mpi_size
sizes = np.ones(mpi_size) * base
sizes[:leftover] += 1
offsets = np.zeros(mpi_size)
offsets[1:] = np.cumsum(sizes)[:-1]
start = offsets[rank]
local_size = sizes[rank]
work_local = np.arange(start, start+local_size, dtype=np.float64)

print("local work: {} in rank {}".format(work_local,rank))

comm.Allgatherv(work_local, [work, sizes, offsets, MPI.DOUBLE])

summe = np.empty(1, dtype=np.float64)
comm.Allreduce(np.sum(work_local), summe, op=MPI.SUM)
print("work {} vs {} in rank {}".format(np.sum(work), summe,rank))
