# Loads the complete CFSRO converted dataset and metadata, and extracts the portion corresponding to the measurements above the thermocline
#
# from the original 2.2TB ocean dataset, this will form an about 1.7TB dataset. Keep in mind can only write 4GB at a time with h5py, so plan to write about 4GB from each process. Ignore the issue of overloading the IO buses of each physical node, that means 
# we can have 32 writer processes per node => need 28 nodes => 896 writer processes
# that gives "exceed job memory limit" errors, so going to go up to 16 nodes
# srun -c 2 -n 896 -u python-mpi -u ./extractThermocline.py

from mpi4py import MPI
import h5py
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
mpiInfo = MPI.Info.Create()
numProcs = comm.Get_size()
procsList = np.arange(numProcs)

levelsToKeep = np.arange(9, 40) # keep all but the first 9 levels (the code assumes the levels kept form a contiguous submatrix)

fnameIn = "/global/cscratch1/sd/gittens/conversion-code/CFSRO_conversion/output/ocean.h5"
metadataFnameIn = "/global/cscratch1/sd/gittens/conversion-code/CFSRO_conversion/output/oceanMetadata.npz"

fnameOut = "/global/cscratch1/sd/gittens/conversion-code/CFSRO_conversion/output/thermoclineOcean.h5"
metadataFnameOut =  "/global/cscratch1/sd/gittens/conversion-code/CFSRO_conversion/output/thermoclineOceanMetadata.npz"

# load the original data and identify rows to keep

fin = h5py.File(fnameIn, "r")
rowsIn = fin["rows"]
metadataIn = np.load(metadataFnameIn)

keepIndices = np.nonzero(np.in1d(metadataIn["observedLevelNumbers"], levelsToKeep))[0]
numRows = len(keepIndices)
numCols = rowsIn.shape[1]

# create the output file and dataset efficiently

propfaid = h5py.h5p.create(h5py.h5p.FILE_ACCESS)
propfaid.set_fapl_mpio(comm, mpiInfo)
fid = h5py.h5f.create(fnameOut, flags=h5py.h5f.ACC_TRUNC, fapl=propfaid)
fout = h5py.File(fid)

spaceid = h5py.h5s.create_simple((numRows, numCols))
plist = h5py.h5p.create(h5py.h5p.DATASET_CREATE)
plist.set_fill_time(h5py.h5d.FILL_TIME_NEVER)
datasetid = h5py.h5d.create(fout.id, "rows,", h5py.h5t.NATIVE_DOUBLE, spaceid, plist)
rowsOut = h5py.Dataset(datasetid)

# write out the metadata (note we only retain some of the metadata)
# this will write out the observedLocations in the original (not subsetted!) dataset,
# so keep this in mind when using this field to unfold from the matrix to the 3d grid
if rank == 0:
    observedLonCoords = metadataIn["observedLonCoords"][keepIndices]
    observedLatCoords = metadataIn["observedLatCoords"][keepIndices]
    observedLevelNumbers = metadataIn["observedLevelNumbers"][keepIndices]
    observedLocations = metadataIn["observedLocations"][keepIndices]
    np.savez(metadataFnameOut, observedLonCoords = observedLonCoords,
            observedLatCoords = observedLatCoords, observedLevelNumbers = observedLevelNumbers,
            observedLocations = observedLocations)

# write out the data
littlePartitionSize = numRows/numProcs
bigPartitionSize = littlePartitionSize + 1
numLittlePartitions = numProcs - numRows % numProcs
numBigPartitions = numRows % numProcs

if (rank < numBigPartitions):
    startOutputRow = bigPartitionSize*rank
    endOutputRow = startOutputRow + bigPartitionSize
    startInputRow = keepIndices[0] + bigPartitionSize*rank
    endInputRow = startInputRow + bigPartitionSize
else:
    startOutputRow = bigPartitionSize*numBigPartitions + littlePartitionSize*(rank - numBigPartitions)
    endOutputRow = startOutputRow + littlePartitionSize
    startInputRow = keepIndices[0] + bigPartitionSize*numBigPartitions + littlePartitionSize*(rank - numBigPartitions) 
    endInputRow = startInputRow + littlePartitionSize

# not sure which is better output scheme: the direct write might be less efficient, but probably doesn't require memory use?
#rowsOut[startOutputRow:endOutputRow, :] = rowsIn[startInputRow:endInputRow, :]
tempMat = rowsIn[startInputRow:endInputRow, :]
rowsOut[startOutputRow:endOutputRow, :] = tempMat

fin.close()
fout.close()
