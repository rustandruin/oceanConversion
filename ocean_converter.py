# TODO:
#  -- processes should be able to handle uneven number of files
#  -- should check that all time steps have the same mask of unobserved entries before proceeding
#  -- should write out observation mask
#  -- should write out column to date mapping

# production run settings:
# salloc -N 30 -t 150 -p regular --qos=premium
# bash
# module load h5py-parallel mpi4py netcdf4-python
# srun -c 3 -n 300 -u python-mpi -u ./fname.py variablename
#
# Optimizations:
# - turn off fill at allocation in hdf5
# - use an output directory that has 140 OSTs
# - turn on alignment with striping (use output from lfs getstripe to set alignment)
#
# Note: apparently, the number of aggregator processes is set to the number of OSTs

from mpi4py import MPI
from netCDF4 import Dataset
import h5py
import numpy as np
from os import listdir
from os.path import isfile, join
import time
import math, sys

rank = MPI.COMM_WORLD.Get_rank()
mpi_info = MPI.Info.Create()
numProcs = MPI.COMM_WORLD.Get_size()

numProcessesPerNode = 10

procslist = np.arange(numProcs)

def status(message, ranks=procslist):
    if rank in ranks:
        print "%s, process %d: %s" % (time.asctime(time.localtime()), rank, message)

def report(message):
    status(message, [0])

def reportbarrier(message):
    MPI.COMM_WORLD.Barrier()
    report(message)

# maps the chunkidx (0...numlevdivs=numWriters) to the rank of the process that should write it out
def chunkidxToWriter(chunkidx):
   return (chunkidx*numProcessesPerNode)%numProcs # should always write from different nodes

datapath = "/global/cscratch1/sd/nrcavana/CFSR_OCEAN/"
metadataFname = "/global/cscratch1/sd/gittens/conversion-code/ocean_conversion/ocean_metadata.npz"
filelist = [fname for fname in listdir(datapath) if fname.endswith(".nc")]
varname = "POT_L160_Avg_1"
timevarname = "ref_date_time"

# FOR TESTING ONLY: REMOVE WHEN RUNNING FINAL JOB
#filelist = [fname for fname in filelist[:1000]]

report("Using %d processes" % numProcs)
report("Writing variable %s" % varname)
report("Found %d input files, starting to open" % len(filelist))

# open all the files associated with this process and keep handles around (avoid metadata costs)
myfiles = [fname for (index, fname) in enumerate(filelist) if (index % numProcs == rank)]
numLocalFiles = len(myfiles)
myhandles = [None]*numLocalFiles
numLocalTimeSlices = [] # some files have different number of time slices
for (idx, fname) in enumerate(myfiles):
    myhandles[idx] = Dataset(join(datapath, fname), "r") 
    numLocalTimeSlices.append(myhandles[idx][varname].shape[0])

# send the number of time slices in each file to the root
timeSlicesPerProcess = MPI.COMM_WORLD.gather(numLocalTimeSlices, root=0)
numColsPerProcess = map(np.sum, timeSlicesPerProcess)

reportbarrier("Finished opening all files")

numlevels = 40
numlats = 360
numlongs = 720

verifymask = False
if verifymask:
    # Optionally check that all the missing masks are the same
    reportbarrier("Verifying that the missing mask is the same for all observations")
    reportbarrier("Checking equality of masks on each process")
    missingLocations = set(np.nonzero(myhandles[0][varname][0, ...].mask.flatten())[0])
    for (fhidx, fh) in enumerate(myhandles):
        for timeslice in xrange(numLocalTimeSlices[fhidx]):
            curMissingLocations = set(np.nonzero(fh[varname][timeslice, ...].mask.flatten())[0])
            if curMissingLocations != missingLocations:
                status("The missing masks do not match for some of my files")
                sys.exit(1)

    # would like to use reduce, but apparently this does a gather first, which
    # would cause an issue (why isn't tree reduce implemented?)
    reportbarrier("Checking equality of masks across processes")
    for sender in xrange(1, numProcs):
        if rank == sender:
            MPI.COMM_WORLD.send(missingLocations, dest=0, tag=0)
        elif rank == 0:
            curMissingLocations = MPI.COMM_WORLD.recv(source=sender, tag=0)
            if curMissingLocations != missingLocations:
                status("The missing masks do not match for some of my files")
                sys.exit(1)
else:
    # Assuming the missing masks are all the same
    missingLocations = set(np.nonzero(myhandles[0][varname][0, ...].mask.flatten())[0])
    
# Store the indices of missing values for future use
# and the time-date stamps corresponding to each column of the matrix
reportbarrier("Storing the locations of missing values and the time-date stamps for each column of the output matrix")
timeStampsList = [None]*numLocalFiles
for (idx, fh) in enumerate(myhandles):
    timeStampsList[idx] = fh[timevarname][:]
timeStamps = np.concatenate(timeStampsList)
timeStamps = MPI.COMM_WORLD.gather(timeStamps, root=0)
if rank == 0:
    timeStamps = np.concatenate(timeStamps)
    numpy.savez(metadataFname, missingLocations=np.array(missingLocations), 
                               timeStamps=timeStamps)

relevantLocations = np.nonzero(np.logical_not(myhandles[0][varname][0, ...].mask.flatten()))[0] # indices of the locations that were observed
numdivs = 2 # number of chunks to write out each level of observations in
numRows = len(relevantLocations)
numCols = np.sum(numColsPerProcess)

numWriters = numlevdivs 
foutname = varname

reportbarrier("Creating output file and dataset")
#Ask for alignment with the stripe size (use lfs getstripe on target directory to determine)
propfaid = h5py.h5p.create(h5py.h5p.FILE_ACCESS)
propfaid.set_fapl_mpio(MPI.COMM_WORLD, mpi_info)
#propfaid.set_alignment(1024, 1024*1024)
#driver='mpio', comm=MPI.COMM_WORLD

fid = h5py.h5f.create(join(datapath, foutname + ".h5"), flags=h5py.h5f.ACC_TRUNC, fapl=propfaid)
fout = h5py.File(fid)

# Don't use filling 
spaceid = h5py.h5s.create_simple((numRows, numCols))
plist = h5py.h5p.create(h5py.h5p.DATASET_CREATE)
plist.set_fill_time(h5py.h5d.FILL_TIME_NEVER)
datasetid = h5py.h5d.create(fout.id, "rows", h5py.h5t.NATIVE_FLOAT, spaceid, plist)
rows = h5py.Dataset(datasetid)
reportbarrier("Finished creating output file and dataset")

localcolumncount = np.sum(numLocalTimeSlices)
curlevdata = np.empty((numlats*numlongs, localcolumncount), dtype=np.float32)
chunktotransfer = np.empty((rowChunkSize*localcolumncount,), dtype=np.float32)

listwriter = map(chunkidxToWriter, np.arange(numWriters))
if rank in listwriter:
    collectedchunk = np.ascontiguousarray(np.empty((numCols*rowChunkSize,), \
            dtype=np.float32))
    chunktowrite = np.ascontiguousarray(np.empty((rowChunkSize, numCols), \
            dtype=np.float32))
else:
    collectedchunk = None
curlevdatatemp=np.ascontiguousarray(np.zeros((numlats*numlongs*numtimeslices), \
            dtype=np.float32))
#currowoffset = 0
for (varidx,curvar) in enumerate(varnames): 
    reportbarrier("Writing variable %d/%d: %s" % (varidx + 1, numvars, curvar))

    for curlev in np.arange(numlevels):

        # load the data for this level from my files
        reportbarrier("Loading data for level %d/%d" % (curlev + 1, numlevels))
        for (fhidx, fh) in enumerate(myhandles):
            if fh[curvar].shape[0] < numtimeslices and fh[curvar].shape[0] >0:
                status("File %s has only %d timesteps for variable %s, simply repeating the first timestep" % (myfiles[fhidx], fh[curvar].shape[0], curvar))
                for idx in np.arange(numtimeslices):
                    curlevdatatemp[numlats*numlongs*idx:numlats*numlongs*(idx+1)] = fh[curvar][0, curlev, ...].flatten()
                curlevdata[:, fhidx*numtimeslices: (fhidx + 1)*numtimeslices]=curlevdatatemp.reshape(numlats*numlongs, numtimeslices)
                curlevdatatemp=np.ascontiguousarray(np.zeros((numlats*numlongs*numtimeslices), dtype=np.float32)) 
            elif fh[curvar].shape[0] ==0:
                status("File %s has only %d timesteps for variable %s, simply repeating the first timestep" % (myfiles[fhidx], fh[curvar].shape[0], curvar))
                curlevdata[:, fhidx*numtimeslices: (fhidx + 1)*numtimeslices] = \
                        curlevdatatemp.reshape(numlats*numlongs, numtimeslices)
            else:
                curlevdata[:, fhidx*numtimeslices: (fhidx + 1)*numtimeslices] = \
                    fh[curvar][:, curlev, ...].reshape(numlats*numlongs, numtimeslices)
        reportbarrier("Done loading data for this level")
        
        # write out this level in several chunks of rows
        reportbarrier("Gathering data for this level from processes to writers")
        for chunkidx in np.arange(numlevdivs):
            startrow = chunkidx*rowChunkSize
            endrow = startrow + rowChunkSize
            chunktotransfer[:] = curlevdata[startrow:endrow, :].flatten()
            MPI.COMM_WORLD.Gather(chunktotransfer, collectedchunk, root = chunkidxToWriter(chunkidx))
        reportbarrier("Done gathering")
        reportbarrier("Writing data for this level on writers")
        for chunkidx in np.arange(numlevdivs):
            if rank == chunkidxToWriter(chunkidx):
                for processnum in np.arange(numProcs):
                    startcol = processnum*localcolumncount
                    endcol = (processnum+1)*localcolumncount
                    startidx = processnum*(localcolumncount * rowChunkSize)
                    endidx = (processnum + 1)*(localcolumncount *rowChunkSize)
                    chunktowrite[:, startcol:endcol] = np.reshape(collectedchunk[startidx:endidx], \
                            (rowChunkSize, localcolumncount))
                start_rows = varidx*numlevels*numlats*numlongs+curlev*numlats*numlongs+rowChunkSize*chunkidx
                end_rows = start_rows+rowChunkSize
                rows[start_rows:end_rows, :] = chunktowrite
        reportbarrier("Done writing")

for fh in myhandles:
    fh.close()
fout.close()

