# TODO:
#  use pre-allocated buffer matrices during read and write for efficiency
#
# Production run settings:
# salloc -N 20 -t 150 -p regular --qos=premium
# bash
# module load h5py-parallel mpi4py netcdf4-python python
# srun -c 3 -n 200 -u python-mpi -u ./simplified.py 

from mpi4py import MPI
from netCDF4 import Dataset
import h5py
import numpy as np
from os import listdir
from os.path import isfile, join
import time, math, sys


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
mpiInfo = MPI.Info.Create()
numProcs = comm.Get_size()
procsList = np.arange(numProcs)

### Helper functions and class

def status(message, ranks=procsList):
    """prints a message for each process in the ranks argument"""

    messageToSend = "%s, process %d: %s" % (time.asctime(time.localtime()), rank, message)
    messages = comm.gather(messageToSend, root=0)
    if rank == 0:
        for (idx, messageToPrint) in enumerate(messages):
            if idx in ranks:
                print messageToPrint

def report(message):
    """print a message from the root process"""
    status(message, [0])

def reportBarrier(message):
    """synchronize all processes, then print a message from the root process"""
    comm.Barrier()
    report(message)

def chunkIdxToWriter(chunkIdx):
    """maps the chunkIdx (0...numWriters) to the rank of the process that should write it out"""
    machineNumber = (chunkIdx % numNodes)
    offsetOnMachine = chunkIdx/numNodes
    return machineNumber*numProcessesPerNode + offsetOnMachine

def loadFiles(dir, varName, timevarName, procInfo):
    """gets the list of all filenames in the data directory, divides them among the processes, opens them, populates some metadata"""
    fileNameList = [fname for fname in listdir(dir) if fname.endswith(".nc")]
    if (DEBUGFLAG):
        fileNameList = fileNameList[:65]
        report("DEBUGGING! LIMITING NUMBER OF FILES CONVERTED")
    report("Found %d input files, starting to open" % len(fileNameList))

    procInfo.fileNameList = [fname for (index, fname) in enumerate(fileNameList) if (index % numProcs == rank)]
    procInfo.numFiles = len(procInfo.fileNameList)
    procInfo.fileHandleList = map( lambda fname: Dataset(join(dir, fname), "r"), procInfo.fileNameList)
    procInfo.numTimeSlices = map( lambda fh: fh[varName].shape[0], procInfo.fileHandleList)
    procInfo.numLocalCols = np.sum(procInfo.numTimeSlices)

    procInfo.colsPerProcess = np.empty((numProcs,), dtype=np.int)
    comm.Allgather(procInfo.numLocalCols, procInfo.colsPerProcess)
    procInfo.numCols = sum(procInfo.colsPerProcess)
    procInfo.outputColOffsets = np.hstack([[0], np.cumsum(procInfo.colsPerProcess[:-1])])
    procInfo.timeStamps = np.concatenate(map(lambda fh: fh[timevarName][:], procInfo.fileHandleList))

    # assumes the missing masks for observations are the same across timeslices
    procInfo.missingLocations = np.nonzero(procInfo.fileHandleList[0][varName][0, ...].mask.flatten())[0]
    procInfo.numRows = len(np.nonzero(np.logical_not(procInfo.fileHandleList[0][varName][0, ...].mask.flatten()))[0])

    return fileNameList

def writeMetadata(foutName, procInfo):
    """writes metadata for the converted dataset to a numpy file"""
    timeStamps = comm.gather(procInfo.timeStamps, root=0)
    if rank == 0:
        timeStamps = np.concatenate(timeStamps)
        np.savez(foutName, missingLocations=np.array(procInfo.missingLocations), timeStamps=timeStamps)

def createDataset(fnameOut, procInfo):
    propfaid = h5py.h5p.create(h5py.h5p.FILE_ACCESS)
    propfaid.set_fapl_mpio(comm, mpiInfo)
    fid = h5py.h5f.create(fnameOut, flags=h5py.h5f.ACC_TRUNC, fapl=propfaid)
    fout = h5py.File(fid)

    spaceid = h5py.h5s.create_simple((procInfo.numRows, procInfo.numCols))
    plist = h5py.h5p.create(h5py.h5p.DATASET_CREATE)
    plist.set_fill_time(h5py.h5d.FILL_TIME_NEVER)
    datasetid = h5py.h5d.create(fout.id, "rows", h5py.h5t.NATIVE_DOUBLE, spaceid, plist)
    rows = h5py.Dataset(datasetid)


    return (fout, rows)

def verifyMask(procInfo):
    """checks that the missing masks are the same for each set of observations"""
    reportBarrier("Verifying that the missing mask is the same for all observations")
    reportBarrier("... checking equality of masks on each process")
    missingLocations = set(np.nonzero(procInfo.fileHandleList[0][varname][0, ...].mask.flatten())[0])
    for (fhIdx, fh) in enumerate(procInfo.fileHandleList):
        for timeslice in xrange(procInfo.numTimeSlices[fhIdx]):
            curMissingLocations = set(np.nonzero(fh[varname][timeslice, ...].mask.flatten())[0])
            if curMissingLocations != missingLocations:
                status("The missing masks do not match for some of my files")
                sys.exit()
    # would like to use reduce, but apparently this does a gather first, which would cause and issue
    # (tree reduce is NOT implemented)
    reportBarrier("... checking equality of masks across processes")
    for sender in xrange(1, numProcs):
        if rank == sender:
            comm.send(missingLocations, dest=0, tag=0)
        elif rank == 0:
            curMissingLocations = comm.recv(source=sender, tag=0)
            if curMissingLocations != missingLocations:
                status("The missing masks do not match for some of the files")
                sys.exit(1)

def chunkIt(length, num):
    """breaks xrange(length) into num roughly equally sized pieces, returns arrays of start and end indices"""
    avg = length/float(num)
    startIndices = []
    endIndices = []
    last = 0.0

    while last < length:
        startIndices.append(int(last))
        endIndices.append(int(last + avg))
        last += avg

    return (startIndices, endIndices)

def loadLevel(procInfo, varname, numLats, numLongs, curLev):
    """loads all the observations from the files assigned to this process at level curLev, and returns as a 
    numObservations * (numColsInMyFiles) matrix"""

    curLevMask = np.logical_not(procInfo.fileHandleList[0][varname][0, curLev, ...].mask.flatten())
    procInfo.numObservations = len(np.nonzero(curLevMask)[0])
    curLevData = np.empty((procInfo.numObservations, procInfo.numLocalCols), dtype=np.float32)
    colOffset = 0
    for (fhidx, fh) in enumerate(procInfo.fileHandleList):
        numTimeSlices = procInfo.numTimeSlices[fhidx]
        observedMask = np.logical_not(fh[varname][:, curLev, ...].mask)
        observedValues = fh[varname][:, curLev, ...].data[observedMask]
        curLevData[:, colOffset:(colOffset + numTimeSlices)] = \
                observedValues.reshape(procInfo.numObservations, numTimeSlices)
        colOffset = colOffset + numTimeSlices

    return curLevData

# TODO: make return chunk a global variable
def gatherDataAtWriter(curLevData, procInfo):
    """Gathers all the row chunks of a given level of observations at the writer processes"""

    chunkStartIndices, chunkEndIndices = chunkIt(procInfo.numObservations, numWriters)
    chunkSizes = map(lambda chunkIdx: chunkEndIndices[chunkIdx] - chunkStartIndices[chunkIdx], xrange(numWriters))
    outputStartRows = np.hstack([[0], np.cumsum(chunkSizes)[:-1]])
    returnChunk = [-1]
    returnRowChunkSize = -1
    returnOutputRowOffset = -1
    for chunkIdx in xrange(numWriters):
        writerRank = chunkIdxToWriter(chunkIdx)
        curRowChunkSize = chunkSizes[chunkIdx]
        chunkToTransfer = curLevData[chunkStartIndices[chunkIdx]:chunkEndIndices[chunkIdx], :].flatten()

        processChunkSizes = curRowChunkSize*procInfo.colsPerProcess
        processChunkDisplacements = np.hstack([[0], np.cumsum(processChunkSizes[:-1])])
        collectedChunk = None
        if rank == writerRank:
            collectedChunk = np.empty((curRowChunkSize*procInfo.numCols), dtype=np.float32)
        comm.Gatherv(sendbuf=[chunkToTransfer, MPI.FLOAT], \
                     recvbuf=[collectedChunk, processChunkSizes, processChunkDisplacements, MPI.FLOAT], \
                     root=writerRank)
        if rank == writerRank:
            returnChunk = collectedChunk
            returnRowChunkSize = curRowChunkSize
            returnOutputRowOffset = outputStartRows[chunkIdx]

    #status("this chunk has %d rows and will be written starting at row %d" % \
    #            (returnRowChunkSize, returnOutputRowOffset), ranks=writersList)
    return (returnChunk, returnRowChunkSize, returnOutputRowOffset)

def writeOutputRowChunks(rowChunk, numRowsInChunk, outputRowOffset, rows, procInfo):
    """On writer processes, writes out the stored chunk of rows"""
    chunkStartIndices, chunkEndIndices = chunkIt(procInfo.numObservations, numWriters)
    for chunkIdx in xrange(numWriters):
        if rank == chunkIdxToWriter(chunkIdx):
            assert(len(rowChunk)== numRowsInChunk*procInfo.numCols)
            processChunkSizes = numRowsInChunk*procInfo.colsPerProcess
            processChunkDisplacements = np.hstack([[0], np.cumsum(processChunkSizes[:-1])])
            for processNum in np.arange(numProcs):
                outputStartCol = procInfo.outputColOffsets[processNum]
                outputEndCol = outputStartCol + procInfo.colsPerProcess[processNum]
                startChunkOffset = processChunkDisplacements[processNum]
                endChunkOffset = startChunkOffset + numRowsInChunk*procInfo.colsPerProcess[processNum]
                chunkToWrite[:numRowsInChunk, outputStartCol:outputEndCol] = np.reshape(rowChunk[startChunkOffset:endChunkOffset], \
                        (numRowsInChunk, procInfo.colsPerProcess[processNum]))
            startOutputRow = outputRowOffset
            endOutputRow = outputRowOffset + numRowsInChunk
            rows[startOutputRow:endOutputRow, :] = chunkToWrite[:numRowsInChunk, :]

class ProcessInformation(object):
    def __init__(self):
        pass

# Variables that should really be command-line settings
DEBUGFLAG = True
numNodes = 20
numProcessesPerNode = 1
numWriters = 20 # a good choice is one per physical node
verifyMaskQ = False
dataInPath = "/global/cscratch1/sd/nrcavana/CFSR_OCEAN/"
dataOutFname = "/global/cscratch1/sd/gittens/conversion-code/ocean_conversion/ocean.h5"
varname = "POT_L160_Avg_1"
timevarname = "ref_date_time"
metadataFnameOut = "/global/cscratch1/sd/gittens/conversion-code/ocean_conversion/oceanMetadata.npz"

numLevels = 40
numLats = 360
numLongs = 720

### Setup the processes for reading and writing

report("Using %d processes" % numProcs)
report("Writing variable %s " % varname)
procInfo = ProcessInformation()
fileNameList = loadFiles(dataInPath, varname, timevarname, procInfo)
report(fileNameList[0])

if (verifyMaskQ):
    verifyMask(procInfo)
writeMetadata(metadataFnameOut, procInfo)

report(" ".join(map(lambda idx: str(chunkIdxToWriter(idx)), xrange(numWriters))))
reportBarrier("Creating output file and dataset")
fout, rows = createDataset(dataOutFname, procInfo)
reportBarrier("Finished creating output file and dataset")

### Write the data to the output file
reportBarrier("Writing %s to file" % varname)

chunkToWrite = None
levelStartRow = 0
writersList = map(chunkIdxToWriter, xrange(numWriters))
for curLev in xrange(numLevels):
    reportBarrier("Loading data for level %d/%d" % (curLev + 1, numLevels))
    curLevData = loadLevel(procInfo, varname, numLats, numLongs, curLev)
    reportBarrier("Done loading data for this level")
    reportBarrier("There are %d observed grid points on this level" % procInfo.numObservations)

    # Preallocate buffers used during gathering and writing, to avoid reallocation costs
    chunkToWrite = np.empty((procInfo.numObservations/numWriters + 1,procInfo.numCols), dtype=np.float32)

    reportBarrier("Gathering data for this level from processes to writers")
    (curOutputRowChunk, curNumOutputRows, curOutputRowOffset) = gatherDataAtWriter(curLevData, procInfo)
    reportBarrier("Done gathering")

    reportBarrier("Writing data for this level on writers")
    writeOutputRowChunks(curOutputRowChunk, curNumOutputRows, levelStartRow + curOutputRowOffset, rows, procInfo)
    levelStartRow = levelStartRow + procInfo.numObservations

reportBarrier("Done writing")

# close the open files
map(lambda fh: fh.close(), procInfo.fileHandleList)
fout.close()







