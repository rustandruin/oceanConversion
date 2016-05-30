# Code for testing that the ocean climate data has been converted correctly from the 
# multiple netcdf files into one large hdf5 file, and the latitudes and level depths 
# recorded for each observation are correct
#
# To run: first do a 
#  module load python netcdf4-python h5py
# then start ipython and run this code interactively

import numpy as np
from netCDF4 import Dataset
import h5py

# FOR QUICK SANITY CHECKING DURING DEVELOPMENT, RUN THIS BLOCK OF CODE
fname = "/global/cscratch1/sd/nrcavana/CFSR_OCEAN/ocnh01.gdas.20071021-20071025.grb2.nc"
varname = "POT_L160_Avg_1"
rawFin = Dataset(fname, "r")
rawData = rawFin[varname][:].data
rawMask = np.logical_not(rawFin[varname][:].mask)
rawCol = rawData[0, rawMask[0, ...]]
outputMat = h5py.File("ocean.h5", "r")["rows"]
convertedCol = outputMat[:, 0]
np.linalg.norm(rawCol - convertedCol) # this should be zero on success

# FOR TESTING AFTER FULL CONVERSION, RUN THIS BLOCK OF CODE
# randomly sample columns and check for equality with the data from the corresponding
# original files and check that the latitudes and level depths for each observation were recorded accurately

numColSamples = 20

outputFname = "output/ocean.h5"
outputMat = h5py.File(outputFname, "r")["rows"]
numCols = outputMat.shape[1]
colIndices = np.sort(np.random.randint(numCols, size=numColSamples))
sampledCols = outputMat[:, colIndices]

baseDir = "/global/cscratch1/sd/nrcavana/CFSR_OCEAN/"
varname = "POT_L160_Avg_1"
metadataFname = "output/oceanMetadata.npz"
metadata = np.load(metadataFname)
timeOffsets = np.concatenate([np.array(item) for item in metadata["timeSliceOffsets"]])
fileNames = np.concatenate([np.array(item) for item in metadata["fileNames"]])
fNames = fileNames[colIndices]
timeSliceOffsets = timeOffsets[colIndices]
recordedLats = np.tile(metadata["observedLatCoords"], (numColSamples, 1)).transpose()
recordedLevelDepths =  np.tile(metadata["observedLevelDepths"], (numColSamples, 1)).transpose()

rawCols = np.empty_like(sampledCols)
rawLats = np.empty_like(sampledCols)
rawDepths = np.empty_like(sampledCols)
for (idx, fName) in enumerate(fNames):
    timeOffset = timeSliceOffsets[idx]
    rawFin = Dataset(baseDir + fName, "r")
    rawData = rawFin[varname][:].data
    rawMask = np.logical_not(rawFin[varname][:].mask)
    lonGrid, latGrid = np.meshgrid(rawFin["lon"][:], rawFin["lat"][:])
    latMesh = np.tile(latGrid, (rawData.shape[1],1,1))
    rawCols[:, idx] = rawData[timeOffset, rawMask[timeOffset, ...]]
    rawLats[:, idx] = latMesh[rawMask[timeOffset, ...]]

    levelDepths = rawFin["level0"]
    curLevelDepths = []
    for levNum in xrange(rawData.shape[1]):
        curLevelDepths = np.concatenate([curLevelDepths, [levelDepths[levNum]] * np.count_nonzero(rawMask[timeOffset, levNum, ...])])
    rawDepths[:, idx] = curLevelDepths
    rawFin.close()

# these should be zero on success
np.linalg.norm(rawCols - sampledCols)
np.linalg.norm(rawLats - recordedLats)
np.linalg.norm(rawDepths - recordedLevelDepths)
