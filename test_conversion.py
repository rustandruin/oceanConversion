# Code for testing that the ocean climate data has been converted correctly from the 
# multiple netcdf files into one large hdf5 file, and the latitudes recorded for each
# observation are correct
#
# To run: first do a 
#  module load python netcdf4-python h5py-parallel mpi4py
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
# original files and check that the latitudes for each observation were recorded accurately

numColSamples = 20

outputMat = h5py.File("ocean.h5", "r")["rows"]
numCols = outputMat.shape[1]
colIndices = np.sort(np.random.randint(numCols, size=numColSamples))
sampledCols = outputMat[:, colIndices]

baseDir = "/global/cscratch1/sd/nrcavana/CFSR_OCEAN/"
varname = "POT_L160_Avg_1"
metadata = np.load("oceanMetadata.npz")
timeOffsets = np.concatenate([np.array(item) for item in metadata["timeSliceOffsets"]])
fileNames = np.concatenate([np.array(item) for item in metadata["fileNames"]])
fNames = fileNames[colIndices]
timeSliceOffsets = timeOffsets[colIndices]
recordedLats = np.tile(metadata["observedLatCoords"], (20, 1)).transpose()

rawCols = np.empty_like(sampledCols)
rawLats = np.empty_like(sampledCols)
for (idx, fName) in enumerate(fNames):
    timeOffset = timeSliceOffsets[idx]
    rawFin = Dataset(baseDir + fName, "r")
    rawData = rawFin[varname][:].data
    rawMask = np.logical_not(rawFin[varname][:].mask)
    lonGrid, latGrid = np.meshgrid(rawFin["lon"][:], rawFin["lat"][:])
    latMesh = np.tile(latGrid, (rawData.shape[1],1,1))
    rawCols[:, idx] = rawData[timeOffset, rawMask[timeOffset, ...]]
    rawLats[:, idx] = latMesh[rawMask[timeOffset, ...]]
    rawFin.close()

# these should be zero on success
np.linalg.norm(rawCols - sampledCols)
np.linalg.norm(rawLats - recordedLats)
