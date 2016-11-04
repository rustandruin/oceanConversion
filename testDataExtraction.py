# Example showing how to extract data from the ocean matrix and the corresponding metadata file
#
# to run the visualization bit, you'll need to SSH tunnel X , so make sure you have XQuartz installed and login w/ e.g.
# ssh -Y wss@cori.nersc.gov

import numpy as np
from h5py import File
from matplotlib import pyplot as plt

def extract_region(matrix, metadata, lats, lons, levelindices):
    """ 
    matrix -- the h5 dataset containing the ocean temperature data
    metadata --- the numpy dictionary containing the metadata information
    lats --- a list of the latitude values to limit to
    lons --- a list of the longitude values to limit to
    levelindices --- a list of the levels to limit to

    submat --- a submatrix of the input matrix, consisting of rows that correspond to 
    allowed (lat, lon, depth) coordinates
    lats -- a vector of the latitudes corresponding to each row of submat
    lons -- ditto for longitudes
    depths -- ditto for depths
    indices -- the indices of the rows in submat, in the original matrix
    """

    keepQ = np.logical_and(np.in1d(metadata["observedLatCoords"], lats), np.in1d(metadata["observedLonCoords"], lons))
    keepQ = np.logical_and(keepQ, np.in1d(metadata["observedLevelNumbers"], levelindices) )
    indices = np.nonzero(keepQ)[0]

    submat = matrix[indices, :]
    lats = metadata["observedLatCoords"][indices]
    lons = metadata["observedLonCoords"][indices]
    depths = metadata["observedLevelNumbers"][indices]

    return (submat, lats, lons, depths, indices)

def extract_depth(matrix, metadata, levelindex):
    """ Same as extract_region, but takes a single level index and returns all the data on that level """
    result = extract_region(matrix, metadata, metadata["latList"], metadata["lonList"], [levelindex])
    return (result[0], result[1], result[2], result[4])

def visualize_depth(matrix, metadata, levelindex):
    """ Visualize the average temperatures on one level of the ocean; note that we need to account for some grid points that don't have corresponding measurements;
    this is just for sanity checking that the other functions are returning the correct data """

    latList = metadata["latList"]
    lonList = metadata["lonList"]
    (temps, lats, lons, indices) = extract_depth(matrix, metadata, levelindex)
    avgTemps = np.mean(temps, axis=1)

    fillvalue = -1
    tempMat = fillvalue*np.ones((len(latList), len(lonList)))
    for idx in xrange(len(avgTemps)):
        rowCoord = np.where( np.in1d(latList, lats[idx]) )
        colCoord = np.where( np.in1d(lonList, lons[idx]) )
        tempMat[rowCoord, colCoord] = avgTemps[idx]

    return tempMat

fin = File("/global/cscratch1/sd/gittens/conversion-code/CFSRO_conversion/output/ocean.h5", "r")
md = np.load("/global/cscratch1/sd/gittens/conversion-code/CFSRO_conversion/output/oceanMetadata.npz")

# remember the data is upside down: level 0 is the bottom of the ocean, level 39 is the surface
result = visualize_depth(fin["rows"], md, 39)
plt.imshow(result, interpolation="nearest")
plt.show()
