# Run to convert the metadata for the ocean data into csv files that Spark can load

import numpy as np
import csv

# for the CFSRO dataset, maps level numbers from the input netcdf files into actual level depth in meters
depthLookupTable = {
        5: 10,
        15: 10,
        25: 10,
        35: 10,
        45: 10,
        55: 10,
        65: 10,
        75: 10,
        85: 10,
        95: 10,
        105: 10,
        115: 10,
        125: 10,
        135: 10,
        145: 10,
        155: 10,
        165: 10,
        175: 10,
        185: 10,
        195: 10,
        205: 10,
        215: 10,
        225: 11.5,
        238: 18.5,
        262: 32.5,
        303: 52,
        366: 78,
        459: 109,
        584: 144,
        747: 182.5,
        949: 223,
        1193: 265,
        1479: 307,
        1807: 347.5,
        2174: 386,
        2579: 421,
        3016: 452,
        3483: 478,
        3972: 497.5,
        4478: 512
}

#outDir = 'testOutputs'
outDir = 'output'
metadataFname = outDir + "/oceanMetadata.npz" # the file containing all the metadata collected during the conversion process
observedLatFname = outDir + "/observedLatitudes.csv" # latitude of the measurements on each row of the matrix
observedLonFname = outDir + "/observedLongitudes.csv" # longitude of the measurements on each row of the matrix
observedLevelFname = outDir + "/observedLevelIndices.csv" # level indicator of the measurements on each row of the matrix
#observedDepthFname = outDir + "/observedDepths.csv" # depths of the measurements on each row of the matrix
observedLocationsFname = outDir + "/observedLocations.lst" # for each row of the matrix, indicates the corresponding grid point in the original 3D grid (flattened to a vector) of measurements
latListFname = outDir + "/latList.lst" # the values of latitude sampled to form the original 3D grid of measurements
lonListFname = outDir + "/lonList.lst" # the values of longitude sampled to form the original 3D grid of measurements
depthListFname = outDir + "/depthList.lst" # the values of depths sampled to form the original 3D grid of measurements
dateListFname = outDir + "/columnDates.lst" # the date/time for each column of the matrix

metadata = np.load(metadataFname)
recordedLats = metadata["observedLatCoords"]
recordedLons = metadata["observedLonCoords"]
recordedLevelIndices = metadata["observedLevelNumbers"]
#recordedLevelIndices =  metadata["observedLevelDepths"]

# convert the level indices to actual depths in meters
#observedLevelDepths = map(lambda levelIdx: depthLookupTable[int(levelIdx)], recordedLevelIndices)

def strLine(number):
    return str(number) + "\n"

with open(dateListFname, 'w') as fout:
    fout.writelines( map( lambda str: str + "\n", map("".join, metadata["timeStamps"]) ) )

with open(latListFname, 'w') as fout:
    fout.writelines( map(strLine, metadata["latList"]) )

with open(lonListFname, 'w') as fout:
    fout.writelines( map(strLine, metadata["lonList"]) )

with open(depthListFname, 'w') as fout:
    fout.writelines( map(strLine, map(lambda levelIdx: depthLookupTable[levelIdx], metadata["depthList"])) )

with open(observedLocationsFname, 'w') as fout:
    fout.writelines( map(strLine, metadata["observedLocations"]) )

with open(observedLevelFname, 'w') as csvfile:
    fieldnames = ['rowidx', 'levelindex']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    for (idx, val) in enumerate(recordedLevelIndices):
        writer.writerow({'rowidx' : idx, 'levelindex' : val})

# with open(observedDepthFname, 'w') as csvfile:
#     fieldnames = ['rowidx', 'thickness']
#     writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
# 
#     for (idx, val) in enumerate(observedLevelDepths):
#         writer.writerow({'rowidx' : idx, 'thickness' : val})

with open(observedLatFname, 'w') as csvfile:
    fieldnames = ['rowidx', 'latitude']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    for (idx, val) in enumerate(recordedLats):
        writer.writerow({'rowidx' : idx, 'latitude' : val})

with open(observedLonFname, 'w') as csvfile:
    fieldnames = ['rowidx', 'longitude']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    for (idx, val) in enumerate(recordedLons):
        writer.writerow({'rowidx' : idx, 'longitude' : val})

