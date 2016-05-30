# Run to convert the dumped metadata for the ocean data into csv files that Spark can load

import numpy as np
import csv

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

metadataFname = "output/oceanMetadata.npz"
outLatFname = "output/latitudes.csv"
outDepthFname = "output/depths.csv"

metadata = np.load(metadataFname)
recordedLats = metadata["observedLatCoords"]
recordedLevelDepths =  metadata["observedLevelDepths"]
outLevelDepths = map(lambda levelIdx: depthLookupTable[int(levelIdx)], recordedLevelDepths)

with open(outDepthFname, 'w') as csvfile:
    fieldnames = ['rowidx', 'thickness']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    for (idx, val) in enumerate(outLevelDepths):
        writer.writerow({'rowidx' : idx, 'thickness' : val})

with open(outLatFname, 'w') as csvfile:
    fieldnames = ['rowidx', 'latitude']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    for (idx, val) in enumerate(recordedLats):
        writer.writerow({'rowidx' : idx, 'latitude' : val})
