
import numpy as np
import scipy as sp
import matplotlib as mpl
import matplotlib.pyplot as plt

numheaderlines = 23
numheaderlines_10mm20ms_and_spec = 22
numdatapoints = 1024
peakrangestart = 207
peakrangeend = 308

filename_100_20 = "0327pm_100mm_20ms.csv"
filename_10_20 = "0327pm_10mm_20ms.csv"
filename_10_40 = "0327pm_10mm_40ms.csv"
filename_10_60 = "0327pm_10mm_60ms.csv"
filename_10_100 = "0327pm_10mm_100ms.csv"
filename_10_200 = "0327pm_10mm_200ms.csv"
filename_35_20 = "0327pm_35mm_20ms.csv"
filename_73_20 = "0327pm_73mm_20ms.csv"
filename_195_20 = "0327pm_195mm_20ms.csv"

def func_readfile(filename, numskip):
    file = open(filename, "r")
    for i in range(numskip):
        header = file.readline()
        print(header)
    counts = np.empty(numdatapoints)
    for i in range(numdatapoints):
        line = file.readline()
        line = line.strip()
        columns = line.split(",")
        counts[i] = int(columns[2])
    file.close()
    return counts

channel = np.arange(0.0, 1024.0, 1.0)
counts_100_20 = func_readfile(filename_100_20, numheaderlines)
counts_10_20 = func_readfile(filename_10_20, numheaderlines_10mm20ms_and_spec)
counts_10_40 = func_readfile(filename_10_40, numheaderlines)
counts_10_60 = func_readfile(filename_10_60, numheaderlines)
counts_10_100 = func_readfile(filename_10_100, numheaderlines)
counts_10_200 = func_readfile(filename_10_200, numheaderlines)
counts_35_20 = func_readfile(filename_35_20, numheaderlines)
counts_73_20 = func_readfile(filename_73_20, numheaderlines)
counts_195_20 = func_readfile(filename_195_20, numheaderlines)

plt.plot(channel, counts_100_20)
plt.plot(channel, counts_10_20)
plt.plot(channel, counts_10_40)
plt.plot(channel, counts_10_60)
plt.plot(channel, counts_10_100)
plt.plot(channel, counts_10_200)
plt.plot(channel, counts_35_20)
plt.plot(channel, counts_73_20)
plt.plot(channel, counts_195_20)
plt.show()
