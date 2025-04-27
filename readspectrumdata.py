
import numpy as np
import scipy as sp
import matplotlib as mpl
import matplotlib.pyplot as plt

numheaderlines = 23
numheaderlines_10mm20ms_and_spec = 22
numdatapoints = 1024
peakrangestart = 207
peakrangeend = 308

filenameb = "0327pm_background_spectrum_unrestricted.csv"
filename100u = "0327pm_100mm_spectrum_unrestricted.csv"
filename100r = "0327pm_100mm_spectrum_restricted.csv"
filename10 = "0327pm_10mm_spectrum.csv"
filename35 = "0327pm_35mm_spectrum.csv"
filename73 = "0327pm_73mm_spectrum.csv"
filename195 = "0327pm_195mm_spectrum.csv"

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
countsb = func_readfile(filenameb, numheaderlines)
counts100u = func_readfile(filename100u, numheaderlines)
counts100r = func_readfile(filename100r, numheaderlines)
counts10 = func_readfile(filename10, numheaderlines_10mm20ms_and_spec)
counts35 = func_readfile(filename35, numheaderlines)
counts73 = func_readfile(filename73, numheaderlines)
counts195 = func_readfile(filename195, numheaderlines)

fig = plt.figure()
plt.plot(channel,countsb)
plt.plot(channel,counts100u)
plt.plot(channel,counts100r)
plt.plot(channel,counts10)
plt.plot(channel,counts35)
plt.plot(channel,counts73)
plt.plot(channel,counts195)
#plt.plot(channel[peakrangestart:peakrangeend], countsb[peakrangestart:peakrangeend])
plt.show()
