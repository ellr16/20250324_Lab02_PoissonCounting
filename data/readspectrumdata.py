
import numpy as np
import scipy as sp
import matplotlib as mpl
from scipy import stats as st
import matplotlib.pyplot as plt

numheaderlines =   23
numheaderlines_10mm20ms_and_spec = 22
numdatapoints  = 1024
peakrangestart =  207
peakrangeend   =  307

filenameb    = "0327pm_background_spectrum_unrestricted.csv"
filename100u = "0327pm_100mm_spectrum_unrestricted.csv"
filename100r = "0327pm_100mm_spectrum_restricted.csv"
filename10   = "0327pm_10mm_spectrum.csv"
filename35   = "0327pm_35mm_spectrum.csv"
filename73   = "0327pm_73mm_spectrum.csv"
filename195  = "0327pm_195mm_spectrum.csv"

def func_readfile(filename, numskip):
    file = open(filename, "r")
    for i in range(13):
        header = file.readline()
    line = file.readline()
    line = line.strip()
    columns = line.split(",")
    elapsed_time = float(columns[1])
    for i in range(numskip - 14):
        header = file.readline()
    counts = np.empty(numdatapoints)
    for i in range(numdatapoints):
        line = file.readline()
        line = line.strip()
        columns = line.split(",")
        counts[i] = int(columns[2])
    file.close()
    return counts, elapsed_time

channel = np.arange(0.0, 1024.0, 1.0)

counts_b,    elapsed_time_b    = func_readfile(filenameb,    numheaderlines)
counts_100u, elapsed_time_100u = func_readfile(filename100u, numheaderlines)
counts_100r, elapsed_time_100r = func_readfile(filename100r, numheaderlines)
counts_10,   elapsed_time_10   = func_readfile(filename10,   numheaderlines_10mm20ms_and_spec)
counts_35,   elapsed_time_35   = func_readfile(filename35,   numheaderlines)
counts_73,   elapsed_time_73   = func_readfile(filename73,   numheaderlines)
counts_195,  elapsed_time_195  = func_readfile(filename195,  numheaderlines)

print(elapsed_time_b)
print(elapsed_time_100u)
print(elapsed_time_100r)
print(elapsed_time_10)
print(elapsed_time_35)
print(elapsed_time_73)
print(elapsed_time_195)

counts_per_time_b    = counts_b    / elapsed_time_b
counts_per_time_100u = counts_100u / elapsed_time_100u
counts_per_time_100r = counts_100r / elapsed_time_100r
counts_per_time_10   = counts_10   / elapsed_time_10
counts_per_time_35   = counts_35   / elapsed_time_35
counts_per_time_73   = counts_73   / elapsed_time_73
counts_per_time_195  = counts_195  / elapsed_time_195

clean_counts_100u = counts_per_time_100u - counts_per_time_b
clean_counts_100r = counts_per_time_100r - counts_per_time_b
clean_counts_10   = counts_per_time_10   - counts_per_time_b
clean_counts_35   = counts_per_time_35   - counts_per_time_b
clean_counts_73   = counts_per_time_73   - counts_per_time_b
clean_counts_195  = counts_per_time_195  - counts_per_time_b

sum_counts_b    = sum(counts_b[peakrangestart:peakrangeend])
sum_counts_100u = sum(counts_100u[peakrangestart:peakrangeend])
sum_counts_100r = sum(counts_100r[peakrangestart:peakrangeend])
sum_counts_10   = sum(counts_10[peakrangestart:peakrangeend])
sum_counts_35   = sum(counts_35[peakrangestart:peakrangeend])
sum_counts_73   = sum(counts_73[peakrangestart:peakrangeend])
sum_counts_195  = sum(counts_195[peakrangestart:peakrangeend])

print(sum_counts_b)
print(sum_counts_100u)
print(sum_counts_100r)
print(sum_counts_10)
print(sum_counts_35)
print(sum_counts_73)
print(sum_counts_195)

sum_counts_unc_b    = np.sqrt(sum_counts_b)
sum_counts_unc_100u = np.sqrt(sum_counts_100u)
sum_counts_unc_100r = np.sqrt(sum_counts_100r)
sum_counts_unc_10   = np.sqrt(sum_counts_10)
sum_counts_unc_35   = np.sqrt(sum_counts_35)
sum_counts_unc_73   = np.sqrt(sum_counts_73)
sum_counts_unc_195  = np.sqrt(sum_counts_195)

print(sum_counts_unc_b)
print(sum_counts_unc_100u)
print(sum_counts_unc_100r)
print(sum_counts_unc_10)
print(sum_counts_unc_35)
print(sum_counts_unc_73)
print(sum_counts_unc_195)

scaled_sum_100    = sum_counts_100r / elapsed_time_100r
scaled_sum_10     = sum_counts_10   / elapsed_time_10
scaled_sum_35     = sum_counts_35   / elapsed_time_35
scaled_sum_73     = sum_counts_73   / elapsed_time_73
scaled_sum_195    = sum_counts_195  / elapsed_time_195
scaled_sum_100_20 = sum_counts_100r * (0.02 / elapsed_time_100r)
scaled_sum_10_20  = sum_counts_10   * (0.02 / elapsed_time_10)
scaled_sum_10_40  = sum_counts_10   * (0.04 / elapsed_time_10)
scaled_sum_10_60  = sum_counts_10   * (0.06 / elapsed_time_10)
scaled_sum_10_100 = sum_counts_10   * (0.10 / elapsed_time_10)
scaled_sum_10_200 = sum_counts_10   * (0.20 / elapsed_time_10)
scaled_sum_35_20  = sum_counts_35   * (0.02 / elapsed_time_35)
scaled_sum_73_20  = sum_counts_73   * (0.02 / elapsed_time_73)
scaled_sum_195_20 = sum_counts_195  * (0.02 / elapsed_time_195)

print(scaled_sum_100)
print(scaled_sum_10)
print(scaled_sum_35)
print(scaled_sum_73)
print(scaled_sum_195)
print(scaled_sum_100_20)
print(scaled_sum_10_20)
print(scaled_sum_10_40)
print(scaled_sum_10_60)
print(scaled_sum_10_100)
print(scaled_sum_10_200)
print(scaled_sum_35_20)
print(scaled_sum_73_20)
print(scaled_sum_195_20)

#stats_b    = st.describe(counts_b[peakrangestart:peakrangeend])
#stats_100u = st.describe(counts_100u[peakrangestart:peakrangeend])
#stats_100r = st.describe(counts_100r[peakrangestart:peakrangeend])
#stats_10   = st.describe(counts_10[peakrangestart:peakrangeend])
#stats_35   = st.describe(counts_35[peakrangestart:peakrangeend])
#stats_73   = st.describe(counts_73[peakrangestart:peakrangeend])
#stats_195  = st.describe(counts_195[peakrangestart:peakrangeend])

#print(stats_b)
#print(stats_100u)
#print(stats_100r)
#print(stats_10)
#print(stats_35)
#print(stats_73)
#print(stats_195)

#plt.plot(channel, counts_per_time_b, marker=".", linestyle="", markersize = 1, markerfacecolor = "orange", markeredgecolor = "orange", label = "Decay Counts with no source")
#plt.plot(channel, counts_per_time_100u, marker=".", linestyle="", markersize = 1, markerfacecolor = "blue", markeredgecolor = "blue", label = "Decay Counts with Cs137 100mm away")
#plt.plot([peakrangestart, peakrangestart, peakrangeend, peakrangeend, peakrangestart], [0.0, 5.0, 5.0, 0.0, 0.0], color = "black", label = "Window for MCS mode")
#plt.annotate("0.662 MeV photopeak", (250,5.05))
#plt.xlabel("Channel Number")
#plt.ylabel("Counts per second")
#plt.title("Time Scaled Spectrum of the Cs137 Source at 100mm, and of the background")

#plt.plot(channel[peakrangestart:peakrangeend], counts_per_time_10[peakrangestart:peakrangeend],   marker=".", linestyle="", markersize = 2, label = "10mm")
#plt.plot(channel[peakrangestart:peakrangeend], counts_per_time_35[peakrangestart:peakrangeend],   marker=".", linestyle="", markersize = 2, label = "35mm")
#plt.plot(channel[peakrangestart:peakrangeend], counts_per_time_73[peakrangestart:peakrangeend],   marker=".", linestyle="", markersize = 2, label = "73mm")
#plt.plot(channel[peakrangestart:peakrangeend], counts_per_time_100r[peakrangestart:peakrangeend], marker=".", linestyle="", markersize = 2, label = "100mm")
#plt.plot(channel[peakrangestart:peakrangeend], counts_per_time_195[peakrangestart:peakrangeend],  marker=".", linestyle="", markersize = 2, label = "195mm")
#plt.xlabel("Channel Number")
#plt.ylabel("Counts per second")

#plt.legend()
#plt.show()
