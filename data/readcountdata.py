
import math
import numpy as np
import scipy as sp
import matplotlib as mpl
from scipy import stats as st
from scipy.stats import poisson
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
    counts = np.empty(numdatapoints)
    for i in range(numdatapoints):
        line = file.readline()
        line = line.strip()
        columns = line.split(",")
        counts[i] = int(columns[2])
    file.close()
    return counts

channel = np.arange(0.0, 1024.0, 1.0)
time_20 = channel * 0.02
time_40 = channel * 0.04
time_60 = channel * 0.06
time_100 = channel * 0.1
time_200 = channel * 0.2
counts_100_20 = func_readfile(filename_100_20, numheaderlines)
counts_10_20 = func_readfile(filename_10_20, numheaderlines_10mm20ms_and_spec)
counts_10_40 = func_readfile(filename_10_40, numheaderlines)
counts_10_60 = func_readfile(filename_10_60, numheaderlines)
counts_10_100 = func_readfile(filename_10_100, numheaderlines)
counts_10_200 = func_readfile(filename_10_200, numheaderlines)
counts_35_20 = func_readfile(filename_35_20, numheaderlines)
counts_73_20 = func_readfile(filename_73_20, numheaderlines)
counts_195_20 = func_readfile(filename_195_20, numheaderlines)

#plt.plot(channel, counts_100_20, marker=".", linestyle="")
#plt.plot(channel, counts_10_20, marker=".", linestyle="")
#plt.plot(channel, counts_10_40, marker=".", linestyle="")
#plt.plot(channel, counts_10_60, marker=".", linestyle="")
#plt.plot(channel, counts_10_100, marker=".", linestyle="")
#plt.plot(channel, counts_10_200, marker=".", linestyle="")
#plt.plot(channel, counts_35_20, marker=".", linestyle="")
#plt.plot(channel, counts_73_20, marker=".", linestyle="")
#plt.plot(channel, counts_195_20, marker=".", linestyle="")

#plt.plot(time_20, counts_100_20 / 0.02, marker=".", linestyle="")
#plt.plot(time_20, counts_10_20 / 0.02, marker=".", linestyle="")
#plt.plot(time_40, counts_10_40 / 0.04, marker=".", linestyle="")
#plt.plot(time_60, counts_10_60 / 0.06, marker=".", linestyle="")
#plt.plot(time_100, counts_10_100 / 0.1, marker=".", linestyle="")
#plt.plot(time_200, counts_10_200 / 0.2, marker=".", linestyle="")
#plt.plot(time_20, counts_35_20 / 0.02, marker=".", linestyle="")
#plt.plot(time_20, counts_73_20 / 0.02, marker=".", linestyle="")
#plt.plot(time_20, counts_195_20 / 0.02, marker=".", linestyle="")

stats_100_20 = st.describe(counts_100_20)
stats_10_20 = st.describe(counts_10_20)
stats_10_40 = st.describe(counts_10_40)
stats_10_60 = st.describe(counts_10_60)
stats_10_100 = st.describe(counts_10_100)
stats_10_200 = st.describe(counts_10_200)
stats_35_20 = st.describe(counts_35_20)
stats_73_20 = st.describe(counts_73_20)
stats_195_20 = st.describe(counts_195_20)
print(stats_100_20)
print(stats_10_20)
print(stats_10_40)
print(stats_10_60)
print(stats_10_100)
print(stats_10_200)
print(stats_35_20)
print(stats_73_20)
print(stats_195_20)

def func_cum_mean(counts_data):
    current_sum = 0
    cum_mean = np.empty(numdatapoints)
    cum_std_dev = np.empty(numdatapoints)
    cum_mean_unc = np.empty(numdatapoints)
    #current_sum = current_sum + counts_data[0]
    #cum_mean[0] = current_sum / (0 + 1)
    #cum_std_dev[0] = np.sqrt(np.power((counts_data[0] - cum_mean[0]), 2) / (0 + 1 - 1))
    #cum_mean_unc[0] = cum_std_dev[0] / np.sqrt(1)
    current_sum = counts_data[0]
    cum_mean[0] = counts_data[0]
    cum_std_dev[0] = 0
    cum_mean_unc[0] = 0
    for i in range(1, numdatapoints):
        current_sum = current_sum + counts_data[i]
        cum_mean[i] = current_sum / (i + 1)
        sum_mean_diff_squ = 0
        for j in range(0, (i + 1)):
            sum_mean_diff_squ = sum_mean_diff_squ + np.power((counts_data[j] - cum_mean[i]), 2)
        cum_std_dev[i] = np.sqrt(sum_mean_diff_squ / i)
        cum_mean_unc[i] = cum_std_dev[i] / np.sqrt(i + 1)
    return cum_mean, cum_std_dev, cum_mean_unc

#cum_mean_100_20, cum_std_dev_100_20, cum_mean_unc_100_20 = func_cum_mean(counts_100_20)
#cum_mean_10_20,  cum_std_dev_10_20,  cum_mean_unc_10_20  = func_cum_mean(counts_10_20)
#cum_mean_10_40,  cum_std_dev_10_40,  cum_mean_unc_10_40  = func_cum_mean(counts_10_40)
#cum_mean_10_60,  cum_std_dev_10_60,  cum_mean_unc_10_60  = func_cum_mean(counts_10_60)
#cum_mean_10_100, cum_std_dev_10_100, cum_mean_unc_10_100 = func_cum_mean(counts_10_100)
#cum_mean_10_200, cum_std_dev_10_200, cum_mean_unc_10_200 = func_cum_mean(counts_10_200)
#cum_mean_35_20,  cum_std_dev_35_20,  cum_mean_unc_35_20  = func_cum_mean(counts_35_20)
#cum_mean_73_20,  cum_std_dev_73_20,  cum_mean_unc_73_20  = func_cum_mean(counts_73_20)
#cum_mean_195_20, cum_std_dev_195_20, cum_mean_unc_195_20 = func_cum_mean(counts_195_20)

def func_final_mean(counts_data):
    mean = sum(counts_data) / numdatapoints
    sum_mean_diff_squ = 0
    for i in range(numdatapoints):
        sum_mean_diff_squ = sum_mean_diff_squ + np.power((counts_data[i] - mean), 2)
    variance = sum_mean_diff_squ / (numdatapoints - 1)
    std_dev = np.sqrt(variance)
    mean_unc = std_dev / np.sqrt(numdatapoints)
    return mean, std_dev, mean_unc

mean_100_20, std_dev_100_20, mean_unc_100_20 = func_final_mean(counts_100_20)
mean_10_20,  std_dev_10_20,  mean_unc_10_20  = func_final_mean(counts_10_20)
mean_10_40,  std_dev_10_40,  mean_unc_10_40  = func_final_mean(counts_10_40)
mean_10_60,  std_dev_10_60,  mean_unc_10_60  = func_final_mean(counts_10_60)
mean_10_100, std_dev_10_100, mean_unc_10_100 = func_final_mean(counts_10_100)
mean_10_200, std_dev_10_200, mean_unc_10_200 = func_final_mean(counts_10_200)
mean_35_20,  std_dev_35_20,  mean_unc_35_20  = func_final_mean(counts_35_20)
mean_73_20,  std_dev_73_20,  mean_unc_73_20  = func_final_mean(counts_73_20)
mean_195_20, std_dev_195_20, mean_unc_195_20 = func_final_mean(counts_195_20)

print(mean_100_20, std_dev_100_20, mean_unc_100_20)
print(mean_10_20,  std_dev_10_20,  mean_unc_10_20)
print(mean_10_40,  std_dev_10_40,  mean_unc_10_40)
print(mean_10_60,  std_dev_10_60,  mean_unc_10_60)
print(mean_10_100, std_dev_10_100, mean_unc_10_100)
print(mean_10_200, std_dev_10_200, mean_unc_10_200)
print(mean_35_20,  std_dev_35_20,  mean_unc_35_20)
print(mean_73_20,  std_dev_73_20,  mean_unc_73_20)
print(mean_195_20, std_dev_195_20, mean_unc_195_20)

#plt.plot(channel, cum_mean_100_20)
#plt.plot(channel, cum_mean_10_20)
#plt.plot(channel, cum_mean_10_40)
#plt.plot(channel, cum_mean_10_60)
#plt.plot(channel, cum_mean_10_100)
#plt.plot(channel, cum_mean_10_200)
#plt.plot(channel, cum_mean_35_20)
#plt.plot(channel, cum_mean_73_20)
#plt.plot(channel, cum_mean_195_20)

#plt.plot(channel, cum_std_dev_100_20)
#plt.plot(channel, cum_std_dev_10_20)
#plt.plot(channel, cum_std_dev_10_40)
#plt.plot(channel, cum_std_dev_10_60)
#plt.plot(channel, cum_std_dev_10_100)
#plt.plot(channel, cum_std_dev_10_200)
#plt.plot(channel, cum_std_dev_35_20)
#plt.plot(channel, cum_std_dev_73_20)
#plt.plot(channel, cum_std_dev_195_20)

#plt.plot(channel, cum_mean_unc_100_20)
#plt.plot(channel, cum_mean_unc_10_20)
#plt.plot(channel, cum_mean_unc_10_40)
#plt.plot(channel, cum_mean_unc_10_60)
#plt.plot(channel, cum_mean_unc_10_100)
#plt.plot(channel, cum_mean_unc_10_200)
#plt.plot(channel, cum_mean_unc_35_20)
#plt.plot(channel, cum_mean_unc_73_20)
#plt.plot(channel, cum_mean_unc_195_20)

#plt.errorbar(channel, cum_mean_100_20, cum_mean_unc_100_20)
#plt.errorbar(channel, cum_mean_10_20, cum_mean_unc_10_20)
#plt.errorbar(channel, cum_mean_10_40, cum_mean_unc_10_40)
#plt.errorbar(channel, cum_mean_10_60, cum_mean_unc_10_60)
#plt.errorbar(channel, cum_mean_10_100, cum_mean_unc_10_100)
#plt.errorbar(channel, cum_mean_10_200, cum_mean_unc_10_200)
#plt.errorbar(channel, cum_mean_35_20, cum_mean_unc_35_20)
#plt.errorbar(channel, cum_mean_73_20, cum_mean_unc_73_20)
#plt.errorbar(channel, cum_mean_195_20, cum_mean_unc_195_20)



"""
plt.plot(time_20, counts_10_20, marker=".", linestyle="", markersize = 2, label = "Counts per Time Interval")
plt.errorbar(time_20, cum_mean_10_20, cum_mean_unc_10_20, label = "Cumulative Mean with Uncertainty")
plt.plot([0, 1024 * 0.02], [std_dev_10_20, std_dev_10_20], label = "Final Standard Deviation = " + str(std_dev_10_20))
plt.plot([0, 1024 * 0.02], [mean_unc_10_20, mean_unc_10_20], label = "Final Uncertainty of the Mean = " + str(mean_unc_10_20))
plt.plot([0, 1024 * 0.02], [mean_10_20, mean_10_20], label = "Final mean = " + str(mean_10_20))
plt.plot(time_20, cum_std_dev_10_20, label = "Cumulative Standard Deviation")
plt.plot(time_20, cum_mean_unc_10_20, label = "Cumulative Uncertainty of the Mean")
plt.xlabel("Time (s)")
plt.ylabel("Counts")
plt.legend(loc = "upper right")
"""

"""
plt.plot(time_40, counts_10_40, marker=".", linestyle="", markersize = 2, label = "Counts per Time Interval")
plt.errorbar(time_40, cum_mean_10_40, cum_mean_unc_10_40, label = "Cumulative Mean with Uncertainty")
plt.plot([0, 1024 * 0.04], [std_dev_10_40, std_dev_10_40], label = "Final Standard Deviation = " + str(std_dev_10_40))
plt.plot([0, 1024 * 0.04], [mean_unc_10_40, mean_unc_10_40], label = "Final Uncertainty of the Mean = " + str(mean_unc_10_40))
plt.plot([0, 1024 * 0.04], [mean_10_40, mean_10_40], label = "Final mean = " + str(mean_10_40))
plt.plot(time_40, cum_std_dev_10_40, label = "Cumulative Standard Deviation")
plt.plot(time_40, cum_mean_unc_10_40, label = "Cumulative Uncertainty of the Mean")
plt.xlabel("Time (s)")
plt.ylabel("Counts")
plt.legend(loc = "upper right")
"""

"""
plt.plot(time_60, counts_10_60, marker=".", linestyle="", markersize = 2, label = "Counts per Time Interval")
plt.errorbar(time_60, cum_mean_10_60, cum_mean_unc_10_60, label = "Cumulative Mean with Uncertainty")
plt.plot([0, 1024 * 0.06], [std_dev_10_60, std_dev_10_60], label = "Final Standard Deviation = " + str(std_dev_10_60))
plt.plot([0, 1024 * 0.06], [mean_unc_10_60, mean_unc_10_60], label = "Final Uncertainty of the Mean = " + str(mean_unc_10_60))
plt.plot([0, 1024 * 0.06], [mean_10_60, mean_10_60], label = "Final mean = " + str(mean_10_60))
plt.plot(time_60, cum_std_dev_10_60, label = "Cumulative Standard Deviation")
plt.plot(time_60, cum_mean_unc_10_60, label = "Cumulative Uncertainty of the Mean")
plt.xlabel("Time (s)")
plt.ylabel("Counts")
plt.legend(loc = "upper right")
"""

"""
plt.plot(time_100, counts_10_100, marker=".", linestyle="", markersize = 2, label = "Counts per Time Interval")
plt.errorbar(time_100, cum_mean_10_100, cum_mean_unc_10_100, label = "Cumulative Mean with Uncertainty")
plt.plot([0, 1024 * 0.1], [std_dev_10_100, std_dev_10_100], label = "Final Standard Deviation = " + str(std_dev_10_100))
plt.plot([0, 1024 * 0.1], [mean_unc_10_100, mean_unc_10_100], label = "Final Uncertainty of the Mean = " + str(mean_unc_10_100))
plt.plot([0, 1024 * 0.1], [mean_10_100, mean_10_100], label = "Final mean = " + str(mean_10_100))
plt.plot(time_100, cum_std_dev_10_100, label = "Cumulative Standard Deviation")
plt.plot(time_100, cum_mean_unc_10_100, label = "Cumulative Uncertainty of the Mean")
plt.xlabel("Time (s)")
plt.ylabel("Counts")
plt.legend(loc = "center right")
"""

"""
plt.plot(time_200, counts_10_200, marker=".", linestyle="", markersize = 2, label = "Counts per Time Interval")
plt.errorbar(time_200, cum_mean_10_200, cum_mean_unc_10_200, label = "Cumulative Mean with Uncertainty")
plt.plot([0, 1024 * 0.2], [std_dev_10_200, std_dev_10_200], label = "Final Standard Deviation = " + str(std_dev_10_200))
plt.plot([0, 1024 * 0.2], [mean_unc_10_200, mean_unc_10_200], label = "Final Uncertainty of the Mean = " + str(mean_unc_10_200))
plt.plot([0, 1024 * 0.2], [mean_10_200, mean_10_200], label = "Final mean = " + str(mean_10_200))
plt.plot(time_200, cum_std_dev_10_200, label = "Cumulative Standard Deviation")
plt.plot(time_200, cum_mean_unc_10_200, label = "Cumulative Uncertainty of the Mean")
plt.xlabel("Time (s)")
plt.ylabel("Counts")
plt.legend(loc = "center right")
"""

"""
plt.plot(time_20, counts_35_20, marker=".", linestyle="", markersize = 2, label = "Counts per Time Interval")
plt.errorbar(time_20, cum_mean_35_20, cum_mean_unc_35_20, label = "Cumulative Mean with Uncertainty")
plt.plot([0, 1024 * 0.02], [std_dev_35_20, std_dev_35_20], label = "Final Standard Deviation = " + str(std_dev_35_20))
plt.plot([0, 1024 * 0.02], [mean_unc_35_20, mean_unc_35_20], label = "Final Uncertainty of the Mean = " + str(mean_unc_35_20))
plt.plot([0, 1024 * 0.02], [mean_35_20, mean_35_20], label = "Final mean = " + str(mean_35_20))
plt.plot(time_20, cum_std_dev_35_20, label = "Cumulative Standard Deviation")
plt.plot(time_20, cum_mean_unc_35_20, label = "Cumulative Uncertainty of the Mean")
plt.xlabel("Time (s)")
plt.ylabel("Counts")
plt.legend(loc = "upper right")
"""

"""
plt.plot(time_20, counts_73_20, marker=".", linestyle="", markersize = 2, label = "Counts per Time Interval")
plt.errorbar(time_20, cum_mean_73_20, cum_mean_unc_73_20, label = "Cumulative Mean with Uncertainty")
plt.plot([0, 1024 * 0.02], [std_dev_73_20, std_dev_73_20], label = "Final Standard Deviation = " + str(std_dev_73_20))
plt.plot([0, 1024 * 0.02], [mean_unc_73_20, mean_unc_73_20], label = "Final Uncertainty of the Mean = " + str(mean_unc_73_20))
plt.plot([0, 1024 * 0.02], [mean_73_20, mean_73_20], label = "Final mean = " + str(mean_73_20))
plt.plot(time_20, cum_std_dev_73_20, label = "Cumulative Standard Deviation")
plt.plot(time_20, cum_mean_unc_73_20, label = "Cumulative Uncertainty of the Mean")
plt.xlabel("Time (s)")
plt.ylabel("Counts")
plt.legend(loc = "upper right")
"""

"""
plt.plot(time_20, counts_195_20, marker=".", linestyle="", markersize = 2, label = "Counts per Time Interval")
plt.errorbar(time_20, cum_mean_195_20, cum_mean_unc_195_20, label = "Cumulative Mean with Uncertainty")
plt.plot([0, 1024 * 0.02], [std_dev_195_20, std_dev_195_20], label = "Final Standard Deviation = " + str(std_dev_195_20))
plt.plot([0, 1024 * 0.02], [mean_unc_195_20, mean_unc_195_20], label = "Final Uncertainty of the Mean = " + str(mean_unc_195_20))
plt.plot([0, 1024 * 0.02], [mean_195_20, mean_195_20], label = "Final mean = " + str(mean_195_20))
plt.plot(time_20, cum_std_dev_195_20, label = "Cumulative Standard Deviation")
plt.plot(time_20, cum_mean_unc_195_20, label = "Cumulative Uncertainty of the Mean")
plt.annotate("Data point with greatest deviation from mean", (7.8, 5.95))
plt.xlabel("Time (s)")
plt.ylabel("Counts")
plt.legend(loc = "upper right")
"""



histogram_size = 400
xline = np.arange(0.0, histogram_size, 1.0)

def func_histogram(counts_data):
    histogram = np.zeros(histogram_size)
    histogram_unc = np.zeros(histogram_size)
    freq_histogram = np.zeros(histogram_size)
    freq_histogram_unc = np.zeros(histogram_size)
    for count in counts_data:
        histogram[int(count)] = histogram[int(count)] + 1
    for i in range(histogram_size):
        histogram_unc[i] = np.sqrt(histogram[i])
    freq_histogram = histogram / numdatapoints
    freq_histogram_unc = histogram_unc / numdatapoints
    return histogram, histogram_unc, freq_histogram, freq_histogram_unc

#histogram_100_20, histogram_unc_100_20, freq_histogram_100_20, freq_histogram_unc_100_20 = func_histogram(counts_100_20)
histogram_10_20,  histogram_unc_10_20,  freq_histogram_10_20,  freq_histogram_unc_10_20  = func_histogram(counts_10_20)
histogram_10_40,  histogram_unc_10_40,  freq_histogram_10_40,  freq_histogram_unc_10_40  = func_histogram(counts_10_40)
histogram_10_60,  histogram_unc_10_60,  freq_histogram_10_60,  freq_histogram_unc_10_60  = func_histogram(counts_10_60)
histogram_10_100, histogram_unc_10_100, freq_histogram_10_100, freq_histogram_unc_10_100 = func_histogram(counts_10_100)
histogram_10_200, histogram_unc_10_200, freq_histogram_10_200, freq_histogram_unc_10_200 = func_histogram(counts_10_200)
histogram_35_20,  histogram_unc_35_20,  freq_histogram_35_20,  freq_histogram_unc_35_20  = func_histogram(counts_35_20)
histogram_73_20,  histogram_unc_73_20,  freq_histogram_73_20,  freq_histogram_unc_73_20  = func_histogram(counts_73_20)
histogram_195_20, histogram_unc_195_20, freq_histogram_195_20, freq_histogram_unc_195_20 = func_histogram(counts_195_20)

#def func_poisson(xdata, mu):
#    ydata = np.empty(len(xdata))
#    for i in range(len(xdata)):
#        ydata[i] = (np.power(mu, xdata[i]) / math.factorial(int(xdata[i]))) * np.power(np.e, (-1 * mu)) 
#    return ydata

#poisson_mu_1   = func_poisson(rdata, 1.0)
#poisson_mu_4   = func_poisson(rdata, 4.0)
#poisson_mu_10  = func_poisson(rdata, 10.0)
#poisson_mu_30  = func_poisson(rdata, 30.0)

#poisson_mu_60  = func_poisson(rdata, 60.0)
#poisson_mu_120 = func_poisson(rdata, 120.0)
#poisson_mu_240 = func_poisson(rdata, 240.0)

#poisson_mu_90  = func_poisson(rdata, 90.0)
#poisson_mu_150 = func_poisson(rdata, 150.0)
#poisson_mu_300 = func_poisson(rdata, 300.0)

#data_mean_100_20 =   2.494140625
data_mean_10_20  =  26.83203125
data_mean_10_40  =  54.984375
data_mean_10_60  =  82.5517578125
data_mean_10_100 = 137.35546875
data_mean_10_200 = 273.92578125
data_mean_35_20  =  10.0615234375
data_mean_73_20  =   3.9970703125
data_mean_195_20 =   0.9951171875

#spectrum_count_100_20 =   2.5752530633990416
spectrum_count_10_20  =  27.131107966369772
spectrum_count_10_40  =  54.262215932739544
spectrum_count_10_60  =  81.3933238991093
spectrum_count_10_100 = 135.65553983184884
spectrum_count_10_200 = 271.3110796636977
spectrum_count_35_20  =  10.09818321093737
spectrum_count_73_20  =   3.972459038230985
spectrum_count_195_20 =   0.9907025129222067

#poisson_100_20 = poisson.pmf(xline, 2.494140625)
poisson_10_20  = poisson.pmf(xline, spectrum_count_10_20)
poisson_10_40  = poisson.pmf(xline, spectrum_count_10_40)
poisson_10_60  = poisson.pmf(xline, spectrum_count_10_60)
poisson_10_100 = poisson.pmf(xline, spectrum_count_10_100)
poisson_10_200 = poisson.pmf(xline, spectrum_count_10_200)
poisson_35_20  = poisson.pmf(xline, spectrum_count_35_20)
poisson_73_20  = poisson.pmf(xline, spectrum_count_73_20)
poisson_195_20 = poisson.pmf(xline, spectrum_count_195_20)

#plt.bar(xline, freq_histogram_100_20, yerr = freq_histogram_unc_100_20)
#plt.plot(xline, poisson_100_20)
"""
plt.bar(xline, freq_histogram_10_20,  yerr = freq_histogram_unc_10_20, label = "Histogram of data from 10mm source with 20ms intervals")
plt.plot(xline, poisson_10_20, label = "Poisson distibution with mean = " + str(spectrum_count_10_20))

plt.bar(xline, freq_histogram_10_40,  yerr = freq_histogram_unc_10_40, label = "Histogram of data from 10mm source with 40ms intervals")
plt.plot(xline, poisson_10_40, label = "Poisson distibution with mean = " + str(spectrum_count_10_40))

plt.bar(xline, freq_histogram_10_60,  yerr = freq_histogram_unc_10_60, label = "Histogram of data from 10mm source with 60ms intervals")
plt.plot(xline, poisson_10_60, label = "Poisson distibution with mean = " + str(spectrum_count_10_60))

plt.bar(xline, freq_histogram_10_100, yerr = freq_histogram_unc_10_100, label = "Histogram of data from 10mm source with 100ms intervals")
plt.plot(xline, poisson_10_100, label = "Poisson distibution with mean = " + str(spectrum_count_10_100))

plt.bar(xline, freq_histogram_10_200, yerr = freq_histogram_unc_10_200, label = "Histogram of data from 10mm source with 200ms intervals")
plt.plot(xline, poisson_10_200, label = "Poisson distibution with mean = " + str(spectrum_count_10_200))

plt.bar(xline, freq_histogram_35_20,  yerr = freq_histogram_unc_35_20, label = "Histogram of data from 35mm source with 20ms intervals")
plt.plot(xline, poisson_35_20, label = "Poisson distibution with mean = " + str(spectrum_count_35_20))

plt.bar(xline, freq_histogram_73_20,  yerr = freq_histogram_unc_73_20, label = "Histogram of data from 73mm source with 20ms intervals")
plt.plot(xline, poisson_73_20, label = "Poisson distibution with mean = " + str(spectrum_count_73_20))

plt.bar(xline, freq_histogram_195_20, yerr = freq_histogram_unc_195_20, label = "Histogram of data from 195mm source with 20ms intervals")
plt.plot(xline, poisson_195_20, label = "Poisson distibution with mean = " + str(spectrum_count_195_20))

plt.xlabel("Counts")
plt.ylabel("Frequency")
"""

#plt.bar(xline[10:45], freq_histogram_10_20[10:45],  yerr = freq_histogram_unc_10_20[10:45], label = "Histogram of data from 10mm source with 20ms intervals")
#plt.plot(xline[10:45], poisson_10_20[10:45], label = "Poisson distibution with mean = " + str(spectrum_count_10_20))

#plt.bar(xline[30:80], freq_histogram_10_40[30:80],  yerr = freq_histogram_unc_10_40[30:80], label = "Histogram of data from 10mm source with 40ms intervals")
#plt.plot(xline[30:80], poisson_10_40[30:80], label = "Poisson distibution with mean = " + str(spectrum_count_10_40))

#plt.bar(xline[50:120], freq_histogram_10_60[50:120],  yerr = freq_histogram_unc_10_60[50:120], label = "Histogram of data from 10mm source with 60ms intervals")
#plt.plot(xline[50:120], poisson_10_60[50:120], label = "Poisson distibution with mean = " + str(spectrum_count_10_60))

#plt.bar(xline[100:180], freq_histogram_10_100[100:180], yerr = freq_histogram_unc_10_100[100:180], label = "Histogram of data from 10mm source with 100ms intervals")
#plt.plot(xline[100:180], poisson_10_100[100:180], label = "Poisson distibution with mean = " + str(spectrum_count_10_100))

#plt.bar(xline[220:340], freq_histogram_10_200[220:340], yerr = freq_histogram_unc_10_200[220:340], label = "Histogram of data from 10mm source with 200ms intervals")
#plt.plot(xline[220:340], poisson_10_200[220:340], label = "Poisson distibution with mean = " + str(spectrum_count_10_200))

#plt.bar(xline[0:25], freq_histogram_35_20[0:25],  yerr = freq_histogram_unc_35_20[0:25], label = "Histogram of data from 35mm source with 20ms intervals")
#plt.plot(xline[0:25], poisson_35_20[0:25], label = "Poisson distibution with mean = " + str(spectrum_count_35_20))

#plt.bar(xline[0:15], freq_histogram_73_20[0:15],  yerr = freq_histogram_unc_73_20[0:15], label = "Histogram of data from 73mm source with 20ms intervals")
#plt.plot(xline[0:15], poisson_73_20[0:15], label = "Poisson distibution with mean = " + str(spectrum_count_73_20))

#plt.bar(xline[0:10], freq_histogram_195_20[0:10], yerr = freq_histogram_unc_195_20[0:10], label = "Histogram of data from 195mm source with 20ms intervals")
#plt.plot(xline[0:10], poisson_195_20[0:10], label = "Poisson distibution with mean = " + str(spectrum_count_195_20))

#plt.xlabel("Counts")
#plt.ylabel("Frequency")
#plt.legend()

plt.show()
