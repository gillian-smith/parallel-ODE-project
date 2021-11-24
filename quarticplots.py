import numpy as np
import matplotlib.pyplot as plt

euler=1
sympeuler=0
heun=0
modsympeuler=0
ridc=1
RIDC_N = 10 # number of RIDC steps
h_list = 0.01*2**(np.array(range(2,9,2))/4)

file = open("error_std_lists_h0","rb")
error_std_lists_h0 = np.load(file)

file = open("error_std_lists_h1","rb")
error_std_lists_h1 = np.load(file)

file = open("error_std_lists_h2","rb")
error_std_lists_h2 = np.load(file)

file = open("error_std_lists_h3","rb")
error_std_lists_h3 = np.load(file)

# Each file contains, for a single h:
# Error_Euler ### Error_SymplecticEuler ### Error_Heun ### Error_modsympeuler ### Error_RIDC
# std_Euler   ### std_sympeuler         ### std_heun   ### std_modsympeuler   ### std_RIDC

# Euler = column 0
if euler==1:
	error_list_Euler = np.zeros(len(h_list))
	stddev_Euler = np.zeros(len(h_list))
	error_list_Euler[0] = error_std_lists_h0[0,0]
	stddev_Euler[0] = error_std_lists_h0[1,0]
	error_list_Euler[1] = error_std_lists_h1[0,0]
	stddev_Euler[1] = error_std_lists_h1[1,0]
	error_list_Euler[2] = error_std_lists_h2[0,0]
	stddev_Euler[2] = error_std_lists_h2[1,0]
	error_list_Euler[3] = error_std_lists_h3[0,0]
	stddev_Euler[3] = error_std_lists_h3[1,0]



# heun = column 1
if heun==1:
	error_list_Heun = np.zeros(len(h_list))
	stddev_Heun = np.zeros(len(h_list))
	error_list_Heun[0] = error_std_lists_h0[0,1]
	stddev_Heun[0] = error_std_lists_h0[1,1]
	error_list_Heun[1] = error_std_lists_h1[0,1]
	stddev_Heun[1] = error_std_lists_h1[1,1]
	error_list_Heun[2] = error_std_lists_h2[0,1]
	stddev_Heun[2] = error_std_lists_h2[1,1]
	error_list_Heun[3] = error_std_lists_h3[0,1]
	stddev_Heun[3] = error_std_lists_h3[1,1]



# RIDC = column 4
if ridc==1:
	error_list_RIDC = np.zeros(len(h_list))
	stddev_RIDC = np.zeros(len(h_list))
	error_list_RIDC[0] = error_std_lists_h0[0,2]
	stddev_RIDC[0] = error_std_lists_h0[1,2]
	error_list_RIDC[1] = error_std_lists_h1[0,2]
	stddev_RIDC[1] = error_std_lists_h1[1,2]
	error_list_RIDC[2] = error_std_lists_h2[0,2]
	stddev_RIDC[2] = error_std_lists_h2[1,2]
	error_list_RIDC[3] = error_std_lists_h3[0,2]
	stddev_RIDC[3] = error_std_lists_h3[1,2]	

if euler==1:
	plt.errorbar(h_list,error_list_Euler,yerr=stddev_Euler,capsize=2,label="Euler")
if sympeuler==1:
	plt.errorbar(h_list,error_list_SymplecticEuler,yerr=stddev_SymplecticEuler,capsize=2,label="Symplectic Euler")
if heun==1:
	plt.errorbar(h_list,error_list_Heun,yerr=stddev_Heun,capsize=2,label="Heun")
if modsympeuler==1:
	plt.errorbar(h_list,error_list_ModifiedSymplecticEuler,yerr=stddev_ModifiedSymplecticEuler,capsize=2,label="Modified Symplectic Euler")
if ridc==1:
	plt.errorbar(RIDC_N*h_list,error_list_RIDC,yerr=stddev_RIDC,capsize=2,label="RIDC order 2")
#Reference slopes
order1slope = np.poly1d([1,0])
plot1interval = np.linspace(h_list[0],h_list[-1],100)
plot2interval = RIDC_N*plot1interval
lines = []
lines.extend(plt.plot(plot1interval,order1slope(plot1interval),label="Order 1"))
order2slope = np.poly1d([0.1,0,0])
lines.extend(plt.plot(plot2interval,order2slope(plot2interval),label="Order 2"))
from labellines import labelLine, labelLines
xvals = [h_list[2], RIDC_N*h_list[2]]
labelLines(lines, color='gray', xvals=xvals, align=False, zorder=2.5, drop_label=True)
#Loglog
plt.xscale("log")
plt.yscale("log")
#plt.xlim([8*1e-3,1e-1])
#plt.ylim([1e-6,1e-1])
plt.xlabel("Step size")
plt.ylabel("Invariant measure error")
plt.legend()
plt.show()