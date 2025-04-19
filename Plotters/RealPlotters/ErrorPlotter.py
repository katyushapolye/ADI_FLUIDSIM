import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl

plt.style.use('seaborn-v0_8-darkgrid')
mpl.rcParams['font.family'] = 'serif'
#
plt.figure(figsize=(6, 6)) 
#
#
plt.gca().set_aspect('equal', adjustable='datalim') 




log = np.genfromtxt("Data/Error/re1000/error_re=1000_linear.csv",delimiter=',')
dH = log[0:,0]
error = log[0:,1]
plt.plot(np.log2(dH),np.log2(error),marker='.',label = 'Δt=Δh -Rₑ = 1000')

log = np.genfromtxt("Data/Error/re1000/error_re=1000_quadratic.csv",delimiter=',')
dH = log[0:,0]
error = log[0:,1]
plt.plot(np.log2(dH),np.log2(error),marker='.',label = 'Δt=Δh²-Rₑ = 1000')










log = np.genfromtxt("Data/Error/re10000/error_re=10000_linear.csv",delimiter=',')
dH = log[0:,0]
error = log[0:,1]
plt.plot(np.log2(dH),np.log2(error),marker='.',label = 'Δt=Δh -Rₑ = 10000')

log = np.genfromtxt("Data/Error/re10000/error_re=10000_quadratic.csv",delimiter=',')
dH = log[0:,0]
error = log[0:,1]
plt.plot(np.log2(dH),np.log2(error),marker='.',label = 'Δt=Δh²-Rₑ = 10000')


plt.grid(True)






plt.title("Error Decay in 2D Taylor-Green Vortex",fontsize=16, fontweight='bold')
plt.legend()
plt.savefig("error_log.png")
#
#
plt.legend(loc='best',framealpha=0.3)

x_annotation = 4.0  # x annotation point on the graph (log scale, so use powers of 10)
y_annotation = -12
slope = 2

plt.annotate(xy=(x_annotation-0.5, y_annotation+0.5),xytext=(x_annotation-0.4, y_annotation-0.35),text="k = 2",)

plt.annotate(xy=(x_annotation-0.5, y_annotation+1),xytext=(x_annotation, y_annotation),text="",
             arrowprops=dict(facecolor='black', arrowstyle='->'))

plt.annotate(xy=(x_annotation-0.5, y_annotation),xytext=(x_annotation, y_annotation),text="",
             arrowprops=dict(facecolor='black', arrowstyle='-'))
plt.annotate(xy=(x_annotation-0.5, y_annotation+1),xytext=(x_annotation-0.5, y_annotation),text="",
             arrowprops=dict(facecolor='black', arrowstyle='-'))

plt.plot(x_annotation - 0.5, y_annotation + 1, '.', color='black')  # Start point (head)
plt.plot(x_annotation - 0.5, y_annotation, '.', color='black')  # End point (tail)
plt.plot(x_annotation , y_annotation, '.', color='black')  # End point (tail)
 

#plt.axis('equal')
plt.xlim(2,6)
plt.ylim(-8,-2)
plt.xlabel("Log2(N)")
plt.ylabel("Log2(Max Absolute Error)")
plt.show()
plt.close()
#
#
#
#log = np.genfromtxt("Error/errorGrad.csv",delimiter=',')
#dH = log[0:,0]
#error = log[0:,1]
#print("Calculating error decay")
#order = -(np.log(error[-1]) - np.log(error[0])) / (np.log(dH[-1]) - np.log(dH[0]))
#plt.plot(np.log(dH),np.log(error),marker='*',color='blue',label='Logarithmic Error in \u2207P')
#
#
#log = np.genfromtxt("Error/errorPressure.csv",delimiter=',')
#dH = log[0:,0]
#error = log[0:,1]
#print("Calculating error decay")
#order = -(np.log(error[-1]) - np.log(error[0])) / (np.log(dH[-1]) - np.log(dH[0]))
#plt.plot(np.log(dH),np.log(error),marker='*',color='red',label='Logarithmic Error in P')
#plt.title("Error  P  and \u2207P - \u0394t = \u0394h/2π")
#plt.xlabel('N')
#plt.ylabel('Error')
#plt.legend()
#plt.savefig("error_log_P.png")
#plt.close()