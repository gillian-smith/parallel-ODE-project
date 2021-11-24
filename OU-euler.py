import numpy as np
import matplotlib.pyplot as plt
from labellines import labelLine, labelLines

dt_list = np.array([0.8,0.4,0.2,0.1])
T = 1000
N_list = T/dt_list
gamma=1
beta=1
# Number of sample paths
M = 10

errors = np.zeros(4)
for j in range(len(dt_list)):
    N = int(N_list[j])
    dt = dt_list[j]
    t=np.linspace(0,T,N)
    A = 1-gamma*dt
    B = np.sqrt(2*gamma*dt/beta)
    #A=np.exp(-gamma*dt)
    #B=np.sqrt(sigma**2/(2*gamma)*(1-A**2))
    
    X=np.zeros([N,M]) # initial condition X(0)=0
    integral1=np.zeros([N,1])
    integral2=np.zeros([N,1])

    for i in range(N-1):
        X[i+1,:]=X[i,:]*A+B*np.random.randn(M)
        integral1[i+1]=np.mean(X[0:i+1,:])
        integral2[i+1]=np.mean(X[0:i+1,:]**2)
    
    error = abs(integral2[-1]-1)
    errors[j] = error

#p = np.poly1d(np.polyfit(dt_list,errors,1))
p = np.poly1d([1,0])
    
plt.plot(dt_list,errors,"o-",label="Error")
lines=[]
lines.extend(plt.plot(np.linspace(0.1,0.8,100),p(np.linspace(0.1,0.8,100)),label="Order 1"))
labelLines(lines, color='gray', align=False, zorder=2.5, drop_label=True)
#ax=plt.axes()
#ax.set_xticks([1e-1,2e-1,4e-1,8e-1])
plt.xlabel("Step size $h$")
plt.ylabel("Error in second moment")
plt.xscale('log')
plt.yscale('log')
#plt.legend()
plt.show()