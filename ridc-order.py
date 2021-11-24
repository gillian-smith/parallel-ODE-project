import numpy as np
import matplotlib.pyplot as plt
from labellines import labelLine, labelLines

def integration_matrix(M):
    S = np.zeros((M,M+1))
    for i in range(M+1): # from 0 to M
        rootlist = []
        for k in range(M+1): # from 0 to M
            if k!=i:
                rootlist = np.append(rootlist,[k]) # k = 0 to M, k != i
        l_i = np.poly1d(rootlist,True)/np.prod(i-rootlist) # this is the Lagrange polynomial l_i(x)
        L_i = np.polyint(l_i) # find the indefinite integral
        for m in range(M): # from 0 to M-1
            S[m,i] = L_i(m+1) - L_i(m)
    return S

# Inputs: endpoints of interval [a,b], initial condition y(a) = alpha, 
# number of intervals N, number of correction steps M, number of subintervals per group K
def RIDCpK_FE(a,b,alpha,N,M,K):
    # time step size
    delta_t = (b-a)/N
    
    if N % K != 0:
        raise "Required that N is divisible by K"
    # number of groups of intervals
    J = int(N/K)
    
    if M > 0:
        S = integration_matrix(M)
    
    eta = np.zeros((J,K+1,M+1))
    # Loop over each group of intervals I_j
    for j in range(J):
        #print("Computing solution in group j = " + str(j))
        # Prediction loop
        #print("Predicting...")
        if j==0:
            eta[j,0,0] = alpha # Use the given initial condition
        else:
            eta[j,0,0] = eta[j-1,K,M] # Use the previously computed accurate solution
        for m in range(K):
            t_jm = a+(j*K+m)*delta_t
            eta[j,m+1,0] = eta[j,m,0] + delta_t*f(t_jm,eta[j,m,0]) # Euler method
        # Correction loops: should not run for M = 0
        for l in range(1,M+1):
            #print("Correcting now! Step " + str(l))
            eta[j,0,l] = eta[j,0,l-1]
            for m in range(M):
                f_integral = 0
                for i in range(M+1):
                    t_ji = a+(j*K+i)*delta_t
                    f_integral = f_integral + delta_t*S[m,i]*f(t_ji,eta[j,i,l-1])
                t_jm = a+(j*K+m)*delta_t
                eta[j,m+1,l] = eta[j,m,l] + delta_t*(f(t_jm,eta[j,m,l]) - f(t_jm,eta[j,m,l-1])) + f_integral
            # if there is a mistake, it is probably in the following loop
            for m in range(M,K):
                f_integral = 0
                for i in range(M+1):
                    t_jmminusMplusi = a+(j*K+(m-M+i+1))*delta_t
                    #t_jmminusMplusi = a+(j*K+(m-M+i))*delta_t
                    f_integral = f_integral + delta_t*S[M-1,i]*f(t_jmminusMplusi,eta[j,(m-M+i+1),l-1])
                    #f_integral = f_integral + delta_t*S[M-1,i]*f(t_jmminusMplusi,eta[j,(m-M+i),l-1])
                t_jm = a+(j*K+m)*delta_t
                eta[j,m+1,l] = eta[j,m,l] + delta_t*(f(t_jm,eta[j,m,l]) - f(t_jm,eta[j,m,l-1])) + f_integral
    #print(eta)
    eta_list = np.zeros(N+1)
    eta_list[0] = alpha
    # eta is a matrix of dimensions J x K+1 x M+1
    for j in range(J):
        for m in range(1,K+1):
            eta_list[j*K+m] = eta[j,m,M] # n = jK+m
    return eta_list

a = 0; b = 5; alpha = 1;
K = 40; 
# Right hand side
def f(t,y):
    return 4*t*np.sqrt(y)
# Known solution
def y(t):
    return (1+t**2)**2

N_list = K*2**np.array(range(6))
delta_t_list = 5/N_list
errors = np.zeros((len(N_list),4)) # we will have an error for each value of N
print(delta_t_list)

for M in range(4):
    for x in range(len(N_list)):
        N = N_list[x]
        delta_t = delta_t_list[x]

        # Solve using RIDC
        eta = RIDCpK_FE(a,b,alpha,N,M,K)

        t_list = a+delta_t*np.array(range(N+1))
        # Exact solution
        y_list = y(t_list)
    
#     plt.figure(x)
#     plt.plot(t_list,eta,label='RIDC')
#     plt.plot(t_list,eta_euler,label='Euler')
#     plt.plot(t_list,y_list,label='Exact solution')
#     plt.legend()
#     plt.title('N = '+str(N))
        
        #errors[x] = max(np.absolute(y_list-eta))
        errors[x,M] = np.absolute(y(b)-eta[N])
    

#plt.figure(x+1)
#plt.loglog(delta_t_list, errors, 'o', label='Absolute error')
#print("p="+str(p))
#print("log2 of error ratio:")
#print(np.log2(errors[:-1]/errors[1:]))

#p1 = np.poly1d(np.polyfit(delta_t_list,errors[:,0],1))
p1 = np.poly1d([errors[0,0]*(1/delta_t_list[0]),0])
#p2 = np.poly1d(np.polyfit(delta_t_list,errors[:,1],2))
p2 = np.poly1d([errors[0,1]*(1/delta_t_list[0])**2,0,0])
#p3 = np.poly1d(np.polyfit(delta_t_list,errors[:,2],3))
p3 = np.poly1d([errors[0,2]*(1/delta_t_list[0])**3,0,0,0])
#p4 = np.poly1d(np.polyfit(delta_t_list,errors[:,3],4))
p4 = np.poly1d([errors[0,3]*(1/delta_t_list[0])**4,0,0,0,0])

plotrange = np.linspace(10**(-4),1/2,100)

plt.plot(delta_t_list,errors[:,0], 'o', label='RIDC1 = Euler')
plt.plot(delta_t_list,errors[:,1], 'o', label='RIDC2')
plt.plot(delta_t_list,errors[:,2], 'o', label='RIDC3')
plt.plot(delta_t_list,errors[:,3], 'o', label='RIDC4')
lines=[]
lines.extend(plt.plot(plotrange,p1(plotrange),color='k',label="Order 1"))
lines.extend(plt.plot(plotrange,p2(plotrange),color='k',label="Order 2"))
lines.extend(plt.plot(plotrange,p3(plotrange),color='k',label="Order 3"))
lines.extend(plt.plot(plotrange,p4(plotrange),color='k',label="Order 4"))
plt.xscale('log')
plt.yscale('log')
xvals = [10**(-3),10**(-3),10**(-3),10**(-3)]
labelLines(lines, color='gray', xvals=xvals, align=False, zorder=2.5, drop_label=True)
plt.xlim([delta_t_list[-1]/10,1])
plt.legend()
plt.xlabel('Step size $h$')
plt.ylabel('Global error at time $T$')
plt.show()