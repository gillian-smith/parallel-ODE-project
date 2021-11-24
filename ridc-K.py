import numpy as np
import matplotlib.pyplot as plt

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
    
    # number of dimensions
    D = np.size(alpha)
    
    if N % K != 0:
        raise "Required that N is divisible by K"
    # number of groups of intervals
    J = int(N/K)
    
    if M > 0:
        S = integration_matrix(M)
    
    eta = np.zeros((D,J,K+1,M+1))
    # Loop over each group of intervals I_j
    for j in range(J):
        #print("Computing solution in group j = " + str(j))
        # Prediction loop
        #print("Predicting...")
        if j==0:
            eta[:,j,0,0] = alpha # Use the given initial condition
        else:
            eta[:,j,0,0] = eta[:,j-1,K,M] # Use the previously computed accurate solution
        for m in range(K):
            t_jm = a+(j*K+m)*delta_t
            eta[:,j,m+1,0] = eta[:,j,m,0] + delta_t*f(t_jm,eta[:,j,m,0]) # Euler method
        # Correction loops: should not run for M = 0
        for l in range(1,M+1):
            #print("Correcting now! Step " + str(l))
            eta[:,j,0,l] = eta[:,j,0,l-1]
            for m in range(M):
                f_integral = 0
                for i in range(M+1):
                    t_ji = a+(j*K+i)*delta_t
                    f_integral = f_integral + delta_t*S[m,i]*f(t_ji,eta[:,j,i,l-1])
                t_jm = a+(j*K+m)*delta_t
                eta[:,j,m+1,l] = eta[:,j,m,l] + delta_t*(f(t_jm,eta[:,j,m,l]) - f(t_jm,eta[:,j,m,l-1])) + f_integral
            # if there is a mistake, it is probably in the following loop
            for m in range(M,K):
                f_integral = 0
                for i in range(M+1):
                    t_jmminusMplusi = a+(j*K+(m-M+i+1))*delta_t
                    #t_jmminusMplusi = a+(j*K+(m-M+i))*delta_t
                    f_integral = f_integral + delta_t*S[M-1,i]*f(t_jmminusMplusi,eta[:,j,(m-M+i+1),l-1])
                    #f_integral = f_integral + delta_t*S[M-1,i]*f(t_jmminusMplusi,eta[:,j,(m-M+i),l-1])
                t_jm = a+(j*K+m)*delta_t
                eta[:,j,m+1,l] = eta[:,j,m,l] + delta_t*(f(t_jm,eta[:,j,m,l]) - f(t_jm,eta[:,j,m,l-1])) + f_integral
    #print(eta)
    eta_list = np.zeros((D,N+1))
    eta_list[:,0] = alpha
    # eta is a matrix of dimensions D x J x K+1 x M+1
    for j in range(J):
        for m in range(1,K+1):
            eta_list[:,j*K+m] = eta[:,j,m,M] # n = jK+m
    return eta_list

a = 0; b = 10; alpha = np.array([1,0])
K = 40; N = 1000;
# Order of method
M = 3; p = M + 1

# Right hand side
def f(t,y):
    y1 = y[0]
    y2 = y[1]
    return np.array([-y2+y1*(1-y1**2-y2**2),y1+3*y2*(1-y1**2-y2**2)])

# Known solution
def y(t):
    return np.array([np.cos(t),np.sin(t)])


N = 1000;
p = 4; M = p-1;

# Construct a list of possible Ks (i.e. divisors of N that are not less than M)
K_list = []
for n in range(M,N+1):
    if (N % n == 0):
        K_list.append(n)
print(K_list)

num_of_Ks = len(K_list)
error_infnorm_list = np.zeros(num_of_Ks)
error_sqamp_list = np.zeros(num_of_Ks)
error_ph_list = np.zeros(num_of_Ks)

t_list = np.linspace(a,b,N+1)
y_final = y(t_list[N])

for i in range(num_of_Ks):
    K = K_list[i]
    eta = RIDCpK_FE(a,b,alpha,N,M,K)
    error = max(np.absolute(eta[:,N]-y_final))
    error_infnorm_list[i] = error
    error_sqamp_list[i] = np.absolute(eta[0,N]**2+eta[1,N]**2-1)
    error_ph_list[i] = np.absolute(np.arctan2(eta[1,N],eta[0,N])-np.arctan2(y_final[1],y_final[0]))
    #error_sqamp_list[i] = np.absolute((eta[0,N]-y_final[0])**2+(eta[1,N]-y_final[1])**2)

plt.rcParams['font.size'] = '14'
    
plt.figure(1)    
plt.loglog(K_list,error_infnorm_list,'o-')
plt.xlabel("K")
plt.ylabel("Absolute error")
if p==4:
    plt.xlim((10**0,10**3))
    plt.ylim((10**(-9),10**(-6)))
plt.show()

plt.figure(2)    
plt.loglog(K_list,error_sqamp_list,'o-')
plt.xlabel("K")
plt.ylabel("Squared amplitude error")
if p==4:
    plt.xlim((10**0,10**3))
    plt.ylim((10**(-10),10**(-7)))
plt.show()

plt.figure(3)    
plt.loglog(K_list,error_ph_list,'o-')
plt.xlabel("K")
plt.ylabel("Phase error")
if p==4:
    plt.xlim((10**0,10**3))
    plt.ylim((10**(-9),10**(-6)))
plt.show()