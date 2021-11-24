import numpy as np
import matplotlib.pyplot as plt

gamma = 4
beta = 2

def V(q):
    return (1-q**2)**2-q/2

def dV(q):
    return -4*q*(1-q**2)-1/2

dq = 1e-6
N = int(6/dq)
q = np.linspace(-3,3,N) # most of the mass of the integrands is in the interval [-3,3]
func1samples = np.exp(-beta*V(q))
func2samples = (1/beta + q**2)*func1samples

Z = 1/np.trapz(func1samples,x=None,dx=dq)
#print(Z)
phi_integral = Z*np.trapz(func2samples,x=None,dx=dq)
#print(phi_integral)

#checksamples = Z*func1samples
#print(np.trapz(checksamples,x=None,dx=dq)) # should be 1

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
    return eta_list[:,N]

def f(t,pq):
    p = pq[0]
    q = pq[1]
    return np.array([-dV(q),p])

#h_list = 1/(2**(np.array(range(4,8))))
#h_list = 0.01*2**(np.array(range(1,9))/4) # this is what they actually used in the paper I think
h_list = 0.01*2**(np.array(range(2,9,2))/4)
#print(h_list)
#error_list_Euler = np.zeros(len(h_list))
#stddev_Euler = np.zeros(len(h_list))
#error_list_SymplecticEuler = np.zeros(len(h_list))
#stddev_SymplecticEuler = np.zeros(len(h_list))
#error_list_Heun = np.zeros(len(h_list))
#stddev_Heun = np.zeros(len(h_list))
#error_list_ModifiedSymplecticEuler = np.zeros(len(h_list))
#stddev_ModifiedSymplecticEuler = np.zeros(len(h_list))
#error_list_RIDC = np.zeros(len(h_list))
#stddev_RIDC = np.zeros(len(h_list))
T = 10**5
N_list = T/h_list

euler=1
sympeuler=0
heun=1
modsympeuler=0
ridc=1
RIDC_N = 10 # number of RIDC steps

import time
tic = time.perf_counter()



# Average over m trajectories
m = 5
#Progress bar


h = h_list[0]
#print(h)
N = int(T/h)
t = np.linspace(0,T,N)

from tqdm import tqdm

if euler==1:
    # Explicit Euler
    print("Euler")
    phisum = np.zeros(m)
    for j in tqdm(range(m)):
        #print(".",end="")
        p = -3/2
        q = -3/2
        for k in range(1,N):
            # Theta
            ptemp = np.exp(-gamma*h)*p+np.sqrt((1-np.exp(-2*gamma*h))/beta)*np.random.randn()
            qtemp = q
            # Phi: explicit Euler
            p = ptemp - h*dV(qtemp)
            q = qtemp + h*ptemp
            phisum[j] = (k-1)/k * phisum[j] + 1/k * (p**2+q**2)
    error_Euler = np.absolute(np.mean(phisum) - phi_integral)
    stddev_Euler = np.std(phisum)

if sympeuler==1:
    # Symplectic Euler
    print("Symplectic Euler")
    phisum = np.zeros(m)
    for j in tqdm(range(m)):
        #print(".",end="")
        p = -3/2
        q = -3/2
        for k in range(1,N):
            # Theta
            ptemp = np.exp(-gamma*h)*p+np.sqrt((1-np.exp(-2*gamma*h))/beta)*np.random.randn()
            qtemp = q
            # Phi: symplectic Euler
            p = ptemp - h*dV(qtemp)
            q = qtemp + h*p
            phisum[j] = (k-1)/k * phisum[j] + 1/k * (p**2+q**2)
    error_SymplecticEuler = np.absolute(np.mean(phisum) - phi_integral)
    stddev_SymplecticEuler = np.std(phisum)

if heun==1:    
    # Heun
    print("Heun")
    phisum = np.zeros(m)
    for j in tqdm(range(m)):
        #print(".",end="")
        p = -3/2
        q = -3/2
        for k in range(1,N):
            # Theta
            ptemp = np.exp(-gamma*h)*p+np.sqrt((1-np.exp(-2*gamma*h))/beta)*np.random.randn()
            qtemp = q
            # Phi: Heun
            p = ptemp - h*dV(qtemp+ptemp*h/2)
            q = qtemp + h*(ptemp-dV(qtemp)*h/2)
            phisum[j] = (k-1)/k * phisum[j] + 1/k * (p**2+q**2)
    error_Heun = np.absolute(np.mean(phisum) - phi_integral)
    stddev_Heun = np.std(phisum)

if modsympeuler==1:    
    # Modified symplectic Euler
    print("Modified symplectic Euler")
    phisum = np.zeros(m)
    for j in tqdm(range(m)):
        #print(".",end="")
        p = -3/2
        q = -3/2
        for k in range(1,N):
            # Theta
            ptemp = np.exp(-gamma*h)*p+np.sqrt((1-np.exp(-2*gamma*h))/beta)*np.random.randn()
            qtemp = q
            # Phi: modified symplectic Euler
            alpha = 1 + h/2 * beta * ptemp * dV(qtemp)
            p = ptemp - alpha*h*dV(qtemp)
            q = qtemp + alpha*h*p
            phisum[j] = (k-1)/k * phisum[j] + 1/k * (p**2+q**2)
    error_ModifiedSymplecticEuler = np.absolute(np.mean(phisum) - phi_integral)
    stddev_ModifiedSymplecticEuler = np.std(phisum)

if ridc==1:
    h_hat = RIDC_N*h
    # RIDC
    print("RIDC")
    phisum = np.zeros(m)
    for j in tqdm(range(m)):
        #print(".",end="")
        p = -3/2
        q = -3/2
        for k in range(1,N):
            # Theta
            ptemp = np.exp(-gamma*h_hat)*p+np.sqrt((1-np.exp(-2*gamma*h_hat))/beta)*np.random.randn()
            qtemp = q
            pqtemp = np.array([ptemp,qtemp])
            # Phi: RIDC order 2, N=K
            pq = RIDCpK_FE(0,h_hat,pqtemp,RIDC_N,1,RIDC_N)
            p = pq[0]
            q = pq[1]
            phisum[j] = (k-1)/k * phisum[j] + 1/k * (p**2+q**2)
    error_RIDC = np.absolute(np.mean(phisum) - phi_integral)
    stddev_RIDC = np.std(phisum)

toc = time.perf_counter()
print(f"Time elapsed = {toc - tic:0.4f} seconds") 

error_std_lists = np.array([[error_Euler,error_Heun,error_RIDC],[stddev_Euler,stddev_Heun,stddev_RIDC]])
#print(np.shape(error_std_lists))

file = open("error_std_lists_h0","wb")
np.save(file,error_std_lists)



