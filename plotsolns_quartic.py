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
    return eta_list[:,N]

def V(q):
    return (1-q**2)**2-q/2

def dV(q):
    return -4*q*(1-q**2)-1/2

def H(p,q):
    return p**2/2 + V(q)

def f(t,pq):
    p = pq[0]
    q = pq[1]
    return np.array([-dV(q),p])

gamma = 0.5
beta = 1

T = 1000
N = 100*T
t = np.linspace(0,T,N)
h = T/N
RIDC_N = 10

soln_Euler = np.zeros((2,N))
soln_SympEuler = np.zeros((2,N))
soln_Heun = np.zeros((2,N))
soln_RIDC2 = np.zeros((2,int(N/RIDC_N)))
soln_RIDC3 = np.zeros((2,int(N/RIDC_N)))

# Explicit Euler
p = -3/2
q = -3/2
soln_Euler[0,0]=p
soln_Euler[1,0]=q
for k in range(1,N):
    # Theta
    ptemp = np.exp(-gamma*h)*p+np.sqrt((1-np.exp(-2*gamma*h))/beta)*np.random.randn()
    qtemp = q
    # Phi: explicit Euler
    p = ptemp - h*dV(qtemp)
    q = qtemp + h*ptemp
    soln_Euler[0,k]=p
    soln_Euler[1,k]=q
    
# Symplectic Euler
p = -3/2
q = -3/2
soln_SympEuler[0,0]=p
soln_SympEuler[1,0]=q
for k in range(1,N):
    # Theta
    ptemp = np.exp(-gamma*h)*p+np.sqrt((1-np.exp(-2*gamma*h))/beta)*np.random.randn()
    qtemp = q
    # Phi: symplectic Euler
    p = ptemp - h*dV(qtemp)
    q = qtemp + h*p
    soln_SympEuler[0,k]=p
    soln_SympEuler[1,k]=q
    
# Heun
p = -3/2
q = -3/2
soln_Heun[0,0]=p
soln_Heun[1,0]=q
for k in range(1,N):
    # Theta
    ptemp = np.exp(-gamma*h)*p+np.sqrt((1-np.exp(-2*gamma*h))/beta)*np.random.randn()
    qtemp = q
    # Phi: Heun
    p = ptemp - h*dV(qtemp+ptemp*h/2)
    q = qtemp + h*(ptemp-dV(qtemp)*h/2)
    soln_Heun[0,k]=p
    soln_Heun[1,k]=q
    
# RIDC order 2
h_hat = RIDC_N*h
p = -3/2
q = -3/2
soln_RIDC2[0,0]=p
soln_RIDC2[1,0]=q
for k in range(1,int(N/RIDC_N)):
    # Theta
    ptemp = np.exp(-gamma*h_hat)*p+np.sqrt((1-np.exp(-2*gamma*h_hat))/beta)*np.random.randn()
    qtemp = q
    pqtemp = np.array([ptemp,qtemp])
    # Phi: RIDC order 2, N=K
    pq = RIDCpK_FE(0,h_hat,pqtemp,RIDC_N,1,RIDC_N)
    p = pq[0]
    q = pq[1]
    soln_RIDC2[0,k]=p
    soln_RIDC2[1,k]=q

# RIDC order 3
h_hat = RIDC_N*h
p = -3/2
q = -3/2
soln_RIDC3[0,0]=p
soln_RIDC3[1,0]=q
for k in range(1,int(N/RIDC_N)):
    # Theta
    ptemp = np.exp(-gamma*h_hat)*p+np.sqrt((1-np.exp(-2*gamma*h_hat))/beta)*np.random.randn()
    qtemp = q
    pqtemp = np.array([ptemp,qtemp])
    # Phi: RIDC order 3, N=K
    pq = RIDCpK_FE(0,h_hat,pqtemp,RIDC_N,2,RIDC_N)
    p = pq[0]
    q = pq[1]
    soln_RIDC3[0,k]=p
    soln_RIDC3[1,k]=q

fig, ax = plt.subplots()
p_range = np.arange(-4,4,0.1)
q_range = np.arange(-4,4,0.1)
P,Q = np.meshgrid(p_range,q_range)
H_array = H(P,Q)
plt.contour(P,Q,H_array,[0,1,2,3,4])
#plt.plot(soln_Euler[0,range(0,N,40)],soln_Euler[1,range(0,N,40)],'o',markersize=1,label="Euler solution")
#plt.plot(soln_SympEuler[0,:],soln_SympEuler[1,:],label="Symplectic Euler")
#plt.plot(soln_Heun[0,:],soln_Heun[1,:],label="Heun")
plt.plot(soln_RIDC2[0,range(0,int(N/RIDC_N),4)],soln_RIDC2[1,range(0,int(N/RIDC_N),4)],'o',markersize=1,label="RIDC2")
#plt.plot(soln_RIDC3[0,range(0,int(N/RIDC_N),4)],soln_RIDC3[1,range(0,int(N/RIDC_N),4)],'o',markersize=1,label="RIDC3")
plt.xlim([-4,4])
plt.ylim([-2.5,2.5])
plt.xlabel("$p$")
plt.ylabel("$q$")
plt.title(r"$h=$" + np.str(h) + r", $T=$" + np.str(T))
ax.set_aspect('equal')
ax.grid(True, which='both')
ax.axhline(y=0, color='k')
ax.axvline(x=0, color='k')
plt.legend()
plt.show()