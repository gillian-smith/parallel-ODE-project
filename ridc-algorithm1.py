# Solve the ODE dy/dt = f(t,y) using the deferred correction method
# Use the Euler method for both the prediction step and correction step

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# Right hand side
def f(t,y):
	return y

# Endpoints of interval
a = 0
b = 5

# Initial condition y(a) = alpha
alpha = 1

# Number of intervals
N = 10

delta_t = (b-a)/N

# Number of correction steps (N must be divisible by M)
M = 10
p = M+1 # order of method
if N % M != 0:
	raise "Required that N is divisible by M"
J = int(N/M)

# Known solution
def y(t):
	return alpha*np.exp(t-a)

t_list = a+delta_t*np.array(range(N+1))
y_list = y(t_list)

# Precalculate quadrature weights
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

eta = np.zeros((J,M+1,M+1))
# Loop over each group of intervals I_j
for j in range(J):
	# Prediction loop
	if j==0:
		eta[j,0,0] = alpha # Use the given initial condition
	else:
		eta[j,0,0] = eta[j-1,M,M] # Use the previously computed accurate solution
	for m in range(M):
		t_jm = a+(j*M+m)*delta_t
		eta[j,m+1,0] = eta[j,m,0] + delta_t*f(t_jm,eta[j,m,0]) # Euler method
	# Correction loops
	for l in range(1,M+1):
		eta[j,0,l] = eta[j,0,l-1]
		for m in range(M):
			f_integral = 0
			for i in range(M+1):
				t_ji = a+(j*M+i)*delta_t
				f_integral = f_integral + delta_t*S[m,i]*f(t_ji,eta[j,i,l-1])
			eta[j,m+1,l] = eta[j,m,l] + delta_t*(f(t_jm,eta[j,m,l]) - f(t_jm,eta[j,m,l-1])) + f_integral

eta_lists = np.zeros((M+1,N+1))
eta_lists[:,0] = alpha
for j in range(J):
	for m in range(1,M+1):
		eta_lists[:,j*M+m] = eta[j,m,:]
#print(eta_lists)

errors = np.zeros(M+1)
for l in range(M+1):
	error_list = np.absolute(y_list - eta_lists[l,:])
	errors[l] = max(error_list)

plt.figure(1)
plt.plot(t_list,y_list,label='Exact solution')
plt.plot(t_list,eta_lists[0,:],label='Prediction')
for l in range(1,M+1):
	plt.plot(t_list,eta_lists[l,:],label='Correction '+str(l))
ax = plt.gca()
ax.set_xlabel('t')
ax.set_ylabel('y(t)')
ax.set_title('Solving dy/dt = f(t,y) using N=' + str(N) + ' intervals and M=' + str(M) + ' correction levels')
plt.legend()

plt.figure(2)
plt.plot(np.array(range(M+1)), errors, 'o')
plt.yscale('log')
plt.plot(np.array(range(M+1)), 100*0.5**(np.array(range(M+1))+1))
plt.plot(np.array(range(M)), errors[0:-2]/errors[1:-1])
ax = plt.gca()
ax.set_xlabel('Number of corrections M')
ax.set_ylabel('Max error in solution')
ax.set_title('Solving dy/dt = f(t,y) using N=' + str(N) + ' intervals')

plt.show()