# Solve the equation dy/dt = y using Classical Deferred Correction
# Perform the algorithm for different step sizes to show how the order of accuracy depends on number of nodes N and number of correction steps M

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# initial condition
y0 = 1

# number of nodes
N = 10

# number of correction steps
M = 5

# order of accuracy
p = min(N,M+1)

#dt_list = np.array([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])
dt_list = np.array([0.125,0.25,0.5,1])

errornorm_list = np.zeros(np.size(dt_list))

for dt_index in range(np.size(dt_list)):

	# step size
	dt = dt_list[dt_index]

	# start and end points
	t0 = 0
	tN = t0 + N*dt

	t_list = np.linspace(t0,tN,num=N+1)

	y_list = np.zeros(N+1)
	y_list[0] = y0

	# solve for provisional y
	for n in range(N):
		y_list[n+1] = y_list[n] + dt*y_list[n] # Euler's method
	#print("Provisional solution")
	#print(y_list)
	plt.figure(dt_index)
	ax = plt.gca()
	ax.plot(t_list, y_list, label='Provisional solution')

	for k in range(M):
		
		# interpolate to find dy/dt
		dydt_list = np.zeros(N+1)
		for n in range(N+1):
			for i in range(N+1):
				sum1 = 0
				for j in range(N+1):
					if j!=i:
						prod1 = 1
						for m in range(N+1):
							if m!=i and m!=j:
								prod1 = prod1 * (t_list[n]-t_list[m])/(t_list[i]-t_list[m])
						sum1 = sum1 + prod1 / (t_list[i] - t_list[j])
				dydt_list[n] = dydt_list[n] + y_list[i]*sum1

		# solve for error function
		e_list = np.zeros(N+1)
		for n in range(N):
			e_list[n+1] = e_list[n] + dt*( e_list[n] - dydt_list[n] + y_list[n] )

		# corrected solution
		y_list = y_list + e_list

		curlabel = "Correction level " + str(k+1)
		ax.plot(t_list, y_list, label=curlabel)

		#print('Corrected solution - correction #', k+1)
		#print(y_list)

	y_exact = np.exp(t_list)

	ax.plot(t_list, y_exact, label='Exact solution')
	#print("Exact solution")
	#print(y_exact)

	ax.set_xlabel('t')
	ax.set_ylabel('y')
	plottitle = "Solution to y' = y, with dt = " + str(dt)
	ax.set_title(plottitle)
	ax.legend()

	errornorm_list[dt_index] = np.linalg.norm(y_list - y_exact, ord=np.inf)

C = max(errornorm_list)/(dt**p)
dt_plot = np.linspace(0,max(dt_list))
ChN_plot = C*dt_plot**N
ChMplus1_plot = C*dt_plot**(M+1)

#print(dt_list)
#print(errornorm_list)
plt.figure(np.size(dt_list)+1)
#ax = plt.gca()
#ax.plot(dt_list, errornorm_list, 'o', label='Error')
#ax.plot(dt_plot, ChN_plot, label='C h^N')
#ax.plot(dt_plot, ChMplus1_plot, label='C h^(M+1)')
#ax.set_xlabel('dt')
#ax.set_ylabel('Error norm: max(y_approx - y_exact)')
#plottitle = "Order of accuracy p = min(N,M+1) = " + str(p)
#ax.set_title(plottitle)
#ax.legend()
plt.loglog(dt_list,errornorm_list)
plt.show()
