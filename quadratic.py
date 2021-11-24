import numpy as np
import scipy.linalg as linalg
import matplotlib.pyplot as plt

gamma = 4
beta = 1

h_list = 1/(2**(np.array(range(8))))
error_list_Euler = np.zeros(len(h_list))
error_list_Heun = np.zeros(len(h_list))

F = np.array([[0,-1],
              [1, 0]])
I = np.identity(2)

for i in range(len(h_list)):
    h = h_list[i]
    
    U = np.array([[np.exp(-gamma*h), -h],
                  [h*np.exp(-gamma*h),1]])
    V = np.sqrt(1-np.exp(-2*gamma*h))*np.array([[1, 0],
                                                [h, 0]])
    Sigma = linalg.solve_discrete_lyapunov(U,V@np.transpose(V))
    error_list_Euler[i] = linalg.norm(Sigma - I,2)
    
    A = I + (h/2)*F@(I+F)
    B = np.array([[np.exp(-gamma*h), 0],
                  [0               , 1]])
    C = np.sqrt(1-np.exp(-2*gamma*h))*np.array([[1],
                                                [0]])
    U = A@B
    V = A@C
    Sigma = linalg.solve_discrete_lyapunov(U,V@np.transpose(V))
    error_list_Heun[i] = linalg.norm(Sigma - I,2)
    
p = np.poly1d([1,0])    
    
plt.plot(h_list,error_list_Euler,'o--',label='Error')
lines = []
lines.extend(plt.plot(h_list,p(h_list),color='gray',label='Order 1'))
from labellines import labelLine, labelLines
labelLines(lines, color='gray', align=False, zorder=2.5, drop_label=True)
#plt.plot(h_list,error_list_Heun,'o--')
plt.xscale('log')
plt.yscale('log')
plt.xlabel("Step size h")
plt.ylabel("$||\Sigma_h - \\Sigma||_2$")
plt.legend()
plt.show()