# -*- coding: utf-8 -*-
"""
Created on Mon Jun 20 15:54:49 2022

@author: gthiran
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Jun 20 14:14:17 2022

@author: gthiran
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Jun  8 12:25:08 2022

@author: gthiran
"""

import numpy as np
import numpy.linalg as la
from scipy.linalg import sqrtm
import matplotlib.pyplot as plt

n = 30
w = np.arange(1,n**2+1)
www = w.reshape((n,n)).T
A = np.sin(93*www**3)
Q = A.T @ A
b = 10*np.sin(27*np.arange(1,n+1)**3)
c = b @ la.solve(Q,b)+1
DD = np.sin(15*www**3)
Z = sqrtm(la.inv(DD@DD.T))
D = Z @ DD


e = np.ones((n,)) #unit vector

def soft_thres(eta,x):
    return np.maximum(np.abs(x)-eta,0)*np.sign(x)

def obj(x,Q,b,c,D,e):
    return np.sqrt(x.T@Q@x+2*b.T@x+c) + 0.2*np.linalg.norm(D@x+e,ord=1)

def gradf(x,Q,b,c):
    return (Q@x+b)/np.sqrt(x.T@Q@x+2*b.T@x+c) 


I=10000#number of iterations
#Proximal gradient
x_PG = np.zeros((n,I+1))
x_PG[:,0] = np.ones((n,))
obj_PG = np.zeros((I+1,))
obj_PG[0] = obj(x_PG[:,0],Q,b,c,D,e)

L = np.linalg.norm(Q,ord=2)/np.sqrt(c-b.T@np.linalg.inv(Q)@b)

#FISTA
x_FISTA = np.zeros((n,I+1))
x_FISTA[:,0] = np.ones((n,))
obj_FISTA = np.zeros((I+1,))
obj_FISTA[0] = obj(x_FISTA[:,0],Q,b,c,D,e)
y_FISTA = x_FISTA[:,0]
t_FISTA=1



for i in range(I):
    #Perform PS update
    x_c = x_PG[:,i]
    x_next = D.T@soft_thres(0.2/L,D@(x_c-1/L*gradf(x_c,Q,b,c))+e)-D.T@e
    x_PG[:,i+1] = x_next
    #compute objective function
    obj_PG[i+1] = obj(x_next,Q,b,c,D,e)
    
    #Perform FISTA update
    x_c = x_FISTA[:,i]
    x_next = D.T@soft_thres(0.2/L,D@(y_FISTA-1/L*gradf(y_FISTA,Q,b,c))+e)-D.T@e
    t_FISTA_next = (1+np.sqrt(1+4*t_FISTA**2))/2
    y_FISTA = x_next+(t_FISTA-1)/t_FISTA_next*(x_next - x_c)
    t_FISTA = t_FISTA_next
    x_FISTA[:,i+1] = x_next
    #compute objective function
    obj_FISTA[i+1] = obj(x_next,Q,b,c,D,e)
    
#plot
I_graph = 1001
plt.close('all')
plt.figure()
Fct_opt = min(np.min(obj_PG),np.min(obj_FISTA))
plt.semilogy(obj_PG[:I_graph]-Fct_opt,'b', label='Proximal Gradient')
plt.semilogy(obj_FISTA[:I_graph]-Fct_opt,'r', label='FISTA')
plt.legend()
plt.show
plt.savefig('part2_ex1.pdf')

    