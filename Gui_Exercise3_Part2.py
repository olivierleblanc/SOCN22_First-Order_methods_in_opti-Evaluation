# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 06:43:57 2022

@author: gthiran
"""

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
import numpy.matlib
import numpy.linalg as la
from scipy.linalg import sqrtm
import matplotlib.pyplot as plt


plt.close('all')

x = np.sin(10*np.arange(1,41)**3)
y = np.sin(28*np.arange(1,41)**3)
cl = (2*x<y+0.5)+1
x = np.hstack((x,0.2))
y = np.hstack((y,-0.2))
cl = np.hstack((cl,2))
A1=np.column_stack((x[cl==1], y[cl==1]))
A2=np.column_stack((x[cl==2], y[cl==2]))
plt.plot(A1[:,0], A1[:,1],'*')
plt.plot(A2[:,0], A2[:,1],'d')
plt.show()


n = len(cl)
X = np.column_stack((x,y))
m = 2
Y = 2*cl-3
Z = X* np.column_stack((Y,Y))
Z=Z.T


C=1

def obj(w,C,Z):
    return 0.5*w.T@w+C*np.sum(np.maximum(0,1-w.T@Z))



I=40#number of iterations
#Proximal gradient
x_DPG = np.zeros((m,I+1))
obj_DPG = np.zeros((I+1,))
obj_DPG[0] = obj(x_DPG[:,0],C,Z)
y_DPG = np.zeros((n,))
A = Z.T
L = np.linalg.norm(A,ord=2)**2

#FISTA
x_FDPG = np.zeros((m,I+1))
obj_FDPG = np.zeros((I+1,))
obj_FDPG[0] = obj(x_FDPG[:,0],C,Z)
y_FDPG = np.zeros((n,))
w_FDPG = np.zeros((n,))
t_FDPG=1



for i in range(I):
    #Perform DPG update
    x_c = x_DPG[:,i]
    y_DPG = np.minimum(np.maximum(y_DPG-1/L*A@x_c+1/L,0),C)
    x_next = A.T@y_DPG
    x_DPG[:,i+1] = x_next
    #compute objective function
    obj_DPG[i+1] = obj(x_next,C,Z)
    
    #Perform FDPG update
    u_c = A.T@w_FDPG
    y_FDPG_next = np.minimum(np.maximum(w_FDPG-1/L*A@u_c+1/L,0),C)
    t_FDPG_next = (1+np.sqrt(1+4*t_FDPG**2))/2
    w_FDPG = y_FDPG_next+(t_FDPG-1)/t_FDPG_next*(y_FDPG_next - y_FDPG)
    t_FDPG = t_FDPG_next
    y_FDPG = y_FDPG_next
    x_FDPG[:,i+1] = A.T@y_FDPG
    #compute objective function
    obj_FDPG[i+1] = obj(x_FDPG[:,i+1] ,C,Z)
    
    

#plot

plt.figure()
Fct_opt = min(np.min(obj_DPG),np.min(obj_FDPG))
plt.semilogy(obj_DPG-Fct_opt,'b')
plt.semilogy(obj_FDPG-Fct_opt,'r')
plt.show

plt.figure()  
x = np.sin(10*np.arange(1,41)**3)
y = np.sin(28*np.arange(1,41)**3)
cl = (2*x<y+0.5)+1
x = np.hstack((x,0.2))
y = np.hstack((y,-0.2))
cl = np.hstack((cl,2))
A1=np.column_stack((x[cl==1], y[cl==1]))
A2=np.column_stack((x[cl==2], y[cl==2]))
plt.plot(A1[:,0], A1[:,1],'*')
plt.plot(A2[:,0], A2[:,1],'d')
plt.show()

fact=1.5
vec_DPG = np.array([x_DPG[1,-1],-x_DPG[0,-1]])
vec_DPG = vec_DPG/np.linalg.norm(vec_DPG,ord=2)*fact
plt.plot(np.array([vec_DPG[0],-vec_DPG[0]]),np.array([vec_DPG[1],-vec_DPG[1]]),'g',label='DPG')
vec_FDPG = np.array([x_FDPG[1,-1],-x_FDPG[0,-1]])
vec_FDPG = vec_FDPG/np.linalg.norm(vec_FDPG,ord=2)*fact
plt.plot(np.array([vec_FDPG[0],-vec_FDPG[0]]),np.array([vec_FDPG[1],-vec_FDPG[1]]),'k',label='FDPG')
plt.grid(True)
plt.legend()
plt.savefig('part2_ex3.pdf')