# -*- coding: utf-8 -*-
"""
Created on Wed Jun 22 14:09:34 2022

@author: gthiran
"""

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

m = 30
n = 25
a = np.arange(m)+1
b = np.arange(n)+0.5
Atilde = np.sin(10*np.outer(a,b)**3)
xi = np.sin(31*np.arange(1,n+1)**3)
b = Atilde@xi+np.sin(23*np.arange(1,m+1)**3)+1.5


#Objective function
def obj(x,Atilde,b):
    return -np.sum(np.log(Atilde@x-b))+np.sum(np.sqrt((x[:-2]-x[1:-1])**2+(x[1:-1]-x[2:])**2))

def prox_h2(beta,b,v):
    m = len(b)
    n = np.int((len(v)+4-2*m)/2)
    p = len(v)
    v1 = v[:m]
    v2 = v[m:m+n-2]
    v3 = v[m+n-2:m+2*n-4]
    v4 = v[m+2*n-4:]
    
    prox = np.zeros((p,))
    prox[:m] = (v1 + b + np.sqrt((v1-b)**2+4/beta))/2
    prox[m:m+n-2] = (1-1/(np.maximum(beta*np.sqrt(v2**2+v3**2),1)))*v2
    prox[m+n-2:m+2*n-4] = (1-1/(np.maximum(beta*np.sqrt(v2**2+v3**2),1)))*v3
    prox[m+2*n-4:] = np.maximum(v4,b)
    
    return prox
    
    
D_full = np.diag(np.ones((n,)),k=0)-np.diag(np.ones((n-1,)),k=1)
D = D_full[:-2,:]
E_full = np.diag(np.ones((n-1,)),k=1)-np.diag(np.ones((n-2,)),k=2)
E = E_full[:-2,:]
A = np.vstack((Atilde,D,E,Atilde))
lmax = np.linalg.norm(A,ord=2)**2

I=10000#number of iterations
#ADLPMM1
rho1 = 1
alpha1 = rho1*lmax
beta1 = rho1
x_ADLPMM1 = np.zeros((n,I+1))
obj_ADLPMM1 = np.zeros((I+1,))
obj_ADLPMM1[0] = obj(x_ADLPMM1[:,0],Atilde,b)
y_ADLPMM1 = np.zeros((2*m+2*n-4,))
z_ADLPMM1 = np.zeros((2*m+2*n-4,))

#ADLPMM2
rho2 = 1/np.linalg.norm(A,ord=2)
alpha2 = rho2*lmax
beta2 = rho2
x_ADLPMM2 = np.zeros((n,I+1))
obj_ADLPMM2 = np.zeros((I+1,))
obj_ADLPMM2[0] = obj(x_ADLPMM2[:,0],Atilde,b)
y_ADLPMM2 = np.zeros((2*m+2*n-4,))
z_ADLPMM2 = np.zeros((2*m+2*n-4,))

#CP1
tau1 = 1/np.linalg.norm(A,ord=2)
sigma1 = 1/np.linalg.norm(A,ord=2)
x_CP1 = np.zeros((n,I+1))
obj_CP1 = np.zeros((I+1,))
obj_CP1[0] = obj(x_CP1[:,0],Atilde,b)
y_CP1 = np.zeros((2*m+2*n-4,))
z_CP1 = np.zeros((2*m+2*n-4,))

#CP2
tau2 = 1/np.linalg.norm(A,ord=2)**2
sigma2 = 1
x_CP2 = np.zeros((n,I+1))
obj_CP2 = np.zeros((I+1,))
obj_CP2[0] = obj(x_CP2[:,0],Atilde,b)
y_CP2 = np.zeros((2*m+2*n-4,))
z_CP2 = np.zeros((2*m+2*n-4,))



for i in range(I):
    #Perform ADLPMM1 update
    x_c = x_ADLPMM1[:,i]
    x_next = x_c-rho1/alpha1*A.T@(A@x_c-z_ADLPMM1+1/rho1*y_ADLPMM1)
    z_ADLPMM1 = prox_h2(rho1,b,A@x_next+1/rho1*y_ADLPMM1)
    y_ADLPMM1 = y_ADLPMM1+rho1*(A@x_next-z_ADLPMM1)
    x_ADLPMM1[:,i+1] = x_next
    #compute objective function
    obj_ADLPMM1[i+1] = obj(x_next,Atilde,b)
    
    #Perform ADLPMM2 update
    x_c = x_ADLPMM2[:,i]
    x_next = x_c-rho2/alpha2*A.T@(A@x_c-z_ADLPMM2+1/rho2*y_ADLPMM2)
    z_ADLPMM2 = prox_h2(rho2,b,A@x_next+1/rho2*y_ADLPMM2)
    y_ADLPMM2 = y_ADLPMM2+rho2*(A@x_next-z_ADLPMM2)
    x_ADLPMM2[:,i+1] = x_next
    #compute objective function
    obj_ADLPMM2[i+1] = obj(x_next,Atilde,b)
    
    #Perform CP1 update
    x_c = x_CP1[:,i]
    x_next = x_c-tau1*A.T@y_CP1
    y_CP1 = y_CP1+sigma1*A@(2*x_next-x_c)-sigma1 * prox_h2(sigma1,b,(y_CP1+sigma1*A@(2*x_next-x_c))/sigma1)
    x_CP1[:,i+1] = x_next
    #compute objective function
    obj_CP1[i+1] = obj(x_next,Atilde,b)
    
    #Perform CP2 update
    x_c = x_CP2[:,i]
    x_next = x_c-tau2*A.T@y_CP2
    y_CP2 = y_CP2+sigma2*A@(2*x_next-x_c)-sigma2 * prox_h2(sigma2,b,(y_CP2+sigma2*A@(2*x_next-x_c))/sigma2)
    x_CP2[:,i+1] = x_next
    #compute objective function
    obj_CP2[i+1] = obj(x_next,Atilde,b)
    

#plot
Ivec = np.array(range(I+1))
I_draw = 500
IADLPMM1_OK  = obj_ADLPMM1==obj_ADLPMM1
IADLPMM2_OK  = obj_ADLPMM2==obj_ADLPMM2
ICP1_OK  = obj_CP1==obj_CP1
ICP2_OK  = obj_CP2==obj_CP2

plt.figure()
Fct_opt = min(np.min(obj_ADLPMM1[IADLPMM1_OK]),np.min(obj_ADLPMM2[IADLPMM2_OK]),np.min(obj_CP1[ICP1_OK]),np.min(obj_CP2[ICP2_OK]))

plt.semilogy(Ivec[IADLPMM1_OK],obj_ADLPMM1[IADLPMM1_OK]-Fct_opt,'b',label = 'ADLPMM v1')
plt.semilogy(Ivec[IADLPMM2_OK],obj_ADLPMM2[IADLPMM2_OK]-Fct_opt,'r',label = 'ADLPMM v2')
plt.semilogy(Ivec[ICP1_OK],obj_CP1[ICP1_OK]-Fct_opt,'g',label = 'CP v1')
plt.semilogy(Ivec[ICP2_OK],obj_CP2[ICP2_OK]-Fct_opt,'c',label = 'CP v2')
plt.show
plt.xlim((0,I_draw))
plt.ylim((1e-8,1e2))
plt.legend()
plt.savefig('part3_ex2_fig1.pdf')

def obj2(x,Atilde,b):
    return x.T@x/2-np.sum(np.log(Atilde@x-b))+np.sum(np.sqrt((x[:-2]-x[1:-1])**2+(x[1:-1]-x[2:])**2))

#ACP1
tau_init = 1/np.linalg.norm(A,ord=2)
sigma_init = 1/np.linalg.norm(A,ord=2)
x_ACP = np.zeros((n,I+1))
obj_ACP = np.zeros((I+1,))
obj_ACP[0] = obj2(x_ACP[:,0],Atilde,b)
y_ACP = np.zeros((2*m+2*n-4,))
x_prev = np.zeros((n,))
theta = 0
sigma = sigma_init
tau = tau_init

#FDPG
t = 1
L = np.linalg.norm(A,ord=2)**2
x_FDPG = np.zeros((n,I+1))
obj_FDPG = np.zeros((I+1,))
obj_FDPG[0] = obj2(x_FDPG[:,0],Atilde,b)
y_FDPG = np.zeros((2*m+2*n-4,))
x_prev = np.zeros((n,))
w_FDPG = np.zeros((2*m+2*n-4,))

for i in range(I):
    #Perform ACP update
    x_c = x_ACP[:,i]
    y_ACP = y_ACP+sigma*A@(x_c+theta*(x_c-x_prev))-sigma * prox_h2(sigma,b,(y_ACP+sigma*A@(x_c+theta*(x_c-x_prev)))/sigma)
    x_next = (x_c-tau*A.T@y_ACP)/(tau+1)
    theta = 1/np.sqrt(1+tau)
    tau = theta * tau
    sigma = sigma/theta
    x_prev = x_c
    
    x_ACP[:,i+1] = x_next
    #compute objective function
    obj_ACP[i+1] = obj2(x_next,Atilde,b)
    
    #Perform FDPG update
    u = A.T@w_FDPG
    y_next = w_FDPG - 1/L * A@u+1/L*prox_h2(1/L,b,A@u-L*w_FDPG)
    t_next = (1+np.sqrt(1+4*t))/2
    w_FDPG = y_next + (t_next-1)/t*(y_next-y_FDPG)
    
    t=t_next
    y_FDPG = y_next
    x_FDPG[:,i+1] = A.T@y_next
    #compute objective function
    obj_FDPG[i+1] = obj2(x_FDPG[:,i+1] ,Atilde,b)
    
IACP_OK  = obj_ACP==obj_ACP
IFDPG_OK  = obj_FDPG==obj_FDPG

I_draw=I
plt.figure()
Fct_opt = min(np.min(obj_ACP[IACP_OK]),np.min(obj_FDPG[IFDPG_OK]))
plt.semilogy(Ivec[IACP_OK],obj_ACP[IACP_OK]-Fct_opt,'b',label = 'ACP')
plt.semilogy(Ivec[IFDPG_OK],obj_FDPG[IFDPG_OK]-Fct_opt,'r',label='FDPG')
plt.xlim((0,I_draw))
plt.ylim((1e-14,1e2))
plt.legend()
plt.savefig('part3_ex2_fig3.pdf')    