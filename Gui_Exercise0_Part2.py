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
import matplotlib.pyplot as plt
m = 100
n = 120
a = np.arange(0,m)+1
b = np.arange(0,n)+0.5
A = np.sin(10 * np.outer(a,b)**3)
xi = np.sin(31 * np.arange(1,n+1)**3)
b = A @ xi
lambda1 = 2
lambda2 = 0.5
At = np.transpose(A)

def soft_thres(eta,x):
    return np.maximum(np.abs(x)-eta,0)*np.sign(x)

def obj(x,lambda1,lambda2):
    return 0.5*np.linalg.norm(A@x-b,ord=2)**2+0.5*lambda1*np.linalg.norm(x,ord=2)**2+ lambda2* np.linalg.norm(x,ord=1)


I=100#number of iterations
#Proximal gradient
x_PG = np.zeros((n,I+1))
obj_PG = np.zeros((I+1,))
obj_PG[0] = obj(x_PG[:,0],lambda1,lambda2) 

L1 = np.linalg.norm(A,ord=2)**2+lambda1
Sig1 = lambda1
k1 = L1/Sig1

#FISTA
x_FISTA = np.zeros((n,I+1))
obj_FISTA = np.zeros((I+1,))
obj_FISTA[0] = obj(x_FISTA[:,0],lambda1,lambda2) 
y_FISTA = x_FISTA[:,0]
t_FISTA=1

#V-FISTA-1
x_V_FISTA_1 = np.zeros((n,I+1))
obj_V_FISTA_1 = np.zeros((I+1,))
obj_V_FISTA_1[0] = obj(x_V_FISTA_1[:,0],lambda1,lambda2) 
Sig1 = lambda1
k1 = L1/Sig1
fact1 = (np.sqrt(k1)-1)/(np.sqrt(k1)+1)
y_V_FISTA_1 = x_V_FISTA_1[:,0]

#V-FISTA-2
x_V_FISTA_2 = np.zeros((n,I+1))
obj_V_FISTA_2 = np.zeros((I+1,))
obj_V_FISTA_2[0] = obj(x_V_FISTA_2[:,0],lambda1,lambda2) 
L2 = np.linalg.norm(A,ord=2)**2
Sig2 = lambda1
k2 = L2/Sig2
fact2 = (np.sqrt(k2)-1)/(np.sqrt(k2)+1)
y_V_FISTA_2 = x_V_FISTA_2[:,0]

#initialization


for i in range(I):
    #Perform PS update
    x_c = x_PG[:,i]
    x_next = soft_thres(lambda2/L1,x_c-1/L1*At@(A@x_c-b)-lambda1/L1*x_c)
    x_PG[:,i+1] = x_next
    #compute objective function
    obj_PG[i+1] = obj(x_next,lambda1,lambda2) 
    
    #Perform FISTA update
    x_c = x_FISTA[:,i]
    x_next = soft_thres(lambda2/L1,y_FISTA-1/L1*At@(A@y_FISTA-b)-lambda1/L1*y_FISTA)
    t_FISTA_next = (1+np.sqrt(1+4*t_FISTA**2))/2
    y_FISTA = x_next+(t_FISTA-1)/t_FISTA_next*(x_next - x_c)
    t_FISTA = t_FISTA_next
    x_FISTA[:,i+1] = x_next
    #compute objective function
    obj_FISTA[i+1] = obj(x_next,lambda1,lambda2) 
    
    #Perform V-FISTA-1 update
    x_c = x_V_FISTA_1[:,i]
    x_next = soft_thres(lambda2/L1,y_V_FISTA_1-1/L1*At@(A@y_V_FISTA_1-b)-lambda1/L1*y_V_FISTA_1)
    y_V_FISTA_1 = x_next+fact1*(x_next - x_c)
    x_V_FISTA_1[:,i+1] = x_next
    #compute objective function
    obj_V_FISTA_1[i+1] = obj(x_next,lambda1,lambda2) 
    
    #Perform V-FISTA-2 update
    x_c = x_V_FISTA_2[:,i]
    x_next = soft_thres(lambda2/(lambda1+L2),1/(1+lambda1/L2)*(y_V_FISTA_2-1/L2*At@(A@y_V_FISTA_2-b)))
    y_V_FISTA_2 = x_next+fact2*(x_next - x_c)
    x_V_FISTA_2[:,i+1] = x_next
    #compute objective function
    obj_V_FISTA_2[i+1] = obj(x_next,lambda1,lambda2) 
    

#plot
plt.close('all')
plt.figure()
Fct_opt = min(np.min(obj_PG),np.min(obj_FISTA),np.min(obj_V_FISTA_1),np.min(obj_V_FISTA_2))
plt.semilogy(obj_PG-Fct_opt,'b')
plt.semilogy(obj_FISTA-Fct_opt,'r')
plt.semilogy(obj_V_FISTA_1-Fct_opt,'g')
plt.semilogy(obj_V_FISTA_2-Fct_opt,'c')
plt.show

    