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

def soft_thres(eta,x):
    return np.maximum(np.abs(x)-eta,0)*np.sign(x)


I=100#number of iterations
x_PS = np.zeros((n,I+1))
obj_PS = np.zeros((I+1,))
obj_PS[0] = np.linalg.norm(A@x_PS[:,0]-b,ord=1)+0.5*np.linalg.norm(x_PS[:,0],ord=2)**2

x_FDPG = np.zeros((n,I+1))
obj_FDPG = np.zeros((I+1,))
obj_FDPG[0] = np.linalg.norm(A@x_FDPG[:,0]-b,ord=1)+0.5*np.linalg.norm(x_FDPG[:,0],ord=2)**2
L = np.linalg.norm(A,ord=2)**2
#initialization
w_next = np.zeros((m,))
y_next = np.zeros((m,))
t_next = 1

for i in range(I):
    #Perform PS update
    x_c = x_PS[:,i]
    t = 1/np.sqrt(i+1)
    x_next = (x_c-t*np.transpose(A)@np.sign(A@x_c-b))/(t+1)
    x_PS[:,i+1] = x_next
    #compute objective function
    obj_PS[i+1] = np.linalg.norm(A@x_next-b,ord=1)+0.5*np.linalg.norm(x_next,ord=2)**2
    
    
    #Perform FDPG
    y = y_next
    t=t_next
    w=w_next
    
    u = np.transpose(A)@w
    y_next = w-1/L*A@u+1/L*  soft_thres(L,A@u-L*w-b)+ b/L
    t_next = (1+np.sqrt(1+4*t**2))/2
    w_next = y_next+(t-1)/t_next*(y_next-y)
    
    x_FDPG[:,i+1] = np.transpose(A)@y
    #compute objective function
    obj_FDPG[i+1] = np.linalg.norm(A@u-b,ord=1)+0.5*np.linalg.norm(u,ord=2)**2

#plot
plt.close('all')
plt.figure()
plt.semilogy(obj_PS,'b')
plt.semilogy(obj_FDPG,'r')
plt.show

    