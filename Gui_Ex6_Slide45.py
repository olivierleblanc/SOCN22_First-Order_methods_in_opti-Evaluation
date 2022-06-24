# -*- coding: utf-8 -*-
"""
Created on Wed Jun  8 14:19:12 2022

@author: gthiran
"""

import numpy as np
import matplotlib.pyplot as plt

#projection on a box
def proj_box(l,u,x):
    return np.minimum(np.maximum(x,l),u)

def error_fct(a,b,l,u,x,mu):
    y = proj_box(l,u,x-mu*a)
    error = a@y-b
    return error

#projection on the intersection of an hyperplane and a box
def proj_H_inter_box(a,b,l,u,x):
    #start with guesses for mu-levels
    mu_low = -1
    mu_high = 1
    
    #check that the levels give respectively negative and positive values
    #for the function error = a@y-1 with y = proj_box(l,u,x-mu*a)

    #positive for low bound 
    j=0
    j_max =100
    error_l  = -1
    while (error_l<0) & (j<j_max) :
        mu_low = mu_low*2 #more negative (always done in first iteration but not important)
        error_l=error_fct(a,b,l,u,x,mu_low)
        j=j+1
        
    #negative for low bound 
    k=0
    k_max =10
    error_h  = 1
    while (error_h>0) & (k<k_max) :
        mu_high = mu_high*2 #more negative (always done in first iteration but not important)
        error_h=error_fct(a,b,l,u,x,mu_high)
        k=k+1
    
    #mu_low lead to positive value of the error, 
    i = 0
    i_max = 100
    tol = 1e-8
    error  = 2*tol
    while (np.abs(error)>tol) & (i<i_max) :
        mu_mid = (mu_low+mu_high)/2
        error = error_fct(a,b,l,u,x,mu_mid)
        
        if error>0:
            mu_low = mu_mid
        else:
            mu_high = mu_mid
            
        i=i+1
        
    #Compute the solution with the good level
    y= proj_box(l,u,x-mu_mid*a)
    return y


x = np.array([2,1,4,1,2,1])
N = len(x)
a = np.ones((N,))
b=3
l=np.zeros((N,))
u = 2*np.ones((N,))
y = proj_H_inter_box(a,b,l,u,x)
prox = x-y
            
    