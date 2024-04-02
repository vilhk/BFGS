#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np

#line search algorithm
def ls(f,x,p,grad):
    c1 = 10.0**(-4)
    c2 = 0.9
    a0 = 0.0
    amax = 50.0
    i = 1
    
    def phi(a):
        return f(x+a*p)
    
    def dphi(a):
        return np.dot(grad(x+a*p),p)
    
    def zoom(lo,hi):
        j = 1
        while True:
            a = (lo+hi)/2.0
            if (phi(a)>phi(0.0)+c1*a*dphi(0.0)) or (phi(a)>=phi(lo)):
                hi = a
            else:
                if abs(dphi(a))<=-c2*dphi(0.0):
                    break
                if dphi(a)*(hi-lo)>=0.0:
                    hi = lo
                lo = a
            j = j+1
            if j>10:
                break
        return a
    
    a = 1.0
    while True:
        if (phi(a)>phi(0.0)+c1*a*dphi(0.0)) or (phi(a)>=phi(a0) and i>1):
            a = zoom(a0,a)
            break
        if abs(dphi(a))<=-c2*dphi(0.0):
            break
        if (dphi(a)>=0.0):
            a = zoom(a,a0)
            break
        if a>=amax:
            a = amax
            break
        a0 = a
        a = 1.5*a
        i = i+1
    return a

#BFGS algorithm
def BFGS(f,x):
    eps = np.finfo(float).eps
    I = np.identity(np.size(x))
    H = np.identity(np.size(x))
    k = 0
    maxk = 99
    
    def grad(x):
        ep = np.sqrt(1.1*10**(-16))
        n = np.size(x)
        grad = np.zeros(n)
        e = np.eye(n)
        for i in range(n):
            grad[i] = (f(x+ep*e[i])-f(x))/ep
        return grad

    while np.linalg.norm(grad(x))>10**(-5):
        p = -np.dot(H,grad(x))
        alpha = ls(f,x,p,grad)
        x1 = x+alpha*p
        s = x1-x
        y = grad(x1)-grad(x)
        if np.dot(y.T,s)>eps:
            if k == 0:
                H = ((np.dot(y.T,s))/(np.dot(y.T,y)))*H
            rho = np.dot(y.T,s)**(-1)
            H = (I-rho*np.outer(s,y.T))@H@(I-rho*np.outer(y,s.T))+(rho*np.outer(s,s.T))
        else:
            H = H
        x = x1
        k = k+1
        if k>maxk:
            break
    return x,k


# In[ ]:




