# -*- coding: utf-8 -*-
"""
Created on Tue Dec  2 18:02:05 2014

@author: Md. Iftekhar Tanveer (itanveer@cs.rochester.edu)
"""
import numpy as np
import matplotlib.pyplot as pp

# Objective function: f(x) + lambda*norm1(x)
def obj(A,x,b,lamda):
    assert(np.size(x,0)==np.size(A,1) and np.size(A,0) == np.size(b,0) and \
    np.size(x,1)== np.size(b,1) == 1 and np.isscalar(lamda))
    return f(A,x,b) + lamda*np.sum(np.abs(x))

# f(x) = (1/2)||Ax-b||^2
def f(A,x,b):
    assert(np.size(x,0)==np.size(A,1) and np.size(A,0) == np.size(b,0) and \
    np.size(x,1)== np.size(b,1) == 1)
    Ax_b = A.dot(x) - b
    return 0.5*(Ax_b.T.dot(Ax_b))

# gradient of f(x)= A'(Ax - b)   
def grf(A,x,b):
    assert(np.size(x,0)==np.size(A,1) and np.size(A,0) == np.size(b,0) and \
    np.size(x,1)== np.size(b,1) == 1)
    return A.T.dot(A.dot(x) - b)
    
# Model function evaluated at x and touches f(x) in xk
def m(x,xk,A,b,GammaK):
    assert(np.size(xk,0) == np.size(x,0) == np.size(A,1) \
    and np.size(A,0) == np.size(b,0) and \
    np.size(xk,1) == np.size(x,1) == np.size(b,1) == 1 and np.isscalar(GammaK))
    innerProd = grf(A,xk,b).T.dot(x - xk)
    xDiff = x - xk
    return f(A,xk,b) + innerProd + (1.0/(2.0*GammaK))*xDiff.T.dot(xDiff)

# Shrinkage or Proximal operation
def proxNorm1(y,lamda):
    assert(np.size(y,1)==1)
    return np.sign(y)*np.maximum(np.zeros(np.shape(y)),np.abs(y)-lamda)

def main():
    # Define parameters. Size of A is n x p
    p = 1000
    n = 500
    kMax = 500   # Number of iteration
    beta = 0.75 # decreasing factor for line search
    # Generate the sparse vector xStar
    # and Randomly set 20 elements
    xStar = np.zeros((p,1))
    xStar[np.floor(p*np.random.rand(20,1)).astype(np.int)]=1
    xStar = xStar*np.random.normal(0,10,(p,1))

    # Generate A and b. b = Ax + error
    A = np.random.randn(n,p)
    b = A.dot(xStar) + np.random.randn(n,1)
    # This lamda is too large and making the x vector zero every time
    lamda = np.sqrt(2*n*np.log(p)).tolist()
    #lamda = 1.
   
    # Proximal Gradient Descent
    xk = np.random.rand(p,1) # Initialize with random
    #xk = np.zeros((p,1))      # Initialize with zero
    
    for k in xrange(kMax):
        Gammak = 0.01
        #Gammak = 1/np.linalg.norm(A.T.dot(A))      
        
        # Line search
        while True:
            #print 'trying stepsize = ', "{0:0.2e}".format(Gammak),
            x_kplus1 = xk - Gammak*grf(A,xk,b)        # Gradient Descent (GD) Step
            if f(A,x_kplus1,b) <= m(x_kplus1,xk,A,b,Gammak):
                #print ' success'
                break
            else:
                #print ' Fail ',
                Gammak = beta*Gammak
        x_kplus1 = proxNorm1(x_kplus1,Gammak*lamda)   # Proximal Operation (Shrinkage)
        
        # Terminating Condition        
        Dobj = np.linalg.norm(obj(A,x_kplus1,b,lamda) - obj(A,xk,b,lamda))
        print 'k:',k, ' obj = ', obj(A,x_kplus1,b,lamda), 'Change = ',Dobj
        if(Dobj<0.1):
            break

        # Update xk
        xk = x_kplus1 
        
        # Graphical Display        
        pp.figure(2)
        pp.clf()        
        pp.subplot(211)    
        pp.plot(xStar)
        pp.title('Original x')
        pp.subplot(212)
        pp.plot(xk)
        pp.title('Reconstructed x')
        pp.draw()
        pp.pause(0.1)
    pp.show()
        
if __name__ == "__main__":
    main()

