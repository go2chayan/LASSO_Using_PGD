# -*- coding: utf-8 -*-
"""
Created on Wed Dec 03 14:52:54 2014

@author: Md.Iftekhar
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

# gradient of f(x)    
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
    
    lamda = np.sqrt(2*n*np.log(p)).tolist()
    #lamda = 1.
    
    # For Proximal Gradient Descent
    #xk = np.random.rand(p,1) # Initialize with random
    xk = np.zeros((p,1))      # Initialize with zero

    # For Accelerated Proximal Gradient Descent    
    xk_acc = xk.copy()
    yk_acc = xk_acc
    tk_acc = 1
    stopUpdate_acc = False
    Dobj_acc = 0
    for k in xrange(kMax):

        # ------------------------- Proximal GD -----------------------------        
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
        
        # Change in the value of objective funtion for this iteration
        Dobj = np.linalg.norm(obj(A,x_kplus1,b,lamda) - obj(A,xk,b,lamda))
        # print 'k:',k, ' obj = ', obj(A,x_kplus1,b,lamda), 'Change = ',Dobj
        
        # Update xk
        xk = x_kplus1.copy()
        # --------------------------------------------------------------------

        # =================== Accelerated Proximal GD ========================
        Gammak = 0.01
    
        # Line search
        while True:
            # Accelerated Gradient Descent (GD) Step
            x_kplus1_acc = yk_acc - Gammak*grf(A,yk_acc,b)        
            if f(A,x_kplus1_acc,b) <= m(x_kplus1_acc,xk_acc,A,b,Gammak):
                break
            else:
                Gammak = beta*Gammak
        # Proximal Operation (Shrinkage)
        x_kplus1_acc = proxNorm1(x_kplus1_acc,Gammak*lamda)   
        t_kplus1_acc = 0.5 + 0.5*np.sqrt(1+4*(tk_acc**2.))
        y_kplus1_acc = x_kplus1_acc + ((tk_acc - 1)/(tk_acc + 1))*\
        (x_kplus1_acc - xk_acc)

        # Change in the value of objective funtion for this iteration
        Dobj_acc = np.linalg.norm(obj(A,x_kplus1_acc,b,lamda) - \
        obj(A,xk_acc,b,lamda))
        
        # Update
        xk_acc = x_kplus1_acc
        tk_acc = t_kplus1_acc
        yk_acc =  y_kplus1_acc
        
        # ====================================================================        

        # Terminating Condition        
        if (Dobj < 1) and (Dobj_acc < 1):
            break

        # Graphical Display - plot x and reconstructed x
        pp.figure('Sparse code')
        pp.clf()        
        pp.subplot(311)    
        pp.plot(xStar)
        pp.title('Original x')
        pp.subplot(312)
        pp.plot(xk)
        pp.title('Reconstructed x (Proximal GD)')
        pp.subplot(313)
        pp.plot(xk_acc)
        pp.title('Reconstructed x (Accelerated Proximal GD)')
        pp.draw()
        pp.pause(0.1)

        # Graphical Display - plot objective function values
        if k > 5:
            pp.figure('Objective function values')
            #pp.subplot(211)
            #pp.scatter(k,-1*np.log(obj(A,xk,b,lamda)),c = 'b')
            #pp.scatter(k,-1*np.log(obj(A,xk_acc,b,lamda)),c = 'r')
            pp.scatter(k,obj(A,xk,b,lamda),c = 'b')
            pp.scatter(k,obj(A,xk_acc,b,lamda),c = 'r')
            pp.title(\
            '- log of Objective Functions (Blue = PGD, Red = Accelerated PGD)')
            #pp.subplot(212)
#            pp.scatter(k,-1*np.log(Dobj),c = 'b')
#            pp.scatter(k,-1*np.log(Dobj_acc),c = 'r')
#            print 'k:',k, ' Change (Proximal) = ', Dobj, ' Change (Accelerated Proximal) = ',Dobj_acc
#            pp.title('-log of change in Objective Functions (Blue = PGD, Red = Accelerated PGD)')        
            pp.draw()
            pp.pause(0.1)        
        
    pp.show()
        
if __name__ == "__main__":
    main()

