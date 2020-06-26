# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 20:06:42 2020

@author: VANSHIKA
"""

import numpy as np

# x and y are datasets
# Use your own datasets for linear discriminant algorithms
# Experiment with different datasets and number of separating dimensions to see how it affects convergence.

####################################### 1 #####################################

def perceptron_batch(x, y, alpha, t):
        itr = -1
        w = [0.000 for i in range(50)]        
        n = 0 
        w0 = 0.000
        c = 0
        while n < t:         
            for i in range(1000):                 
                f = np.dot(x[i], w)                      
                f = f + w0
                if f > 0:                               
                    yhat = 1                               
                else:                                   
                    yhat = 0
                if(int(y[i]) != yhat):
                    c+=1
                for j in range(50):             
                    w[j] = w[j] + alpha * (y[i] - yhat) * x[i][j]
                    w0 = w0 + alpha * (y[i] - yhat)
            if(c == 0):
                itr = n
                break
            c = 0
            n+= 1 
        return w, itr
    
q11,q12=perceptron_batch(x, y, 0.5, 10)
# Returns the hyper-plane and number of iterations when perfect classification is achieved

##################################### 2 #######################################

def perceptron_iterative(x, y, c, e, t):
        itr = -1
        w = [0 for i in range(50)]      
        n = 0 
        w0 = 0
        c = 0
        while n < t:
            for i in range(1000):                 
                f = np.dot(x[i], w)                      
                f = f + w0
                if f > 0:                               
                    yhat = 1                               
                else:                                   
                    yhat = 0
                if(int(y[i]) != yhat):
                    c+= 1
                for j in range(50):             
                    w[j] = w[j] + (c/(n+e)) * (y[i] - yhat) * x[i][j]
                    w0 = w0 + (c/(n+e)) * (y[i] - yhat)
            if(c==0):
                itr = n
                break
            c = 0
            n += 1
        return w,itr

q21,q22=perceptron_iterative(x, y, 1, 5, 10)
# Returns the hyper-plane and number of iterations when perfect classification is achieved

#################################### 3 ########################################

def pocket(x, y, z, alpha, t):
        w = [0.000 for i in range(50)]        
        n = 0 
        w0 = 0
        ws = w
        hs = 0
        h = 0
        while n < t:                                  
            for i in range(40):                 
                f = np.dot(x[i], w)                      
                f = f + w0
                if f > z:                               
                    yhat = 1                               
                else:                                   
                    yhat = 0
                for j in range(50):             
                    w[j] = w[j] + alpha * (y[i] - yhat) * x[i][j]
                    w0 = w0 + alpha * (y[i]-yhat)
                if(y[i] == yhat):
                    h = h + 1
                if(h>hs):
                    hs = h
                    ws = w 
                n += 1 
            h = 0         
        return ws,hs

q31,q32=pocket(x, y, 0, 0.500, 100)
#  Returns the hyper-plane and number of vectors classified correctly  (or incorrectly)

################################### 4 #########################################

def predict(row, coefficients):
    yhat = coefficients[0]
    for i in range(len(row) - 1):
        yhat += coefficients[i + 1] * row[i]
    return yhat
 
def lms_sgd(train, l_rate, t):
    coef = [0.0 for i in range(51)]
    for j in range(t):
        sum_error = 0
        for row in train:
            yhat = predict(row, coef)
            error = yhat - row[-1]
            sum_error += error**2
            coef[0] = coef[0] - l_rate * error
            for i in range(len(row) - 1):
                coef[i + 1] = coef[i + 1] - l_rate * error * row[i]
        #print('>epoch=%d, lrate=%.3f, error=%.3f' % (j, l_rate, sum_error))
    return coef,t

q41,q42 = lms_sgd(x, 0.001, 14)
# Returns the hyper-plane and the number of vectors classified correctly  (or incorrectly) 

###############################################################################