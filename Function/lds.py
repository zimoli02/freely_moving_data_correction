import numpy as np
import pandas as pd
import math
import scipy.stats as ss

def SimulateData(a1, p1, n, e, num):
    a, y = np.zeros(num+1), np.zeros(num)
    a[0] = np.random.normal(a1, p1**0.5)
    
    for i in range(num):
        y[i] = a[i] + np.random.normal(0, e**0.5)
        a[i+1] = a[i] + np.random.normal(0, n**0.5)
    
    return y


def SimulateData_Poisson(m0, v0, n, num):
    x, y = np.zeros(num+1), np.zeros(num)
    x[0] = np.random.normal(m0, v0**0.5)

    for i in range(num):
        y[i] = np.random.poisson(np.exp(x[i]))
        x[i+1] = x[i] + np.random.normal(0, n**0.5)

    return y

def SimulateData_CorrelateNoise(a1, p1, n, e, w, num):
    a, E, y = np.zeros(num+1), np.zeros(num+1), np.zeros(num)
    a[0] = np.random.normal(a1, p1**0.5)
    E[0] = e
    
    for i in range(num):
        y[i] = a[i] + E[i]
        a[i+1] = a[i] + np.random.normal(0, n**0.5)
        E[i+1] = E[i] + np.random.normal(0, w**0.5)
    
    return y, E[:-2]


def SteadyStateVariance(e,n):
    q = n/e
    if q >= 0: return (q + np.sqrt(q**2 + 4*q))*e/2
    else: return 'no steady state'

def ApplyFilter(a1, p1, e, n, y, steady_state = False):
    timepoints = len(y)
    a, p, F = np.zeros(timepoints+1, dtype = np.double), np.zeros(timepoints+1, dtype = np.double), np.zeros(timepoints, dtype = np.double)
    A, P = np.zeros(timepoints, dtype = np.double), np.zeros(timepoints, dtype = np.double)
    p_ = SteadyStateVariance(e,n)
    K_ = p_/(p_+ e)
    if p_ == 'no steady state': return 'No Steady State'
    
    a[0], p[0] = a1, p1
    flag = False
    for i in range(0, timepoints):
        F[i] = p[i] + e
        if flag == False:
            K = p[i]/F[i]
            if steady_state and abs(p[i]-p_)<=1e-3: flag = True
        else:
            K = K_

        if np.isnan(y[i]): 
            A[i] = a[i]
            P[i] = p[i]
        else:
            A[i] = a[i] + K*(y[i] - a[i])
            P[i] = K*e

        a[i+1] = A[i]
        p[i+1] = P[i] + n
    
    return a[0:-1], p[0:-1],F,A,P

def ApplySmoothing(a, p, F, y, v, e):
    timepoints=len(y)
    r, N = np.zeros(timepoints, dtype = np.double), np.zeros(timepoints, dtype = np.double)
    A_, P_ = np.zeros(timepoints, dtype = np.double), np.zeros(timepoints, dtype = np.double)

    for i in range(timepoints-1, 0, -1):
        if np.isnan(y[i]): r[i-1], N[i-1] = r[i], N[i]
        else:
            r[i-1] = (v[i]/F[i]) + (e/F[i])*r[i]
            N[i-1] = (1/F[i]) + ((e/F[i])**2)*N[i]
    r_ = (v[0]/F[0]) + (e/F[0])*r[0]
    N_ = (1/F[0]) + ((e/F[0])**2)*N[0]
    
    A_[0] = a[0] + p[0]*r_
    P_[0] = p[0] - (p[0]**2)*N_
    for i in range(1,timepoints):
        A_[i] = a[i] + p[i]*r[i-1]
        P_[i] = p[i] - (p[i]**2)*N[i-1]
    
    return A_, P_, r, N


def dLikelihood(y, paras, para = 'e'):
    a1, p1, e, n = paras['a1'], paras['p1'],paras['e'],paras['n']
    a, p, F, A, P = ApplyFilter(a1, p1, e, n, y)
    v = [y[i] - a[i] for i in range(len(y))]
    
    dF, dv = [], []
    dF.append(1)
    dv.append(0)
    
    dL = (1/(F[0]**2)*(F[0]-v[0]**2)*dF[0]) + 2*F[0]*v[0]*dv[0]
    for i in range(1, len(F)):
        if para == 'e': 
            dF.append((e/F[i-1])**2*dF[i-1]-2*e/F[i-1]+2)
            dv.append(dv[i-1] - ((1-e/(F[i-1])**2)*dv[i-1] + e*v[i-1]*dF[i-1]/(F[i-1]**2)-v[i-1]/F[i-1]))
        elif para == 'n':
            dF.append((e/F[i-1])**2*dF[i-1]+1)
            dv.append(dv[i-1] - ((1-e/F[i-1])*dv[i-1] + e*v[i-1]*dF[i-1]/(F[i-1]**2)))
        dL += 1/(F[i]**2)*((F[i]-v[i]**2)*dF[i] + 2*F[i]*v[i]*dv[i])

    return -dL/2

def LogLikelihood(y, paras):
    a1, p1, e, n = paras['a1'], paras['p1'],paras['e'],paras['n']
    a, p, F, A, P = ApplyFilter(a1, p1, e, n, y)
    v = [y[i] - a[i] for i in range(len(y))]
    
    L = len(F)*math.log(2*math.pi)
    for i in range(len(F)): L += math.log(F[i]) + v[i]**2/F[i]
    
    return -L/2

def MaximizeLikelihood(y, paras, para = 'e'):
    L, P, grad = [], [], []
    d = 0.0001

    iter = 1
    step = 5
    stop = False
    while stop == False:
        paras[para] += step
        paras_ = paras.copy()
        paras_[para] += d
        
        L.append(LogLikelihood(y, paras))
        P.append(paras[para])
        grad.append((LogLikelihood(y, paras_) - LogLikelihood(y, paras))/d)
        
        print("iter = "+str(iter), " L = "+str(L[-1]), " Para = "+str(P[-1]), " Gradient = "+str(grad[-1]))
        iter += 1
        
        if abs(grad[-1]) <= 1e-6: stop = True
        elif len(grad)>=2 and (grad[-1]*grad[-2]) < 0: step = (P[-1]-P[-2])/2*(grad[-1]/abs(grad[-1]))
        else: step = 5*(grad[-1]/abs(grad[-1]))
        
    return L, P, grad

def DiagnosticCheck(a1, p1, e, n, y, k=10):
    N = len(y)-1
    a, p, F, A, P = ApplyFilter(a1, p1, e, n, y)
    v = [y[i] - a[i] for i in range(len(y))]

    E = np.array([v[i]/np.sqrt(F[i]) for i in range(1,len(y))])
    m,H,c,Q = np.zeros(4), np.zeros(N), np.zeros(k+1), np.zeros(k+1)

    m[0] = np.mean(E) #mean
    m[1] = np.mean([(E[j]-m[0])**2 for j in range(N)]) #variance
    m[2] = np.mean([(E[j]-m[0])**3 for j in range(N)])
    m[3] = np.mean([(E[j]-m[0])**4 for j in range(N)])
    S, K = m[2]/(np.sqrt(m[1]**3)), m[3]/(m[1]**2) #Skewness and Kurtosis
    NN = N*(S**2/6 + (K-3)**2/24)

    H[0] = E[-1]**2 / (E[0] **2)
    for i in range(1,N):
        H[i] = (np.sum([E[j]**2 for j in range(N-i, N)]))/(np.sum([E[j]**2 for j in range(i)]))

    for i in range(k+1):
        c[i] = np.sum([(E[j] - m[0])*(E[j-i]-m[0]) for j in range (i+1, N)])/(N*m[1])
        Q[i] = N*(N+2)*np.sum([c[j]**2/(N-j-1) for j in range(i)])

    p_S, p_K = ss.norm.sf(S, loc=0, scale=(6/N)**0.5),ss.norm.sf(K, loc = 3, scale = (24/N)*0.05)
    p_NN = ss.chi2.sf(NN, 2)
    p_H = [ss.f.sf(H[i], i+1, i+1) for i in range(len(H))]
    p_Q = [ss.chi2.sf(q, 2) for q in Q]
    return E, (S,p_S), (K, p_K), (NN, p_NN), (H, p_H), c[1:], (Q[1:],p_Q[1:])

def TheoreticalQuantile(x):
    (res_o_values, (slope, intercept, r)) = ss.probplot(x, dist="norm")
    return res_o_values