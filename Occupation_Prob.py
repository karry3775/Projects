# Program to find out occupational probability at time t given initial occupation probability and
# State Transition probability matrix

import numpy as np
from numpy.linalg import matrix_power as mp
# N = int(input("Enter the number of states in your Markov chain"))
N = 5

# Function for populating the state transition probability matrix
# You can have your own state transition probability matrix
def STP(N): 
    P = np.random.rand(N,N)
    P = P/np.sum(P,axis=1).reshape(P.shape[0],1)
    return P
    
# Function for populating the initial probability distribution
# You can have your own initial probability distribution   
def IPD(N):
    I = np.random.rand(1,N)
    I = I/np.sum(I,axis=1).reshape(I.shape[0],1)
    return I
  
# Function to calculate final occupation probability distribution after t steps  
def FPD(P,I,t):
    Pt = mp(P,t)
    F = np.dot(I,Pt)
    return F
    
    
P = STP(N) 
I = IPD(N)
print("Your initial distribution vector is:")
print(I)

t = int(input("Enter the number of time steps: "))
F = FPD(P,I,t)

print("Your final probability distribution is: ")
print(F)
