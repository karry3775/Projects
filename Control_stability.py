import numpy as np
from numpy import linalg as LA

A = np.array([[1,0,0],[0,1,0],[0,0,1]]) #Your system dynamics goes here!
eig_val ,eig_vect = LA.eig(A)
eig_val = eig_val.reshape(len(eig_val),1)

# Making the Diagonal matrix
D = np.zeros((len(eig_val),len(eig_val)))
for i in range(len(eig_val)):
    D[i,i] = eig_val[i,0]

# Taking the choice
flag= 1
while(flag==1):
    choice = str(input("Is your system continous(C) or Discrete(D): enter C or D"))
    if choice == 'C'  or choice =='c':
        flag=0
        break
    elif choice == 'D' or choice =='d':
        flag=0
        break
    else:
        flag=1
# Telling about the stability
if choice == 'c' or choice =='C':
  counter = 0
  for i in range(len(eig_val)):
    if eig_val[i] > 0:
      counter +=1
  if counter>0:
    print("Sorry your system isn't stable")
  else:
    print("Your system is stable! Enjoy")
elif choice=='d' or choice=='D':
  counter=0
  for i in range(len(eig_val)):
    if eig_val[i]>1:
      counter+=1
  if counter>0:
    print("Sorry your system isn't stable")
  else:
    print("Your system is stable! Enjoy")
    
