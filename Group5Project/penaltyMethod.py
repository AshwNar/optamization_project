#!/usr/bin/env python
# coding: utf-8

#!/usr/bin/env python
# coding: utf-8
#importing all libraries
import numpy as np
import time
import math
from matplotlib import pyplot as plt

#define a function to evaluate the user specific function
def funcM(x, Q, R, funcInc = 1):
    #switch statement to select question
    #problem 1
    if(Q == 1):
        FM = funcInc*((x[0] - 10)**3 + (x[1] - 20)**3)
        if(((x[0] - 5)**2 + (x[1] - 5)**2 - 100) < 0):
            FM = FM + R*((x[0] - 5)**2 + (x[1] - 5)**2 - 100)**2

        if(((x[0] - 6)**2 + (x[1] - 5)**2 - 82.81) > 0):
            FM = FM + R*((x[0] - 6)**2 + (x[1] - 5)**2 - 82.81)**2
        if(x[0] < 13):
            FM = FM + R*(x[0] - 13)**2
        if(x[0] > 20):
            FM = FM + R*(x[0] - 20)**2
        if(x[1] < 0):
            FM = FM + R*(x[1] - 0)**2
        if(x[1] > 4):
            FM = FM + R*(x[1] - 4)**2
            
        return FM
    
    #problem2
    if(Q == 2):
        FM = funcInc*(((-1*math.sin(math.pi*2*x[0]))**3 * math.sin(2*math.pi*x[1]))/(x[0]**3 * (x[0] + x[1])))
        
        if((x[0]**2 - x[1] + 1) > 0):    
            FM = FM + R * (x[0]**2 - x[1] + 1)**2
        if((1 - x[0] + (x[1] - 4)**2) > 0):
            FM = FM + R*(1 - x[0] + (x[1] - 4)**2)**2
        if(x[0] < 0):
            FM = FM + R*(x[0])**2
        if(x[0] > 10):
            FM = FM + R*(x[0] - 10)**2
        if(x[1] < 0):
            FM = FM + R*(x[1])**2
        if(x[1] > 10):
            FM = FM + R*(x[1] - 10)**2
            
        return FM
    
    #problem 3
    if(Q == 3):
        FM = funcInc*(x[0] + x[1] + x[2])

        if ((-1 + 0.0025*(x[3] + x[5])) > 0):
            FM = FM + R*(-1 + 0.0025*(x[3] + x[5]))**2
    
        if (-1 + 0.0025*(-1*x[3] + x[4] + x[6]) > 0):
            FM = FM + R*(-1 + 0.0025*(-1*x[3] + x[4] + x[6]))**2

        if (-1 + 0.01*(-1*x[5] + x[7]) > 0):
            FM = FM + R*(-1 + 0.01*(-1*x[5] + x[7]))**2
            
        if(-0.0012*x[0] + 1.2e-5*x[0]*x[5] - 0.009999*x[3] + 1 > 0):
            FM = FM + R*(-0.0012*x[0] + 1.2e-5*x[0]*x[5] - 0.009999*x[3] + 1)**2
            
        if(-0.0008*x[1]*x[3] + 0.0008*x[1]*x[6] + x[3] - x[4] > 0):
             FM = FM + R*(-0.0008*x[1]*x[3] + 0.0008*x[1]*x[6] + x[3] - x[4])**2

        if(-0.0000008*x[2]*x[4] + 0.0000008*x[2]*x[7] - 0.002*x[4] + 1 > 0):
             FM = FM + R*(-0.0000008*x[2]*x[4] + 0.0000008*x[2]*x[7] - 0.002*x[4] + 1)**2
        
        if(x[0] < 100):
            FM = FM + R*((x[0] - 100))**2
        if(x[1] < 1000):
            FM = FM + R*((x[1] - 1000))**2
        if(x[2] < 1000):
            FM = FM + R*((x[2] - 1000))**2
        if(x[3] < 10):
            FM = FM + R*((x[3] - 10))**2
        if(x[4] < 10):
            FM = FM + R*((x[4] - 10))**2
        if(x[5] < 10):
            FM = FM + R*((x[5] - 10))**2
        if(x[6] < 10):
            FM = FM + R*((x[6] - 10))**2
        if(x[7] < 10):
            FM = FM + R*((x[7] - 10))**2
        if(x[0] > 10000):
            FM = FM + R*((x[0] - 10000))**2
        if(x[1] > 10000):
            FM = FM + R*((x[1] - 10000))**2
        if(x[2] > 10000):
            FM = FM + R*((x[2] - 10000))**2
        if(x[3] > 1000):
            FM = FM + R*((x[3] - 1000))**2
        if(x[4] > 1000):
            FM = FM + R*((x[4] - 1000))**2
        if(x[5] > 1000):
            FM = FM + R*((x[5] - 1000))**2
        if(x[6] > 1000):
            FM = FM + R*((x[6] - 1000))**2
        if(x[7] > 1000):
            FM = FM + R*((x[7] - 1000))**2
        
        return FM

    #error
    print("Incorrect question number")
    return 0



def constraintViolate(x, Q):
    if(Q == 1):
        g = np.zeros((2))
        g[0] = -1*((x[0] - 5)**2 + (x[1] - 5)**2 - 100)
        g[1] = (x[0] - 6)**2 + (x[1] - 5)**2 - 82.81

        return g
    
    #problem2
    if(Q == 2):
        g = np.zeros((2))   
        g[0] = x[0]**2 - x[1] + 1
        g[1] = 1 - x[0] + (x[1] - 4)**2
            
        return g
    
    #problem 3
    if(Q == 3):
        g = np.zeros((6))
        g[0] = -1 + 0.0025*(x[3] + x[5])
        g[1] = -1 + 0.0025*(-1*x[3] + x[4] + x[6])
        g[2] = -1 + 0.01*(-1*x[5] + x[7])
        g[3] = 100*x[0] - x[0]*x[5] + 833.33252*x[3] - 83333.333
        g[4] = x[1]*x[3] - x[1]*x[6] - 1250*x[3] + 1250*x[4]
        g[5] = x[2]*x[4] - x[2]*x[7] - 2500*x[4] + 1250000
        return g

#function to calculate differential wrt alpha
def partialAlpha(alpha , x, nablaF, Q, R, delta = 1e-10):
    #using central difference formula
    return ((funcM(x - (alpha + delta)*nablaF, Q, R) - funcM(x - (alpha - delta)*nablaF, Q, R))/(2*delta))

#function to find gradient of function
def nablaFuncM(x, Q, NUMBERVAR, R):
    #take small value
    delta = 1e-10
    nablaF = np.zeros((NUMBERVAR)) #declare a zero vector of size number of variables

    #find derivative wrt all dependent variable
    for i in range(NUMBERVAR):
        deltaV = np.zeros((NUMBERVAR))
        deltaV[i] = delta#changine only element corresponding to variable to delta and all other are zero
        nablaF[i] = (funcM(x + deltaV, Q, R) - funcM(x - deltaV, Q, R))/(2*delta)#using central difference method to find derivative
    
    return nablaF


#function to do bisection
def bisection(alpha1, alpha2, x, nablaF, Q, R, fmax, epsilon = 1e-5):
    Df1 = partialAlpha(alpha1, x, nablaF, Q, R)
    Df2 = partialAlpha(alpha2, x, nablaF, Q, R)
    noFunc = 6

    if(Df1 > 0):
        alpha1, alpha2= alpha2, alpha1
        Df1, Df2 = Df2, Df1
    
    x1, x2 = alpha1, alpha2
    z = (x1 + x2)/2
    Df1 = partialAlpha(z, x, nablaF, Q, R)

    while abs(Df1) >= epsilon:
        if Df1 < 0:
            x1 = z
        else:
            if Df1 > 0:
                x2 = z
            else:
                break
        z = (x1 + x2)/2
        if z > alpha2 or z < alpha1 or abs(x1 - x2) <= epsilon:
            break
        #print("bis", funcM(x + z*nablaF, Q, R))
        Df1 = partialAlpha(z, x, nablaF, Q, R)
        noFunc += 2

    return z, noFunc


#function to do bounding phase
#default bounds are fixed. if different values are needed can allocate when calling the function
#bounds are specified only to get random initial value between two numbers. result may be out of bound 
def boundingPhase(x, nablaF, Q, R, fmax,a = 250, b = -250, delta = 1):
    k = 0
    noFunc = 0
    xt = a
    x2 = b
    x1 = 0
     
    temp, fP, fN = funcM(x - (x1 - abs(delta)) * nablaF, Q, R), funcM(x - (x1) * nablaF, Q, R), funcM(x - (x1 + abs(delta)) * nablaF, Q, R)
    noFunc += 3
    
    while True:
        x1 = a + np.random.rand(1)*(b - a)
        
        temp, fP, fN = funcM(x - (x1 - abs(delta)) * nablaF, Q, NUMBERVAR), funcM(x - (x1) * nablaF, Q, NUMBERVAR), funcM(x - (x1 + abs(delta)) * nablaF, Q, NUMBERVAR)
        noFunc += 3
        
        if temp >= fP and fP >= fN:
            delta = abs(delta)
            break

        if temp <= fP and fP >= fN:
            delta = -1 * abs(delta)
            break
    
    while True:
        x2 = x1 + 2*k * delta
        fN = funcM(x - (x2) * nablaF, Q, R)
        noFunc += 1
        #print("bou", fN)
        if fN < fP:
            k = k + 1
            fP = fN
            xt = x1
            x1 = x2
        else:
            return np.array([xt, x1, x2]), noFunc

#function to do steepest descent
def steepestDescent(x0, Q, NUMBERVAR, bounds, R, fmax, MaxIter = 100000, epsilon1 = 1e-3, epsilon2 = 1e-3):
    #initialize
    k = 0
    #set intial values and calculate the gradient of the funcion
    x = x0
    nablaF = nablaFuncM(x0, Q, NUMBERVAR, R)
    totFunc = NUMBERVAR*2 #to calculate total number of function evaluation. in central difference two function are evaluated for each variables
    c = np.ones_like(nablaF) #create a vector like gradient of function which is used to shift the vectors when canculating angale between them
    inBounds = 1
    while  True:
        #first termination condition
        if (np.max(np.abs(nablaF))) < epsilon1 or k >= MaxIter:
            #print("break1",nablaF)
            break
        
        #gradient cliiping
        for i in range(NUMBERVAR):
            if nablaF[i] > 1e3:
                nablaF = nablaF/abs(nablaF[i])
            else:
                if nablaF[i] < 1e-4:
                    nablaF[i] = 0
        #performing search to find minimum alpha value first using boudning phase and then using bisection method
        #print("1")
        alphak, noFunc = boundingPhase(x, nablaF, Q, R, fmax)
        totFunc += noFunc
        #print("2")
        alphak, noFunc = bisection(alphak[0],alphak[2], x, nablaF, Q, R, fmax)
        totFunc += noFunc
        #updating the value
        xk = x - alphak * nablaF
        #print(alphak, nablaF, xk, x)
        #finding gradient of function at new point
        nablaFk = nablaFuncM(xk, Q, NUMBERVAR, R)
        totFunc += NUMBERVAR*2

        #second terminatin condition
        if abs(np.dot(nablaF + c, nablaFk + c)/(np.linalg.norm(nablaF + c)*np.linalg.norm(nablaFk + c))) > (1-epsilon2) or np.linalg.norm(xk - x)/(1e-15 + np.linalg.norm (x)) <= epsilon1:
            #print("break2",alphak, nablaFk)
            break

        #updating the number of steps
        k = k + 1
        
        for i in range(NUMBERVAR):
            if xk[i] < bounds[2*i] or xk[i] > bounds[2*i + 1]:
                inBounds = 0
#            else:
#                x[i] = xk[i]
        if inBounds == 0:
            break
        #updating the variables for next iteration
        x, nablaF = xk, nablaFk
        #break
    return x, totFunc

def brackPenalty(x0, Q, NUMBERVAR, bounds, R0, c,fmax, epsilon1 = 1e-5):
    Fcache = []
    xcache = []
    Ncache = []
    f = funcM(x0, Q, 0)
    noFunc = 1
    R = R0
    x = x0
    inBounds = 1
    while True:
        #print("1")
        xk, totFunc = steepestDescent(x, Q, NUMBERVAR, bounds, R, fmax)
        noFunc += totFunc
        ft = funcM(xk, Q, R)
        if (abs(f - ft)) <= epsilon1 or (funcM(xk, Q, 1, funcInc=0) < 1e7*epsilon1 and R > 2.5e6):
            #print(f, " ", ft)
            return xk, noFunc, Fcache
            break
        
        xcache.append(x)
        Fcache.append(f)
        f = ft
        x = xk
        R = c*R
        #print(x, " ", R)
        #return xk, noFunc
        #break

#main program.
#reading number of variables from file
#Dictionaries containing all the question name and number of variable to be choosen and initial value
questionList = {}
with open("question.txt") as file:
    for line in file:
        (key, value) = line.split(":")
        value = value.strip()
        value = value.split(",")
        value[1:] = [int(k) for k in value[1:]] 
        questionList[int(key)] = value

#print list of all question 
for qno in questionList.keys():
    print(qno, "-", questionList[qno])
#ask user for input 
print("Enter the question number to be solved ")
Q = int(input())
#Q = 1
checkR = 1
#get the data from the dictionaries
NUMBERVAR = questionList[Q][1]

#initial value is calculated here
x0 = np.zeros((NUMBERVAR))

for i in range(NUMBERVAR):
    x0[i] = np.average(questionList[Q][2+2*i:4 +2*i]) + (questionList[Q][2+2*i] - questionList[Q][3 +2*i])*(np.random.rand(1) - .5)
#print(x0)
#x0 = np.array([580, 1360, 5110, 180, 300, 220, 290, 400])
#option to restart
while True:
    #solve
    print("Solving ", questionList[Q][0], "using the Bracket operator penalty method")

    startT = time.time() #get the time at which steepest Descent method start
    x, funcEval, FIterat = brackPenalty(x0, Q, NUMBERVAR, questionList[Q][2:], 1e-3, 5, 1e5, epsilon1=1e-5) #calling the function
    endT = time.time() #get the time at which steepest descent method end

    #print all relevent data to prompt
    print("Initial values of x are", x0)
    print("x values are", x)
    print("Function Value at last iteration is ", funcM(x, Q, 0))
    print("Number of function evaluation is ", funcEval)
    print("Time taken (in seconds) by it is ", endT - startT)
    x0 = x
    print("Constraint equation are ",constraintViolate(x0, Q))
    fig = plt.figure()
    
    plt.plot(FIterat)
    plt.ylabel('Function Value', size = 16)
    plt.xlabel('No. of iteration', size = 16)
    plt.title(questionList[Q][0], size = 20)
    plt.grid('on')
    plt.xticks(fontsize = 14)
    plt.yticks(fontsize = 14)
    plt.show()
    fig.savefig('ConvergencePlot'+questionList[Q][0]+'.png')
    if(True):#funcM(x, Q, 1, funcInc=0) < 1e-3):
        break
