'''
    TEST SCRIPT FOR MULTIVARIATE LINEAR REGRESSION
    AUTHOR Eric Eaton, Vishnu Purushothaman Sreenivasan
'''

'''
Numpy is a standard library in python that lets you do matrix and vector operations like Matlab in python.
Check out documentation here: http://wiki.scipy.org/Tentative_NumPy_Tutorial
If you are a Matlab user this page is super useful: http://wiki.scipy.org/NumPy_for_Matlab_Users 
'''
import numpy as np
from numpy.linalg import *

# our linear regression class
from linreg import LinearRegression
from test_linreg_univariate import visualizeObjective, plotData1D, plotRegLine1D # @@ add by student

if __name__ == "__main__":
    '''
        Main function to test multivariate linear regression
    '''
    
    # load the data
    filePath = "/home/anht/AI_Class/Assignment/CIS419/hw1/multivariateData.dat"
    file = open(filePath,'r')
    allData = np.loadtxt(file, delimiter=',')

    X = np.matrix(allData[:,:-1])
    y = np.matrix((allData[:,-1])).T

    n,d = X.shape
    

    # Standardize
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    X = (X - mean) / std
    
    # Add a row of ones for the bias term
    X = np.c_[np.ones((n,1)), X]
    
    # initialize the model
    init_theta = np.matrix(np.random.randn((d+1))).T
    n_iter = 2000
    alpha = 0.01

    # Instantiate objects
    lr_model = LinearRegression(init_theta = init_theta, alpha = alpha, n_iter = n_iter)
    lr_model.fit(X,y)
    # plotRegLine1D(lr_model, X,y)
    
    # Compute the closed form solution in one line of code
    thetaClosedForm = (X.getT()*X).getI()*X.getT()*y
    print ("thetaClosedForm: ", thetaClosedForm)
    
    
    # # Visualize the objective function convex shape
    # theta1_vals = np.linspace(-10, 10, 100);
    # theta2_vals = np.linspace(-10, 10, 100);
    # visualizeObjective(lr_model,theta1_vals, theta2_vals, X, y)

