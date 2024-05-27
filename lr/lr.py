from mrjob.job import MRJob
from mrjob.protocol import JSONProtocol, RawValueProtocol
from mrjob.step import MRStep
import numpy as np
import json
import time

######################## Helper Methods and Classes  ##########################

def cholesky_solution_linear_regression(x_t_x, x_t_y, regularization=1e-10):
    '''
    Finds parameters of regression through Cholesky decomposition,
    given sample covariance of explanatory variables and covariance 
    between explanatory variable and dependent variable.
    
    Parameters:
    -----------
    x_t_x    - numpy array of size 'm x m', represents sample covariance of explanatory variables
    x_t_y    - numpy array of size 'm x 1', represent covariance between explanatory and dependent variable
    
    Output:
    -------
    Theta   - list of size m, represents values of coefficients 
    '''
    x_t_x += np.eye(x_t_x.shape[0]) * regularization
    L = np.linalg.cholesky(x_t_x)
    z = np.linalg.solve(L, x_t_y)
    theta = np.linalg.solve(np.transpose(L), z)
    return theta

class DimensionMismatchError(Exception):
    def __init__(self, expected, observed):
        self.exp = expected
        self.obs = observed
        
    def __str__(self):
        err = "Expected number of dimensions: " + str(self.exp) + ", observed: " + str(self.obs)
        return err

############################## Map Reduce Job #################################

class LinearRegressionTS(MRJob):
    '''
    Calculates sample covariance matrix of explanatory variables (x_t_x) and 
    vector of covariances between dependent variable and explanatory variables (x_t_y)
    in a single map-reduce pass and then uses Cholesky decomposition to
    obtain values of regression parameters.
    
    Important: Since final computations are performed on a single reducer, 
    the assumption is that the dimensionality of the data is relatively small.
    
    Input File:
    -----------
    Extract relevant features from input line by changing extract_variables
    method. Current code assumes following input line format:
    input line = <dependent variable>, <feature_1>,...,<feature_n>
    
    Class Attributes:
    -----------------
    DIMENSION  - (int) number of explanatory variables
    BIAS       - (bool) if True, regression will include a bias term
    
    Output:
    -------
    JSON-encoded list of parameters
    '''
    
    INPUT_PROTOCOL = RawValueProtocol
    INTERNAL_PROTOCOL = JSONProtocol
    OUTPUT_PROTOCOL = RawValueProtocol
    
    DIMENSION = 8  # Number of explanatory variables (excluding bias term)
    BIAS = True    # Include bias term

    def __init__(self, *args, **kwargs):
        super(LinearRegressionTS, self).__init__(*args, **kwargs)
        n = self.DIMENSION + int(self.BIAS)
        self.x_t_x = np.zeros([n, n])
        self.x_t_y = np.zeros(n)
        self.counts = 0
        
    def extract_variables(self, line):
        ''' (str)--(float,[float,float,float...])
        Extracts set of relevant features.
        '''
        data = [float(e) for e in line.strip().split(",")]
        y, features = data[0], data[1:]
        return y, features

    def mapper_lr(self, _, line):
        ''' Calculates x_t_x and x_t_y for data processed by each mapper '''
        y, features = self.extract_variables(line)
        if len(features) != self.DIMENSION:
            raise DimensionMismatchError(self.DIMENSION, len(features))
        if self.BIAS:
            features.append(1.0)
        x = np.array(features)
        self.x_t_x += np.outer(x, x)
        self.x_t_y += y * x
        self.counts += 1
        
    def mapper_lr_final(self):
        ''' Transforms numpy arrays x_t_x and x_t_y into JSON-encodable list format and sends to reducer '''
        yield 1, ("x_t_x", [list(row) for row in self.x_t_x])
        yield 1, ("x_t_y", [xy for xy in self.x_t_y])
        yield 1, ("counts", self.counts)
        
    def reducer_lr(self, key, values):
        ''' Aggregates results produced by each mapper and obtains x_t_x and x_t_y for all data '''
        n = self.DIMENSION + int(self.BIAS)
        observations = 0
        x_t_x = np.zeros([n, n])
        x_t_y = np.zeros(n)
        for val in values:
            if val[0] == "x_t_x":
                x_t_x += np.array(val[1])
            elif val[0] == "x_t_y":
                x_t_y += np.array(val[1])
            elif val[0] == "counts":
                observations += val[1]
        betas = cholesky_solution_linear_regression(x_t_x, x_t_y)
        yield None, json.dumps([e for e in betas])
            
    def steps(self):
        ''' Defines map-reduce steps '''
        return [MRStep(mapper=self.mapper_lr,
                       mapper_final=self.mapper_lr_final,
                       reducer=self.reducer_lr)]

if __name__ == "__main__":
    start_time = time.time()
    LinearRegressionTS.run()
    end_time = time.time()
    print("Execution time: {:.2f} seconds".format(end_time - start_time))
