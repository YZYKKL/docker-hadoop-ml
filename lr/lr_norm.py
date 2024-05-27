import numpy as np
import pandas as pd
import time
import json

class LinearRegression:
    def __init__(self, dimension, bias=True):
        self.DIMENSION = dimension
        self.BIAS = bias
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

    def process_line(self, line):
        ''' Calculates x_t_x and x_t_y for each line '''
        y, features = self.extract_variables(line)
        if len(features) != self.DIMENSION:
            raise ValueError(f"Expected dimension {self.DIMENSION}, but got {len(features)}")
        if self.BIAS:
            features.append(1.0)
        x = np.array(features)
        self.x_t_x += np.outer(x, x)
        self.x_t_y += y * x
        self.counts += 1

    def cholesky_solution_linear_regression(self):
        '''Solves linear regression using Cholesky decomposition.'''
        L = np.linalg.cholesky(self.x_t_x)
        y = np.linalg.solve(L, self.x_t_y)
        betas = np.linalg.solve(L.T, y)
        return betas

    def run(self, file_path):
        with open(file_path, 'r') as f:
            for line in f:
                self.process_line(line)
        return self.cholesky_solution_linear_regression()

if __name__ == "__main__":
    start_time = time.time()

    DIMENSION = 8

    lr = LinearRegression(dimension=DIMENSION)
    betas = lr.run('data_lr_2.csv')

    end_time = time.time()
    print("Execution time: {:.2f} seconds".format(end_time - start_time))
    print("Betas:", json.dumps([e for e in betas]))
