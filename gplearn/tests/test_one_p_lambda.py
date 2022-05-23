from argparse import ArgumentParser
import os
import numpy as np
import time
from gplearn import one_p_lambda_genetic
from gplearn import genetic
from sklearn.metrics import mean_squared_error

def Test(X, Y, n, Lambda, regressor, file, reg_name):
    regressor.population_size = Lambda
    time_pass = time.time()

    regressor.fit(X,Y)
    time_pass = time.time() - time_pass
    print('reslult_yxgao |',file, '|', reg_name, '|', str(regressor), '|' , mean_squared_error(regressor.predict(X), Y), '|', time_pass)
    return


if __name__=="__main__":
    parse = ArgumentParser(description = 'new')
    parse.add_argument('-f', '--file', required=True)
    args = parse.parse_args()
    if os.path.exists(args.file):
        X_Y = np.loadtxt(args.file, dtype=np.float, skiprows=1)
        print(X_Y.shape)

        X, Y = np.split(X_Y, (-1,), axis=1)
        Test(X, Y, 1, 1000, one_p_lambda_genetic.SymbolicRegressor(), args.file, 'XPLamb')
        Test(X, Y, 1, 1000, genetic.SymbolicRegressor(), args.file, 'gplearn')

    else:
        print("not exist")