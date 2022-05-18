from argparse import ArgumentParser
import os
import numpy as np
from gplearn import one_p_lambda_genetic
from sklearn.metrics import mean_squared_error

def XPlusLambda(X, Y, n, Lambda):
    ss = one_p_lambda_genetic.SymbolicRegressor()
    ss.fit(X,Y)
    print(str(ss))
    print(mean_squared_error(ss.predict(X), Y))
    return

if __name__=="__main__":
    parse = ArgumentParser(description = 'new')
    parse.add_argument('-f', '--file', required=True)
    args = parse.parse_args()
    if os.path.exists(args.file):
        X_Y = np.loadtxt(args.file, dtype=np.float, skiprows=1)
        print(X_Y.shape)

        X, Y = np.split(X_Y, (-1,), axis=1)
        XPlusLambda(X, Y, 1, .2)

    else:
        print("not exist")