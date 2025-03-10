import numpy as np
import sympy as sp
import itertools
from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error


class SimplEq:

    def __init__(self):
        self.function = None
        self.expr = None

    def protected_power(self, x, y):
        
        return np.abs(x) ** y


    def create_function(self, symbol_1, symbol_list, parameters):

        assert len(symbol_list) == len(parameters) - 1

        if symbol_1 == 'x':
            
            def fun(X):
                res = np.zeros_like(X)
                for i in range(X.shape[1]):
                    if symbol_list[i] == 'x':
                        res[:,i] = self.protected_power(X[:,i], parameters[i])
                    elif symbol_list[i] == '+':
                        res[:,i] =  np.abs(parameters[i]) ** X[:,i]
                    else:
                        raise ValueError('Invalid symbol')
                return np.prod(res, axis=1) * parameters[-1]
            return fun

        elif symbol_1 == '+':

            def fun(X):
                res = np.zeros_like(X)
                for i in range(len(symbol_list)):
                    if symbol_list[i] == 'x':
                        res[:,i] = np.log(np.abs(X[:,i])+1e-6) * parameters[i]
                    elif symbol_list[i] == '+':
                        res[:,i] = X[:,i] * parameters[i]
                    else:
                        raise ValueError('Invalid symbol')
                return np.sum(res, axis=1) + parameters[-1]

            return fun
        else:
            raise ValueError('Invalid symbol')
        
    def fit(self, X, y):
        gen = np.random.default_rng(0)
        n_features = X.shape[1]
        results = {}
        for symbol_1 in ['+','x']:
            symbol_combinations = itertools.product(['+','x'], repeat=n_features)
            for symbol_list in symbol_combinations:
                def objective(parameters):
                    fun = self.create_function(symbol_1, symbol_list, parameters)
                    y_pred = np.clip(fun(X), -1e9, 1e9)
                    y_pred = np.nan_to_num(y_pred)
                    return mean_squared_error(y_pred, y)
                initial_guess = gen.uniform(-10, 10, n_features+1)
                result = minimize(objective, initial_guess, method='L-BFGS-B')
                results[(symbol_1, tuple(symbol_list))] = (result.fun, result.x)

        best = min(results, key=lambda x: results[x][0])

        # Get the symbolic representation of the function
        symbol_1, symbol_list = best
        parameters = results[best][1]
        
        xs = []
        for i in range(n_features):
            xs.append(sp.symbols('x' + str(i+1)))

        expr = sp.Number(0)

        # Go through the parameters and replace with integers if they are close enough
        for i in range(len(parameters)):
            if np.abs(parameters[i] - np.round(parameters[i])) < 1e-4:
                parameters[i] = np.round(parameters[i])
        
        
        if symbol_1 == 'x':
            expr = sp.Number(1)

            for x, symbol, parameter in zip(xs, symbol_list, parameters[:-1]):
                if symbol == '+':
                    expr *= sp.Abs(parameter) ** x
                else:
                    expr *= x ** parameter
            expr *= parameters[-1]
        
        else:
            for x, symbol, parameter in zip(xs, symbol_list, parameters[:-1]):
                if symbol == '+':
                    expr += sp.log(x) * parameter
                else:
                    expr += x * parameter

            expr += parameters[-1]

        self.function = self.create_function(symbol_1, symbol_list, parameters)
        self.expr = expr

    def predict(self,X):
        return self.function(X)
    
    def get_expr(self):
        return self.expr
