# %% Libraries
import numpy as np

#%% Partial derivatives of objective function
def partial_deriv(x):
    return [2*x[0]**3 + 1/2*x[0]**2 + 4*x[0] - 13/2,
            2*x[1]**3 + 1/2*x[1]**2 + 4*x[1] - 13/2]

#%% Jacobian of partial derivatives function
def jac_partial_deriv(x):
    return np.array([[6*x[0]**2 + x[0] + 4, 0],
                     [0, 6*x[1]**2 + x[1] + 4]])

#%% Constraints on optimization variables (A x <= b + F theta)
A = np.array([[1, 2]])
b = np.array([2])
F = np.array([[1]])

feas_space_constr = {'A':A, 'b':b, 'F':F}

#%% Space of parameters P_A theta <= P_b
P_A = np.array([[-1], [1]])
P_b = np.array([0, 2])

par_space_constr = {'P_A':P_A, 'P_b':P_b}

#%% Bounds for optimization problems
bnds_theta = (0, 2)
