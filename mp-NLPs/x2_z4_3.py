# %% Libraries
import numpy as np

#%% Partial derivatives of objective function
def partial_deriv(x):
    return [3*x[0]**2 + 4*x[0] - 5,
            2*x[1] - 3]

#%% Jacobian of partial derivatives function
def jac_partial_deriv(x):
    return np.array([[6*x[0] + 4, 0],
                     [0, 2]])

#%% Constraints on optimization variables (A x <= b + F theta):
A = np.array([[2, 1], [1/2, 1], [-1, 0], [0, -1]])
b = np.array([5/2, 3/2, 0, 0])
F = np.array([[1, 0], [0, 1], [0, 0], [0, 0]])

feas_space_constr = {'A':A, 'b':b, 'F':F}

#%% Space of parameters P_A theta <= P_b:
P_A = np.array([[-1, 0], [1, 0], [0, -1], [0, 1]])
P_b = np.array([0, 1, 0, 1])

par_space_constr = {'P_A':P_A, 'P_b':P_b}

#%% Bounds for optimization problems
bnds_theta = (0, 1)
