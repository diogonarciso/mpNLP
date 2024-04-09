# Python libraries
from math import factorial as factorial
import numpy as np
from scipy.optimize import root
from scipy.optimize import linprog
import pyomo.environ as pyo
# from time import process_time_ns
# from time import time
from timeit import default_timer as timer

#%% Definition of left hand side of KKT conditions (right hand side of all equations equal 0). These are written in general form,
# including not only the partial derivatives of the objective function but also the relevant terms for all inequality constraints
# defined in any mp-NLP problem. To solve the KKT conditions for any active set of interest and any z or theta, it suffices to
# specify "act_set" and "b" (b = b + F theta).
def KKT_cond(x, mpNLP, dim, act_set, b):

    # Vector of KKT conditions - initialization
    KKT_cond = np.zeros(dim['N_x']+dim['N_z'])

    # First "dim['N_x']" positions: derivatives of Lagrange function with respect to all optimization variables
    for i in range(dim['N_x']):
        KKT_cond[i] = mpNLP.partial_deriv(x)[i] + sum(act_set[j]*mpNLP.A[j,i]*x[dim['N_x']+j] for j in range(dim['N_z']))

    # Last "dim['N_z']" positions: active constraints
    for i in range(dim['N_z']):
        KKT_cond[dim['N_x']+i] = act_set[i]*(sum(mpNLP.A[i,j]*x[j] for j in range(dim['N_x'])) - b[i])

    return KKT_cond

#%% Definition of matrix of partial derivatives of KKt conditions. Defined consistently with the principles in function "KKT_cond".
def KKT_cond_derivs(x, mpNLP, dim, act_set, b):

    # Initialization of matrix of partial derivatives of KKT conditions
    KKT_cond_derivs = np.zeros((dim['N_x']+dim['N_z'], dim['N_x']+dim['N_z']))

    # Matrix "A" is used to set the lower left and upper right "corners" of the matrix of partial derivatives. A correction
    # must be made to take into account the active set, and set accordingly all relevant entries of "A" to 0. This is done
    # via the auxiliary matrix "A_act_set".
    A_act_set = np.zeros((dim['N_z'],dim['N_x']))
    for i in range(dim['N_z']):
        A_act_set[i] = mpNLP.A[i,:] * act_set[i]

    # Setting all entries of matrix of partial derivatives
    KKT_cond_derivs[0:dim['N_x'], 0:dim['N_x']]                     = mpNLP.jac_partial_deriv(x)[0:dim['N_x'], 0:dim['N_x']]    # upper left corder - second order partial derivatives
    KKT_cond_derivs[0:dim['N_x'], dim['N_x']:dim['N_x']+dim['N_z']] = A_act_set.T
    KKT_cond_derivs[dim['N_x']:dim['N_x']+dim['N_z'], 0:dim['N_x']] = A_act_set

    return KKT_cond_derivs

#%% Bounds for linear programming - when using the parametric space constraints
def lin_prog_bnds(dim, bnds_theta):

    # Bounds per variable and initialization
    bnds_l              = (0, 1e10)
    bnds_min_z_i        = []
    bnds_par_edge_test  = []

    for i in range(1+dim['N_z']):
        bnds_par_edge_test.append(bnds_l)

    # Bounds for all N_theta variables
    for i in range(dim['N_theta']):
        bnds_min_z_i.append(bnds_theta)
        bnds_par_edge_test.append(bnds_theta)

    return bnds_min_z_i, bnds_par_edge_test

# %% Calculation of compact solution for a mp-NLP (Algorithm 1), featuring a convex objective
# function and a set of affine inequality constraints - parameters on right hand side.
def compact_solution_calc(mpNLP, dim, par_space_bnds, general_par):

    # Start counting process time
    # start_time = process_time_ns()
    # start_time = time()
    start_time = timer()

    # Edges initialization
    edges_x = np.zeros((dim['N_x'], dim['N_z']))

    # Reference points initialization (used in optimizer and parametric edges calculation)
    ref_points_x    = np.zeros((dim['N_x'], dim['N_z']))
    Lag_mult        = np.zeros((dim['N_z'], dim['N_z']))
    ref_points_z    = np.zeros((dim['N_z'], dim['N_z']))

    ### Step 1 ### Optimizer vertex, x* (global optimizer - from KKT conditions, no constraint active)
    vertex_x = root(mpNLP.partial_deriv, np.zeros(dim['N_x']), jac=mpNLP.jac_partial_deriv, method='hybr').x

    ### Step 2 ### Parametric vertex, z*
    vertex_z = np.dot(mpNLP.feas_space_constr['A'], vertex_x) - mpNLP.feas_space_constr['b']

    ### Step 3 ### For all inequality constraints obtain reference points along the corresponding optimizer and
    # parametric edges, and estimate the gradients of optimizer edges
    for i in range(dim['N_z']):
        
        # Single constraint active set for current iteration
        act_set     = np.zeros(dim['N_z']).astype(int)
        act_set[i]  = 1

        ### Step 3A.1 ### Minimum of the ith coordinate of z within the defined (theta) parameter space
        z_min_i = lin_prog_min_z_i(mpNLP.feas_space_constr['F'][i,:], mpNLP.par_space_constr['P_A'], mpNLP.par_space_constr['P_b'], par_space_bnds)

        if (z_min_i <= vertex_z[i]):
            b_i = mpNLP.b[i] + z_min_i - general_par['scope_contract_coeff'][i]
        else:
            b_i = mpNLP.b[i] + vertex_z[i] - general_par['single_contract_coeff']

        b       = vertex_z + 1  # slighly relaxed beyond the coordinates of the parametric vertex (any setting would be ok, since all non active constraints are multipled by 0 in the KKT conditions below)
        b[i]    = b_i

        ### Step 3A.2 ### Calculate the coordinates of a reference optimizer along current optimizer edge

        # Calculate optimizer at defined z
        sol = root(KKT_cond, np.zeros(dim['N_x']+dim['N_z']), jac=KKT_cond_derivs, method='hybr', args=(mpNLP, dim, act_set, b)).x
        ref_points_x[:,i]   = sol[0:dim['N_x']]
        Lag_mult[:,i]       = sol[dim['N_x']:dim['N_x']+dim['N_z']]

        # Calculate the corresponding vector of parameters
        ref_points_z[:,i] = np.dot(mpNLP.feas_space_constr['A'], ref_points_x[:,i]) - mpNLP.feas_space_constr['b']

        ### Step 3A.3 & 3B ### Estimate sensitivities of current optimizer edge
        edges_x[:,i] = ref_points_x[:,i] - vertex_x

    ### Step 5 ###
    edges_z = np.dot(mpNLP.feas_space_constr['A'], edges_x)

    # Total process time
    # stop_time = process_time_ns()
    # stop_time = time()
    stop_time = timer()
    total_process_time = stop_time - start_time

    # Note: Step 4 is the definition of the identity matrix; it is directly specified in edges definition ('z_inact')

    # Summary of information: vertices, edges and reference points used in their calculation
    vertices    = {'x':vertex_x, 'z':vertex_z}
    edges       = {'x':edges_x, 'z_inact':np.identity(dim['N_z']), 'z_act':edges_z, 'z_diff':edges_z-np.identity(dim['N_z'])}
    ref_points  = {'x':ref_points_x , 'z':ref_points_z}

    return vertices, edges, ref_points, Lag_mult, total_process_time

#%% Auxiliary LP used in compact solution to determine the minimum of any of the coordinates of theta in the defined
# parameter space.
def lin_prog_min_z_i(cost, A_ub, b_ub, bounds):

    fun = linprog(cost, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs').fun

    return fun

#%% This solution is equivalent in terms of accuracy to the compact solution. However, in this case, only those solution
# fragments defining full-dimensional critical regions in the parameter space are calculated and returned as a result.
def basic_explicit_solution(mpNLP, dim, bnds_par_edge_test, vertices, edges, ref_points, ub, solver_str):

    # Start counting process time
    # start_time = process_time_ns()
    # start_time = time()
    start_time = timer()

    ### Steps 1-4 ### Obtain the admissible statuses for all active sets via a set of auxiliary MILP calculations
    infeas_bool, current_as, free_pos, fixed_pos, N_free_pos, N_fixed_pos, par_edge_active_check = param_edges_redund_test(mpNLP.feas_space_constr, mpNLP.par_space_constr, dim, vertices, edges, ub, solver_str)
    # print(current_as)
    # print(fixed_pos)

    # If no parametric edges are associated with the parameter space, it defines a region of infeasibility
    # In this case, the solution of the mp-NLP is empty (no fragments)
    if (infeas_bool):
        CR_index        = 0
        act_sets        = np.zeros((0,dim['N_z'])).astype(int)
        partitions      = np.zeros((0,dim['N_z']+1,dim['N_z']+1))
        ref_low_dim_as_container    = np.zeros((0,dim['N_z'])).astype(int) ### PARA APAGAR

    # At least one critical region is detected in the parameter space
    else:
        # Number of active constraints in reference/current active set in the fixed positions
        N_active_constraints_fixed_pos = 0 
        for i in range(N_fixed_pos):
            if (current_as[fixed_pos[i]] == 1):
                N_active_constraints_fixed_pos += 1

        # Initialization of vectors and matrices for LP: check if a given active set/critical region is included in the
        # parameter space
        cost    = np.ones(1+dim['N_z']+dim['N_theta'])
        A_eq    = np.zeros((1+dim['N_z'],1+dim['N_z']+dim['N_theta']))
        b_eq    = np.zeros(1+dim['N_z'])
        A_ineq  = np.zeros((dim['N_par_bnds'],1+dim['N_z']+dim['N_theta']))

        # Setting the constant entries of the vectors/matrices above
        A_eq[0:dim['N_z'],1+dim['N_z']:1+dim['N_z']+dim['N_theta']] = mpNLP.feas_space_constr['F']
        b_eq[dim['N_z']]                                            = -1
        A_ineq[:,1+dim['N_z']:1+dim['N_z']+dim['N_theta']]          = mpNLP.par_space_constr['P_A']

        # Initialization of containers: active sets, critical region fragments and reference low dimensional active sets
        act_sets                    = np.zeros((2**N_free_pos,dim['N_z'])).astype(int)
        partitions                  = np.zeros((2**N_free_pos,dim['N_z']+1,dim['N_z']+1))
        ref_low_dim_as_container    = np.zeros((0,dim['N_z'])).astype(int)

        # Initialization of critical region fragment
        CR_frag = np.zeros((dim['N_z']+1,dim['N_z']+1))

        # Setting constant entries in CR_frag
        CR_frag[0:dim['N_z'],0] = vertices['z']
        CR_frag[dim['N_z'],0]   = 1

        # # Initialization of indexes
        CR_index    = 0     # critical region fragments
        k           = 0     # number of active constraints - within the set of free positions

        # Multiple active sets detected in the parameter space
        if (N_free_pos > 0):

            ### Steps 5, 6 ### Calculation of all critical region fragments
            while (k <= dim['N_x'] - N_active_constraints_fixed_pos and k <= N_free_pos):

                # Active sets initialization (within free positions), for given number of active constraints (k)
                N_as_current_k, candidate_active_set, pivot, pos = active_set_init(N_free_pos, k)

                # Test all active sets for current k
                for i in range (N_as_current_k):

                    # Current active set: fixed positions remain constant; free positions set from candidate_active_set
                    for j in range(N_free_pos):
                        current_as[free_pos[j]] = candidate_active_set[j]

                    # Check if current active set can be derived from any active set associated with a low dimensional critical region
                    global_active_set_match = active_set_match_calc(dim, ref_low_dim_as_container, current_as)

                    # Only those active sets not "related" with a previously identified low dimensional active set/critical region
                    # undergo further testing
                    if not(global_active_set_match):

                        ### Step 7, 8 ### Checks if current active set defines a full-dimensional critical region included in the parameter space
                        CR_index, act_sets, partitions, ref_low_dim_as_container = cr_dim_relevance_check(mpNLP.feas_space_constr, dim, edges, ref_points, current_as, CR_frag, cost, A_ineq, mpNLP.par_space_constr, A_eq, b_eq, bnds_par_edge_test, CR_index, act_sets, partitions, ref_low_dim_as_container)

                    # Move to the next active set (including k active constraints in the set of free positions)
                    candidate_active_set, pivot, pos = next_as_calc(N_free_pos, candidate_active_set, pivot, pos)

                ### Step 9 ### Increment the number of active constraints in the set of free positions
                k += 1

        # A single active set detected in the parameter space
        else: # if (N_free_pos == 0):

            # Checks if current active set defines a full-dimensional critical region included in the parameter space
            CR_index, act_sets, partitions, ref_low_dim_as_container = cr_dim_relevance_check(mpNLP.feas_space_constr, dim, edges, ref_points, current_as, CR_frag, cost, A_ineq, mpNLP.par_space_constr, A_eq, b_eq, bnds_par_edge_test, CR_index, act_sets, partitions, ref_low_dim_as_container)

    ### Step 10 ### General optimizer functions
    opt_fun = np.zeros((dim['N_x'], dim['N_z']+1))
    opt_fun[:,0] = vertices['x']
    opt_fun[:,1:dim['N_z']+1] = ref_points['x']

    # Total process time
    # stop_time = process_time_ns()
    # stop_time = time()
    stop_time = timer()
    total_process_time = stop_time - start_time

    # Key outputs (diagnostics)
    # print(infeas_bool)
    # print(act_sets[0:CR_index,:])
    # print(ref_low_dim_as_container)

    return act_sets[0:CR_index,:], partitions[0:CR_index,:,:], opt_fun, par_edge_active_check, CR_index, total_process_time

#%% For any given active set, this function checks: (i) if the corresponding critical region is full dimensional, and (ii) it is included
# in the space of parameters. If so, it is added to the matrix of partitions. Low dimensional critical regions are also detected and saved.
def cr_dim_relevance_check(feas_space_constr, dim, edges, ref_points, current_as, CR_frag, cost, A_ineq, par_space_constr, A_eq, b_eq, bnds_par_edge_test, CR_index, as_container, CR_container, ref_low_dim_as_container):

    # Critical region dimensionality check: this may be checked via the parametric edges; since all inactive
    # edges are all orthogonal, it suffices to check those associated with the active constraints. In the case
    # of mp-NLP, it's best to compare the normal directions of all constraints of the feasible space

    A_active_constraints = feas_space_constr['A'][current_as.astype(bool),:]
    N_active_constraints = A_active_constraints.shape[0]
    if (N_active_constraints != 0):
        dim_active_constraints = np.linalg.matrix_rank(A_active_constraints)
    else:
        dim_active_constraints = 0

    # Only those active sets/critical regions defining full dimensional critical regions undergo further testing
    if (dim_active_constraints == N_active_constraints):

        # Define critical region fragment from the relevant inactive edges or reference points
        # The first column corresponds to the parametric vertex (initialized before the current while loop)
        for j in range(dim['N_z']):
            if (current_as[j] == 0):
                CR_frag[0:dim['N_z'],j+1]   = edges['z_inact'][:,j]
                CR_frag[dim['N_z'],j+1]     = 0
            else:
                CR_frag[0:dim['N_z'],j+1]   = ref_points['z'][:,j]
                CR_frag[dim['N_z'],j+1]     = 1

        # Setting A_eq from CR_frag and checking if current active set/critical region is detected in the
        # parameter space, via LP: if feasible/optimal solution is found, status = 0 is returned
        A_eq[:,0:dim['N_z']+1] = -CR_frag
        status = lin_prog_cr_check(cost, A_ineq, par_space_constr['P_b'], A_eq, b_eq, bnds_par_edge_test)

        # If current active set/critical region is detected in the parameter space, both are added to the
        # corresponding containerers
        if (status == 0): ## check...
            as_container[CR_index,:]    = current_as
            CR_container[CR_index,:,:]  = CR_frag

            # Increment index
            CR_index += 1

    # If current active set corresponds to a lower dimencional critical region - facet - this active set is
    # used to detect any active sets including the same set of active constraints: these are also low dimensional
    elif (dim_active_constraints == N_active_constraints-1):
        ref_low_dim_as_container = np.append(ref_low_dim_as_container, current_as.reshape(-1, dim['N_z']), axis=0)

    return CR_index, as_container, CR_container, ref_low_dim_as_container

#%% Auxiliary LP used in compact solution to determine the minimum of any of the coordinates of theta in the defined
# parameter space.
def lin_prog_cr_check(cost, A_ub, b_ub, A_eq, b_eq, bounds):

    status = linprog(cost, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs').status

    return status

#%% Tests all parametric edges, to check if they are associated with any critical region in the parameter space. From here, important
# conclusions are made with respect to the permitted constraint statuses (inactive and active, always inactive, always active)
def param_edges_redund_test(feas_space_constr, par_space_constr, dim, vertices, edges, ub, solver_str):

    ### Step 1 ### initialization of reference active set; represents the permissable constraint statuses of the defined mp-NLP
    # (-1 denotes that both statuses inactive:0 and active:1 are possible) 
    par_edge_active_check   = np.zeros(dim['N_z']).astype(bool)
    par_edge_inactive_check = np.zeros(dim['N_z']).astype(bool)

    ### Step 2 ### Checking always inactive constraints
    for i in range(dim['N_z']):
        MILP_status = lin_prog_par_edge(feas_space_constr, par_space_constr, dim, vertices, edges, True, i, ub, solver_str)
        # print(MILP_status)
        if (MILP_status == 'infeasible'):
            par_edge_active_check[i] = True

    ### Step 3 ### Checking always active constraints
    for i in range(dim['N_z']):
        # if (ref_as[i] == -1):
        MILP_status = lin_prog_par_edge(feas_space_constr, par_space_constr, dim, vertices, edges, False, i, ub, solver_str)
        # print(MILP_status)
        if (MILP_status == 'infeasible'):
            par_edge_inactive_check[i] = True

    ### Step 4 ### Infeasibility check and reference active set
    if ((sum(par_edge_active_check) == dim['N_z']) and (sum(par_edge_inactive_check) == dim['N_z'])):
        infeas_bool = True
        ref_as, free_pos, fixed_pos, N_free_pos, N_fixed_pos = None, None, None, None, None

    else:
        infeas_bool = False

        ref_as  = -np.ones(dim['N_z']).astype(int)

        # Calculation of reference active set
        for i in range(dim['N_z']):
            if (par_edge_active_check[i]):
                ref_as[i] = 0
            elif (par_edge_inactive_check[i]):
                ref_as[i] = 1

        # Free and fixed positions of reference active set from parametric edges redundancy tests
        free_pos    = np.array(range(dim['N_z']))[(ref_as == -1)]
        fixed_pos   = np.array(range(dim['N_z']))[(ref_as != -1)]
        N_free_pos  = free_pos.shape[0]
        N_fixed_pos = fixed_pos.shape[0]

    return infeas_bool, ref_as, free_pos, fixed_pos, N_free_pos, N_fixed_pos, par_edge_active_check

#%% For any given parametric edge (via "active_edge_bool" and "index_col"), this function determines if it is
# associated with at least one vector of parameters in the parameter space, and thus relevant for the general
# solution of the mp-NLP problem (only if MILP is not infeasible).
def lin_prog_par_edge(feas_space_constr, par_space_constr, dim, vertices, edges, active_edge_bool, index_col, ub, solver_str):

    # Construct vectors and matrices for MILP definition
    c_theta, A_eq_l, A_eq_theta, b_eq, A_ineq_y, A_ineq_l, b_ineq, A_bnds_theta, b_bnds, MILP_dim = lin_prog_par_edge_mat_vec(feas_space_constr, par_space_constr, dim, vertices, edges, active_edge_bool, index_col, ub)

    # Define and solve MILP via Pyomo using vectors and matrices above - returns optimization status
    MILP_status = lin_prog_par_edge_solve(c_theta, A_eq_l, A_eq_theta, b_eq, A_ineq_y, A_ineq_l, b_ineq, A_bnds_theta, b_bnds, MILP_dim, solver_str)

    return MILP_status

#%% This function creates the vectors and matrices necessary for the definition of the MILP enabling
# to determine if a given parametric edge (via "active_edge_bool" and "index_col") are associated to
# at least one point in the defined parameter space.
def lin_prog_par_edge_mat_vec(feas_space_constr, par_space_constr, dim, vertices, edges, active_edge_bool, index_col, ub):

    # Number of variables and inequality constraints
    N_var_y     = dim['N_z']-1
    N_var_l     = 2*dim['N_z']-1
    N_var_theta = dim['N_theta']
    N_eq        = dim['N_z']
    N_ineq      = 3*(dim['N_z']-1)
    N_bnds      = dim['N_par_bnds']

    # Initialization
    c_theta         = np.ones(N_var_theta)
    A_eq_l          = np.zeros((dim['N_z'], N_var_l))
    A_ineq_y        = np.zeros((N_ineq, N_var_y))
    A_ineq_l        = np.zeros((N_ineq, N_var_l))
    b_ineq          = np.zeros(N_ineq)

    # Selected columns index
    index_bool              = np.ones(dim['N_z']).astype(bool)
    index_bool[index_col]   = False

    ### A_eq_x and b_eq ###

    # Sensitivities to multiplier of current edge (via "active_edge_bool" and "index_col")
    if (active_edge_bool):
        A_eq_l[:,0:1] = -edges['z_act'][:,index_col].reshape(dim['N_z'],-1)
    else:
        A_eq_l[:,0:1] = -edges['z_inact'][:,index_col].reshape(dim['N_z'],-1)

    # Sensitivities to remaining - inactive - multipliers (l)
    A_eq_l[:,1:dim['N_z']] = -edges['z_inact'][:,index_bool]

    # Sensitivities to remaining - active - multipliers (alpha = y*l)
    A_eq_l[:,dim['N_z']:2*(dim['N_z']-1)+1] = -edges['z_diff'][:,index_bool]

    # Sensitivities to theta
    A_eq_theta = feas_space_constr['F']

    b_eq = vertices['z']

    ### A_ineq (y,l) and b_ineq ###
    for i in range(dim['N_z']-1):
        A_ineq_y[i,i] = -ub
        A_ineq_l[i,dim['N_z']+i] = 1
        A_ineq_l[dim['N_z']-1+i,1+i] = -1
        A_ineq_l[dim['N_z']-1+i,dim['N_z']+i] = 1
        A_ineq_y[2*(dim['N_z']-1)+i,+i] = ub
        A_ineq_l[2*(dim['N_z']-1)+i,1+i] = 1
        A_ineq_l[2*(dim['N_z']-1)+i,dim['N_z']+i] = -1
        b_ineq[2*(dim['N_z']-1)+i] = ub

    ### A_bnds (theta) and b_bnds - bounds of parameter space ###
    A_bnds_theta = par_space_constr['P_A']
    b_bnds = par_space_constr['P_b']

    ### MILP dimensions ###
    MILP_dim = {'N_var_y':N_var_y, 'N_var_l':N_var_l, 'N_var_theta':N_var_theta,
                'N_eq':N_eq, 'N_ineq':N_ineq, 'N_bnds':N_bnds}

    return c_theta, A_eq_l, A_eq_theta, b_eq, A_ineq_y, A_ineq_l, b_ineq, A_bnds_theta, b_bnds, MILP_dim

#%% Taking the vector and matrix outputs from "lin_prog_par_edge_mat_vec", this function defines the
# corresponding MILP via Pyomo. This requires the coding of a number of elements of the optimization
# problem consistently with Pyomo's syntax. This includes also a number of dictionaries
def lin_prog_par_edge_solve(c_theta, A_eq_l, A_eq_theta, b_eq, A_ineq_y, A_ineq_l, b_ineq, A_bnds_theta, b_bnds, MILP_dim, solver_str):

    ### Model declaration ###
    model = pyo.ConcreteModel()

    ### Sets ###
    model.I_eq      = pyo.Set(initialize=range(MILP_dim['N_eq']))           # Equality constraints
    model.I_ineq    = pyo.Set(initialize=range(MILP_dim['N_ineq']))         # Inequality constraints
    model.I_bnds    = pyo.Set(initialize=range(MILP_dim['N_bnds']))         # Parameter space bounds
    model.J_y       = pyo.Set(initialize=range(MILP_dim['N_var_y']))        # Integer variables
    model.J_l       = pyo.Set(initialize=range(MILP_dim['N_var_l']))        # Real variables - multipliers
    model.J_theta   = pyo.Set(initialize=range(MILP_dim['N_var_theta']))    # Real variables - theta

    ### Dicts - conversion from matrix form (requirement in Pyomo) ###
    c_theta_dict        = dict(zip(model.J_theta, c_theta))
    A_eq_l_dict         = dict_constr(model.I_eq, model.J_l, A_eq_l)
    A_eq_theta_dict     = dict_constr(model.I_eq, model.J_theta, A_eq_theta)
    b_eq_dict           = dict(zip(model.I_eq, b_eq))
    A_ineq_y_dict       = dict_constr(model.I_ineq, model.J_y, A_ineq_y)
    A_ineq_l_dict       = dict_constr(model.I_ineq, model.J_l, A_ineq_l)
    b_ineq_dict         = dict(zip(model.I_ineq, b_ineq))
    A_bnds_theta_dict   = dict_constr(model.I_bnds, model.J_theta, A_bnds_theta)
    b_bnds_dict         = dict(zip(model.I_bnds, b_bnds))

    ### Decision variables ###
    model.y         = pyo.Var(model.J_y, within=pyo.Binary)
    model.l         = pyo.Var(model.J_l, within=pyo.NonNegativeReals)
    model.theta     = pyo.Var(model.J_theta, within=pyo.Reals)

    ### Objective function ###
    model.c_theta = pyo.Param(model.J_theta, initialize=c_theta_dict)

    def obj_func(model):
        return sum(model.theta[j] * model.c_theta[j] for j in model.J_theta)

    model.obj = pyo.Objective(rule=obj_func, sense=pyo.minimize)

    ### Equalities ###
    model.A_eq_l        = pyo.Param(model.I_eq, model.J_l, initialize=A_eq_l_dict)
    model.A_eq_theta    = pyo.Param(model.I_eq, model.J_theta, initialize=A_eq_theta_dict)
    model.b_eq          = pyo.Param(model.I_eq, initialize=b_eq_dict)

    def eq_constr_def(model, i):
        return sum(model.A_eq_l[i, j] * model.l[j] for j in model.J_l) + sum(model.A_eq_theta[i, j] * model.theta[j] for j in model.J_theta) == model.b_eq[i]

    model.eq_constr = pyo.Constraint(model.I_eq, rule=eq_constr_def)

    ### Inequalities ###
    model.A_ineq_y  = pyo.Param(model.I_ineq, model.J_y, initialize=A_ineq_y_dict)
    model.A_ineq_l  = pyo.Param(model.I_ineq, model.J_l, initialize=A_ineq_l_dict)
    model.b_ineq    = pyo.Param(model.I_ineq, initialize=b_ineq_dict)

    def ineq_constr_def(model, i):
        return sum(model.A_ineq_y[i, j] * model.y[j] for j in model.J_y) + sum(model.A_ineq_l[i, j] * model.l[j] for j in model.J_l) <= model.b_ineq[i]

    model.ineq_constr = pyo.Constraint(model.I_ineq, rule=ineq_constr_def)

    ### Parameter space bounds ###
    model.A_bnds_theta  = pyo.Param(model.I_bnds, model.J_theta, initialize=A_bnds_theta_dict)
    model.b_bnds        = pyo.Param(model.I_bnds, initialize=b_bnds_dict)

    def par_space_bnds_def(model, i):
        return sum(model.A_bnds_theta[i, j] * model.theta[j] for j in model.J_theta) <= model.b_bnds[i]

    model.par_space_bnds = pyo.Constraint(model.I_bnds, rule=par_space_bnds_def)

    ### Set solver and solve MILP ###
    opt = pyo.SolverFactory(solver_str)
    # opt.options['mipgap'] = 1e-5
    sol = opt.solve(model)#, tee=False)
    # model.pprint()
    # print(model.y[0].values)

    return sol.solver.termination_condition

#%% Constructs a dict from a numpy array, and the corresponding row and column indices (sets in Pyomo).
def dict_constr(row_set, col_set, mat_input):

    dict_output = {}
    for k, i in enumerate(row_set):
        for l, j in enumerate(col_set):
            dict_output[i, j] = mat_input[k, l]

    return dict_output

#%% Taking the validated active sets from the basic explicit solution, a set of partitions is created to improve solution accuracy.
# First all edges are refined (additional points added for more accurate representation), then using these points and an initial set
# of partitions, these are broken down in smaller partitions as necessary to satisfy an error criteria on all of them (optimizer accuracy)
def refined_explicit_solution(mpNLP, dim, vertices, edges, ref_points, act_sets, N_act_sets, par_edge_active_check, bnds_par_edge_test, general_par):

    # Start counting process time
    # start_time = process_time_ns()
    # start_time = time()
    start_time = timer()

    # Edges refinement: additional set of points added to all edges to improve solution accuracy (based on error testing in the middle of segments)
    opt_edges, par_edges = edges_refinment(mpNLP, dim, vertices, ref_points, par_edge_active_check, general_par['tolerance_edges'])

    # Set of initial partitions for all active sets (relevant points expressed as positions in all edges)
    ref_positions, act_constr_index = initial_partitions(dim, act_sets, N_act_sets, par_edges)

    # All initial partitions are checked and broken down in smaller partitions to improve accuracy of their optimizer functions
    partitions_dict, opt_funcs_dict = validate_break_all_active_sets(mpNLP, dim, edges, opt_edges, par_edges, act_sets, ref_positions, act_constr_index, bnds_par_edge_test, general_par['tolerance_partitions'])

    # The number of partitions per active set and per initial partition is variable, and are initially saved as dicts. Here they are converted to standard numpy arrays
    act_sets_array, partitions_array, opt_funcs_array = list_results_as_arrays(dim, act_sets, partitions_dict, opt_funcs_dict)

    # Total process time
    # stop_time = process_time_ns()
    # stop_time = time()
    stop_time = timer()
    total_process_time = stop_time - start_time

    return act_sets_array, partitions_array, opt_funcs_array, opt_edges, par_edges, total_process_time

#%% Application of function "single_edge_refinement" to all optimizer edges. Includes providing the initial set of points per deges
# and saving results in dicts.
def edges_refinment(mpNLP, dim, vertices, ref_points, par_edge_active_check, tolerance):

    # NOTE: from the basic explicit solution, not all optimizer edges necessarily require refinment; this function may be improved
    # to dismiss such edges

    # Initialization of set of representative points for all optimizer edges (initialized as a dict, since the number of points is
    # undetermined a priori)
    opt_edges = {}
    par_edges = {}

    ### Steps 1, 7 ### Processing all optimizer edges
    for i in range(dim['N_z']):
        
        ### Step 2 ### Exclude all edges not relevant to the BES from refinement (their points are not used in the BES)
        if not(par_edge_active_check[i]):

            ### Step 3 ### Setting the initial optimizers per optimizer edges: optimizer vertex and reference/extreme point from compact solution
            current_iter_x = np.array([[ref_points['x'][:,i], vertices['x']], ])
            current_iter_z = np.array([[ref_points['z'][i,i], vertices['z'][i]]])
    
            ### Steps 5, 6, 8 ### Finding all representative points for selected tolerance and saving them in dicts
            opt_edges[str(i)], par_edges[str(i)] = single_edge_refinement(mpNLP, dim, current_iter_x, current_iter_z, i, tolerance)

    return opt_edges, par_edges

#%% This function takes the extreme points of an optimizer edge (optimizer vertex and extreme point - from compact solution)
# and breaks this initial segment in half to test optimizer accuracy in the middle of the segment. If the error check is satisfied,
# no additioal points are added as representatives of the optimizer edge. Else, a new representative point is added, and new
# segments are created and checked sequentially until all segments are validated (error check satisfied at their middle points).
def single_edge_refinement(mpNLP, dim, current_iter_x, current_iter_z, opt_index, tolerance):

    # Set of representative point for optimizer edge - initialized with the coordinates of the optimizer vertex
    # and the reference optimizer obtained from the compact solution. This includes also the corresponding z values.
    opt_edge = current_iter_x[0]
    repres_z = current_iter_z[0]
    
    # Setting active set from "opt_index"
    act_set             = np.zeros(dim['N_z']).astype(int)
    act_set[opt_index]  = 1

    # Segments to process in first iteration
    to_proc_segments = 1

    # While at least one z-segment is to be processed, their assessment is executed within the while loop.
    while (to_proc_segments > 0):

        # Set of additional representative points to add to the optimizer edge (and the corresponding z values)
        # (initialized empty - new segments are added if applicable when the current iteration segments are checked)
        next_iter_x = np.empty((0,2,dim['N_x']))
        next_iter_z = np.empty((0,2))

        # Processing all segments for current iteration
        for i in range(to_proc_segments):

            ### Step 5.1 ### Optimizer at the average z of the current segment (current iteration)
            z_half  = current_iter_z[i].mean()
            opt_KKT = root(KKT_cond, np.zeros(dim['N_x']+dim['N_z']), jac=KKT_cond_derivs, method='hybr', args=(mpNLP, dim, act_set, mpNLP.b + z_half)).x[0:dim['N_x']]
            # print('optim: ' + str(opt_KKT))

            ### Steps 5.2, 5.3 ### Optimizer from linear interpolation in the same segment and error calculation (against result from explicit optimization)
            opt_interp  = current_iter_x[i].mean(axis=0)
            error       = error_calc(opt_KKT, opt_interp)
            # print('edge check: ' + str(error))

            ### Step 5.5 ### Error check 
            if (error > tolerance):

                # Adding points to set of representative points
                opt_edge = np.append(opt_edge, [opt_KKT], axis=0)
                repres_z = np.append(repres_z, z_half)

                # Adding segments for the next iteration
                next_iter_x = np.append(next_iter_x, np.array([[current_iter_x[i][0,:], opt_KKT], [opt_KKT, current_iter_x[i][1,:]]]), axis=0)
                next_iter_z = np.append(next_iter_z, np.array([[current_iter_z[i][0], z_half], [z_half, current_iter_z[i][1]]]), axis=0)

        # Once all segments are processed in current iteration, these are updated from the new set of segments detected (in the current iteration)
        current_iter_x = next_iter_x
        current_iter_z = next_iter_z

        # Update number of segments to process in next iteration
        to_proc_segments = next_iter_z.shape[0]

    ### Step 8 ### Sort vectors of representative points: from higest to lowest z
    sort_key = np.argsort(-repres_z)
    opt_edge = opt_edge[sort_key,:]

    ### Step 6 ### Representative points for the corresponding parametric edge
    par_edge = np.dot(mpNLP.A, opt_edge.T)
    for i in range(dim['N_z']):
        par_edge[i,:] -= mpNLP.b[i]

    return opt_edge, par_edge.T

#%% Taking any two vectors (of the same size), this function returns the sum of the squared differences on their coordinates.
def error_calc(opt1, opt2):

    # Size of error vector and its initialization
    N           = opt1.shape[0]
    error_vec   = np.zeros(N)

    # Error at each coordinates of the optimizers
    for i in range(N):
        error_vec[i] = (opt1[i] - opt2[i])**2

    # Sum of errors
    error_total = sum(error_vec)

    return error_total

#%% This function applies "initial_partitions_single_active_set" to a vector of all active sets, and returns the corresponding
# results in two dictionaries.
def initial_partitions(dim, act_sets, N_act_sets, par_edges):

    # Initialization of (index) reference points and matching active constraint indices associated to all active sets
    ref_points          = {}
    act_constr_index    = {}

    # Processing all active sets
    for i in range(N_act_sets):
        ref_points[str(i)], act_constr_index[str(i)] = initial_partitions_single_active_set(dim, par_edges, act_sets[i,:])

    return ref_points, act_constr_index

#%% Given an active set (expressed as a binary vector of size dim['N_z']), this function delivers the full set of associated partitions
# that may be constructed using the corresponding reference points (transformed parameters) obtained from their parametric edges. Partitions
# are delivered not as a set of coordinates and vectors from inactive edges, but at this stage keeping only the information on the points
# necessary to define them.
def initial_partitions_single_active_set(dim, par_edges, active_set):

    # Active constraints: from active set
    ###########################################################################
    act_constr_index = np.zeros(0).astype(int)

    for i in range(dim['N_z']):
        if (active_set[i]):
            act_constr_index = np.append(act_constr_index, [i])

    N_act_constr = act_constr_index.shape[0]

    # Auxiliary vectors - management of partitions construction
    ###########################################################################
    current_pos = np.ones(N_act_constr).astype(int)     # positions for relevant edges from where partitions are created
    max_pos     = np.zeros(N_act_constr).astype(int)    # maximum allowed positions per edge - initialization

    # Set from the number of points in the corresponding edges
    for i in range(N_act_constr):
        max_pos[i] = par_edges[str(act_constr_index[i])].shape[0]-1

    # Points available for partition construction per parametric edges
    points_avail = max_pos - current_pos

    # Edge to increment - the one with the most points available (from parametric vertex) for partition construction
    if (N_act_constr > 0):
        max_points_avail    = np.max(points_avail)
        to_incr_edge_index  = np.argmax(points_avail)
    else:
        max_points_avail    = 0

    # Partitions (identifies all relevant points - transformed parameters - via indices)
    ###########################################################################
    N_partitions    = sum(points_avail)+1                                   # Number of partitions (from sum of total points in their edges)
    ref_points      = np.zeros((N_act_constr+2, N_partitions)).astype(int)  # Initialization of matrix identifying all partitions for the given active set

    # First partition - including parametric vertex
    ref_points[0:N_act_constr,0]    = current_pos
    ref_points[N_act_constr,0]      = 0
    ref_points[N_act_constr+1,0]    = 0

    # Initialization of partitions index (column index)
    count = 1

    # Partition continues while there is at least one point left to increment on at least one of the associated edges
    while (max_points_avail > 0):

        # Setting references to the current partition (at column "count")
        ref_points[0:N_act_constr,count]    = current_pos
        ref_points[N_act_constr,count]      = to_incr_edge_index
        ref_points[N_act_constr+1,count]    = current_pos[to_incr_edge_index]+1

        # Update all support variables
        current_pos[to_incr_edge_index] += 1                        # Increment current position at edge used in this iteration to extend critical region construction
        points_avail                    = max_pos - current_pos
        max_points_avail                = np.max(points_avail)
        to_incr_edge_index              = np.argmax(points_avail)
        count                           += 1                        # Increment column index for next iteration

    return ref_points, act_constr_index

#%% Applies function "validate_break_single_active_set" for all initial partitions of all active sets, to return all associated
# partitions and optimizer functions (in the conventional format)
def validate_break_all_active_sets(mpNLP, dim, edges, opt_edges, par_edges, act_sets, ref_points, act_constr_index, bnds_par_edge_test, tolerance):

    # Initializes outputs: dictionaries of dictionaries of dictionaries of partitions and optimizer functions
    partitions = {}
    opt_funcs = {}

    # Processing all initial partitions for all active sets
    for i in range(act_sets.shape[0]):
        partitions[str(i)], opt_funcs[str(i)] = validate_break_single_active_set(mpNLP, dim, edges, opt_edges, par_edges, act_sets[i], ref_points[str(i)], act_constr_index[str(i)], bnds_par_edge_test, tolerance)

    return partitions, opt_funcs

#%% Applies function "validate_break_single_partition" for all initial partitions of a given active set, to return all associated
# partitions and optimizer functions (in the conventional format)
def validate_break_single_active_set(mpNLP, dim, edges, opt_edges, par_edges, act_sets, ref_points, act_constr_index, bnds_par_edge_test, tolerance):

    # Initializes outputs: dictionaries of dictionaries of partitions and optimizer functions
    partitions = {}
    opt_funcs = {}

    # Processing all initial partitions for current active set
    for i in range(ref_points.shape[1]):
        partitions[str(i)], opt_funcs[str(i)] = validate_break_single_partition(mpNLP, dim, edges, opt_edges, par_edges, act_sets, ref_points[:,i], act_constr_index, bnds_par_edge_test, tolerance)

    return partitions, opt_funcs

#%% This function assesses an initial partition (identified via the arguments "ref_points" and "act_constr_index"), constructs the partition and
# matching optimizer function in the conventional format, and then assesses it, and "breaks" it as many times as necessary in smaller partitions.
# Only those critical regions included in the parameter space and matching optimizer functions satisfying an error check are returned as results.
def validate_break_single_partition(mpNLP, dim, edges, opt_edges, par_edges, act_set, ref_points, act_constr_index, bnds_par_edge_test, tolerance):

    # Initialize outputs: dictionaries with all partitions and optimizer functions for initial partition being processed
    # (includes the initial partition, or it may be broken down in smaller partitions for the purpose of solution accuracy)
    partitions  = {}
    opt_funcs   = {}

    # Number of active constraints in active set being processed (from "act_constr_index")
    N_act_constr = act_constr_index.shape[0]

    # Construction of partition and optimizer function in conventional form for the initial partition being processed
    current_iter_partitions = single_partition_from_ref_points(dim, edges, par_edges, ref_points, act_constr_index).reshape(-1,dim['N_z']+1,dim['N_z']+1)
    current_iter_opt_funcs  = single_opt_func_from_ref_points(dim, edges, opt_edges, ref_points, act_constr_index).reshape(-1,dim['N_x'],N_act_constr+1)

    # Initialization of vectos and matrices for LP: checks if given partition is included in the parameter space
    # (this information may potentially be initialized and saved elsewhere...)
    cost    = np.ones(1+dim['N_z']+dim['N_theta'])
    A_ineq  = np.zeros((dim['N_par_bnds'],1+dim['N_z']+dim['N_theta']))
    A_eq    = np.zeros((1+dim['N_z'],1+dim['N_z']+dim['N_theta']))
    b_eq    = np.zeros(1+dim['N_z'])

    # Setting vector/matrix entries
    A_eq[0:dim['N_z'],1+dim['N_z']:1+dim['N_z']+dim['N_theta']] = mpNLP.feas_space_constr['F']
    b_eq[dim['N_z']]                                            = -1
    A_ineq[:,1+dim['N_z']:1+dim['N_z']+dim['N_theta']]          = mpNLP.par_space_constr['P_A']

    # Number of partitions to process in first iteration (initial partition)
    to_proc_partitions = 1

    # Index used to save results in dictionaries
    partitions_index = 0
    # count = 0

    # Validate and break partitions proceeds until for all of them accuracy of matching optimizer functions is within defined tolerance
    while (to_proc_partitions > 0): # count < 1

        # Initialization of partitions and optimizer functions to be processed in next iteration
        next_iter_partitions    = np.empty((0,dim['N_z']+1,dim['N_z']+1))
        next_iter_opt_funcs     = np.empty((0,dim['N_x'],N_act_constr+1))

        # Processing all partitions to be processed in current iteration. Check: (i) if they are included in the parameter space, and
        # (ii) their optimizer functions satisfy the defined error test
        for i in range(to_proc_partitions):

            # Setting "A_eq" to reflect the space they span in the transformed parameter space
            A_eq[:,0:dim['N_z']+1] = -current_iter_partitions[i,:,:]

            # Checking if a feasible solution exists for the defined LP (if so, partition is included in the parameter space)            
            status = lin_prog_cr_check(cost, A_ineq, mpNLP.par_space_constr['P_b'], A_eq, b_eq, bnds_par_edge_test)
            # print(status)

            # Optimizer function error test proceeds only if feasible solution exists for the LP (only if status: 0)
            if (status == 0):

                # Exact optimizer is obtained via the KKT conditions at the middle of the convex hull, z_half (using reference points for active constraints)
                z_half  = current_iter_partitions[i,0:dim['N_z'],0:N_act_constr+1].mean(axis=1)
                opt_KKT = root(KKT_cond, np.zeros(dim['N_x']+dim['N_z']), jac=KKT_cond_derivs, method='hybr', args=(mpNLP, dim, act_set, mpNLP.b + z_half)).x[0:dim['N_x']]

                # Optimizer at the middle of the convex hull via linear interpolation of matching optimizers
                opt_interp  = current_iter_opt_funcs[i,:,:].mean(axis=1)
                error       = error_calc(opt_KKT, opt_interp)
                # print(opt_KKT)
                # print(opt_interp)
                # print('partition check: ' + str(error))

                # Error test satisfied
                if (error < tolerance):

                    # New elements added to dictionary
                    partitions[str(partitions_index)]   = current_iter_partitions[i,:,:]
                    opt_funcs[str(partitions_index)]    = current_iter_opt_funcs[i,:,:]

                    # Index incremented to enable recording of next elements (if any)
                    partitions_index += 1

                # Error test not satisfied
                else:

                    # Transformed parameter at optimizer: will be used to create new partitions as the pivotal point
                    z_at_opt_KKT = np.dot(mpNLP.A, opt_KKT) - mpNLP.b

                    for j in range(N_act_constr+1):

                        # Creating "N_act_constr+1" new partitions from z_at_opt_KKT and reference points from the current partition
                        new_partition                   = 1*current_iter_partitions[i,:,:].reshape(-1,dim['N_z']+1,dim['N_z']+1) # multiplication by 1 to avoid objects becoming equal
                        new_partition[0,0:dim['N_z'],j] = z_at_opt_KKT

                        # Creating "N_act_constr+1" new optimizer functions from opt_KKT and reference optimizers from the current optimizer function
                        new_opt_func        = 1*current_iter_opt_funcs[i,:,:].reshape(-1,dim['N_x'],N_act_constr+1)
                        new_opt_func[0,:,j] = opt_KKT

                        # Saving new partition and optimizer function to be processed in next iteration
                        next_iter_partitions = np.append(next_iter_partitions, new_partition, axis=0)
                        next_iter_opt_funcs = np.append(next_iter_opt_funcs, new_opt_func, axis=0)

        # When processing of all partitions and optimizer functions to be processed in current iteration is concluded, these are set to the corresponding
        # to the information gathered on all partitions and optimizer functions to be processed in next: sets up new pass in while loop.
        current_iter_partitions = next_iter_partitions
        current_iter_opt_funcs  = next_iter_opt_funcs

        # Number of partitions to be processed updated  
        to_proc_partitions = current_iter_partitions.shape[0]
        # count += 1

    return partitions, opt_funcs

#%% Taking the information from compact solution, additional points for all parametric edges and indexes for partitions
# construction, this function delivers the partition in the position "part_index" for a given active set.
def single_partition_from_ref_points(dim, edges, par_edges, ref_points, act_constr_index):

    # Number of active sets - from indices of active constraints
    N_act_constr = act_constr_index.shape[0]

    # Vector of bool: True/False at inactive/active constraints (support for partition definition)
    inact_constr_key                    = np.ones(dim['N_z']).astype(bool)
    inact_constr_key[act_constr_index]  = 0

    # Partition - initialization
    partition   = np.zeros((dim['N_z']+1, dim['N_z']+1))

    # First set of "N_act_constr" specified via the corresponding points (from parametric edges and relevant indices)
    for i in range(N_act_constr):
        partition[0:dim['N_z'],i] = par_edges[str(act_constr_index[i])][ref_points[i],:]

    # The "N_act_constr$th column is specified from an additional reference point (using the last two rows of the reference positions matrix)
    if (N_act_constr > 0):
        partition[0:dim['N_z'],N_act_constr] = par_edges[str(act_constr_index[ref_points[N_act_constr]])][ref_points[N_act_constr+1],:]
    else:
        partition[0:dim['N_z'],N_act_constr] = par_edges[str(0)][0,:]

    # The last "dim['N_z']-N_act_constr" (equal number of inactive constraints) are set from the relevant rows of the matrix of inactive parametric edges
    partition[0:dim['N_z'],1+N_act_constr:1+dim['N_z']] = edges['z_inact'][:,inact_constr_key]

    # Last row set to ones in the first N_act_constr+1 columns: convex hull constraints to limit span of partitions
    partition[dim['N_z'],0:N_act_constr+1] = np.ones(N_act_constr+1)

    return partition

#%% Creates optimizer function for the corresponding partition
def single_opt_func_from_ref_points(dim, edges, opt_edges, ref_points, act_constr_index):

    # Number of active sets - from indices of active constraints
    N_act_constr = act_constr_index.shape[0]

    # Optimizer function - initialization
    opt_func    = np.zeros((dim['N_x'], N_act_constr+1))

    # First set of "N_act_constr" specified via the corresponding points (from optimizer edges and relevant indices)
    for i in range(N_act_constr):
        opt_func[:,i] = opt_edges[str(act_constr_index[i])][ref_points[i],:]

    # The "N_act_constr$th column is specified from an additional reference point (using the last two rows of the reference positions matrix)
    if (N_act_constr > 0):
        opt_func[:,N_act_constr] = opt_edges[str(act_constr_index[ref_points[N_act_constr]])][ref_points[N_act_constr+1],:]
    else:
        opt_func[:,N_act_constr] = opt_edges[str(0)][0,:]

    return opt_func

#%% Convert results from dicts into equivalent numpy arrays.
def list_results_as_arrays(dim, act_sets, all_partitions, all_opt_funcs):

    # Initialize number of partitions included in dict "all_partitions"
    N_partitions = 0

    # Count the total number of partitions in dict
    for i in range(len(all_partitions)): # active set index
        for j in range(len(all_partitions[str(i)])): # initial partition index
            N_partitions += len(all_partitions[str(i)][str(j)])

    # Initialize results: active sets, partitions, optimizer functions (all sharing the first index)
    active_sets_array   = np.zeros((N_partitions, dim['N_z'])).astype(int)
    partitions_array    = np.zeros((N_partitions, dim['N_z']+1, dim['N_z']+1))
    opt_funcs_array     = np.zeros((N_partitions, dim['N_x'], dim['N_z']+1))

    # Initialize partition index
    partition_index = 0

    # Process all entries of all dicts to convert results (active sets, partitions and optimizer functions)
    # in standard numpy array objects
    for i in range(len(all_partitions)): # active set index
        for j in range(len(all_partitions[str(i)])): # initial partition index
            for k in range(len(all_partitions[str(i)][str(j)])): # partitions from initial partition index

                # Populate arrays
                active_sets_array[partition_index,:]                = act_sets[i,:]
                partitions_array[partition_index,:,:]               = all_partitions[str(i)][str(j)][str(k)]
                N_act_constr                                        = sum(act_sets[i,:]).astype(int)
                opt_funcs_array[partition_index,:,0:N_act_constr+1] = all_opt_funcs[str(i)][str(j)][str(k)]

                # Increment partition index to populate next element in all arrays
                partition_index += 1

    return active_sets_array, partitions_array, opt_funcs_array

#%% Obtains critical region fragments
def solution_fragments_calc(mpNLP_def, dim, vertices, edges):
    
    # Initialization of list of optimal active sets
    optimal_active_sets = np.zeros((0, dim['N_z']))
    cr_fragments = np.zeros((0, dim['N_z'], dim['N_z']))
    
    for k in range(dim['N_z']+1):
        
        # Initialization of variables: k active constraints
        N_as_current_k, current_active_set, pivot, pos = active_set_init(dim['N_z'], k)
        
        for i in range (N_as_current_k):
  
            # Initialization of current edges matrix
            edges_matrix = np.zeros((dim['N_z'], 0))
            
            for j in range(dim['N_z']):
                
                # Obtain relevant parametric edges for current optimal active set
                if (current_active_set[j] == 0):
                    edges_matrix = np.append(edges_matrix, edges['z_inact'][:,j].reshape(dim['N_z'], -1), axis=1)
                else:
                    edges_matrix = np.append(edges_matrix, edges['z_act'][:,j].reshape(dim['N_z'], -1), axis=1)
            
            inv_matrix = np.linalg.inv(edges_matrix)
            
            # Append current active set and corresponding edges inverse matrix to solution
            optimal_active_sets = np.append(optimal_active_sets, current_active_set.reshape(-1, dim['N_z']), axis=0)
            cr_fragments = np.append(cr_fragments, inv_matrix.reshape(-1, dim['N_z'], dim['N_z']), axis=0)
            
            # Calculation of next active set
            current_active_set, pivot, pos = next_as_calc(dim['N_z'], current_active_set, pivot, pos)
    
    # Solution summary
    solution_fragments = {'oas':optimal_active_sets, 'cr':cr_fragments}
    
    return solution_fragments

# %% Problem dimensions
def dim_calc(feas_space_constr, par_space_constr):

    # Number of optimization variables and constraints
    N_x         = feas_space_constr['A'].shape[1]
    N_z         = feas_space_constr['A'].shape[0]
    N_theta     = feas_space_constr['F'].shape[1]
    N_par_bnds  = par_space_constr['P_A'].shape[0]

    # Maximum number of optimal active sets
    N_oas_max = 2**N_z

    dim = {'N_x':N_x, 'N_z':N_z, 'N_theta':N_theta, 'N_par_bnds':N_par_bnds, 'N_oas_max':N_oas_max}

    return dim

# %% From mp-QP library
def active_set_init(N_total, N_active_constr):
    # Number of possible active_sets for a given number of active constraints
    N_active_sets = int(factorial(N_total) / (factorial(N_total - N_active_constr) * factorial(N_active_constr)))
    
    if (N_total == 0):
        N_active_sets = 0

    # Initial active set including "N_active_constr" active constraints
    initial_active_set = np.zeros(N_total)
    for i in range(N_active_constr):
        initial_active_set[i] = 1
    
    # Auxiliary parameters facilitating the calculation of all active sets
    # including "N_active_constr" active constraints
    pivot   = N_active_constr-2
    pos     = N_active_constr-1
    
    # Define as vector of integers
    initial_active_set = initial_active_set.astype(int)
    
    return N_active_sets, initial_active_set, pivot, pos

# %% From mp-QP library
def next_as_calc(N_constr, current_as, pivot, pos):
    
    # Initialization of next active set
    next_as = np.zeros(N_constr)
    
    for i in range(N_constr):
        next_as[i] = current_as[i]
    
    if (pos < N_constr-1):
        
        # Logic branch 1: moves the rightmost "1" to the right
        next_as[pos] = 0
        next_as[pos+1] = 1
        
        pos = pos + 1
        
    else:
        
        next_as[pos] = 0
        next_as[pivot] = 0
            
        if (pos - pivot > 1):
            # Logic branch 2: moves the second rightmost "1" to the right and
            # the rightmost "1" immediatelly after
            pivot = pivot + 1
            pos = pivot + 1
            
            next_as[pivot] = 1
            next_as[pos] = 1
            
        else:
            # Logic branch 3: Finds the first "1" which can be moved to the
            # right. All remaining "1"s are placed in sequence.
            add = 0
            first_0 = False
            for i in range(pivot-1, -1, -1):
                if not(first_0):
                    if (current_as[i] == 1):
                        add = add + 1
                    else:
                        first_0 = True
                        
                else:
                    if (current_as[i] == 1):
                        next_as[i] = 0
                        pivot = i + 1
                        
                        next_as[pivot:pivot+add+3] = 1
                        next_as[pivot+add+3:N_constr] = 0
                        
                        pivot = pivot+add+1
                        pos = pivot+1
                        
                        break
    
    next_as = next_as.astype(int)
    
    return next_as, pivot, pos

# %% For a given array of active sets and for a current / candidate active set,
# tests if the current active set does not include combinations of constraints
# in the array provided
def active_set_match_calc(dim, active_sets_array, candidate_active_set):
    
    # Initialization of global match bool
    global_active_set_match = False
                
    for j in range(active_sets_array.shape[0]):
        
        # Initialization of single match bool
        current_active_set_match = True
        
        for k in range(dim['N_z']):
            if (active_sets_array[j,k] == 1 and candidate_active_set[k] == 0):
                current_active_set_match = False
                break
            
        if (current_active_set_match):
            global_active_set_match = True
            break
                    
    return global_active_set_match