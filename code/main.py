# Directories including all .py files necessary to execute the methods in this master file.
root_dir_str = ''                           # Root directory
code_dir_str = root_dir_str + '\code'       # files with code
prob_dir_str = root_dir_str + '\mp-NLPs'    # files with mp-NLP problems # \comp_study

# Insert path directories to locate necessary files
import sys

sys.path.insert(1, code_dir_str)
sys.path.insert(1, prob_dir_str)

# Problem to be solved - must be coded and saved in 'mp_NLPs' subfolder
import x2_z4_3 as mpNLP

# Project libraries
import f_aux
import par

# Algorithm parameters
general_par = par.general_par

# Step-by-step
#%% Instantiate key problem dimensions and bounds for auxiliary optimization problems
dim                                 = f_aux.dim_calc(mpNLP.feas_space_constr, mpNLP.par_space_constr)
bnds_min_z_i, bnds_par_edge_test    = f_aux.lin_prog_bnds(dim, mpNLP.bnds_theta)

#%% Compact solution
vertices, edges, ref_points, Lag_mult, total_process_time_cs = f_aux.compact_solution_calc(mpNLP, dim, bnds_min_z_i, general_par)

print(vertices['x'])
# print(edges['z_act'])
# print(ref_points['x'][1])
# print(Lag_mult)
# print(total_process_time_cs)

#%% Basic explicit solution (selection of an appropriate upper bound is key to obtain the correct results from the support MILPs;
# improvements on the numerical side of this algorithm still to consider - using GAMS or Gurobi algorithms may be one option)
act_sets, bes_partitions, opt_func, par_edge_active_check, N_act_sets, total_process_time_bes = f_aux.basic_explicit_solution(mpNLP, dim, bnds_par_edge_test, vertices, edges, ref_points, 1e5, "glpk") # , baron

print(act_sets)
# print(bes_partitions)
# print(opt_func)
# print(par_edge_active_check)
# print(total_process_time_bes)

#%% Refined explicit solution
act_sets_array, partitions_array, opt_funcs_array, opt_edges, par_edges, total_process_time_res = f_aux.refined_explicit_solution(mpNLP, dim, vertices, edges, ref_points, act_sets, N_act_sets, par_edge_active_check, bnds_par_edge_test, general_par)

print(act_sets_array.shape[0])
# print(act_sets_array)
# print(partitions_array.shape[0])
# print(partitions_array[2,:])
# print(opt_funcs_array)
# print(par_edges)
# print(opt_edges)
# print(total_process_time_res)
