###############################################################################
# General parameters for algorithm execution
mode                    = 'abs_spec'    # 'abs_spec', 'rel_spec'
scope_contract_coeff    = [0.05, 0.05, 0.05, 0.05, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00]  # delta
perc_contract_coeff     = 0             # alpha
single_contract_coeff   = 0.05          # Delta z
tolerance_edges         = 1e-5          # zeta - edges
tolerance_partitions    = 1e-6          # zeta - partitions

general_par = {'mode':mode,
               'scope_contract_coeff':scope_contract_coeff,
               'perc_contract_coeff':perc_contract_coeff,
               'single_contract_coeff':single_contract_coeff,
               'tolerance_edges':tolerance_edges,
               'tolerance_partitions':tolerance_partitions
               }

###############################################################################
# Heuristics on parameter setting
#
# 1) mode (permitted values: 'abs_spec', 'rel_spec')
# This parameter controls which route is used to specify the vector of parameters
# "scope_contract_coeff"; if set to 'abs_spec', this vector of parameters is
# specified explicitly in this file (2) and parameter "perc_contract_coeff" is
# ignored in all calculations; if set to 'rel_spec', the vector of parameters
# "scope_contract_coeff" is ignored and specified at a later stage using
# parameter "perc_contract_coeff" (3).
#
# 2) scope_contract_coeff (vector with size equal to the number of inequality 
# constraints, where all entries are non-negative real values)
# This parameter is used in Algorithm 1. A motivation for its significance is
# schematically depicted in Figure 7 of the main manuscript: it allows a small
# extension of the critical regions in the BES and RES such that these solutions
# fully cover the space of parameters. In many problems, this extension is not
# required, and setting this vector such that all entries are 0 is adequate in
# this case. Computing a priori this parameter presents several challenges, and
# we argue that a simplified specification is a better compromise between accuracy
# and complexity. This route allows users to specify all entries of the vector
# manually. Small values are recommended to ensure that all points calculated per
# edge in Algorithm 1 are representative of the space of parameters and thus
# contributing to solution accuracy.
#
# 3) perc_contract_coeff (nonnegative real value, typically in the range 0 to 0.2)
# This parameter is better alligned with the definition of an heuristic. All
# entries of the vector of parameters "scope_contract_coeff" are calculated
# automatically as a relative extension beyond "z_min_i" (see Step 3A in Algorithm 1
# and Section 3.2.5 - main manuscript) at a later stage of The value 0.1 is 
# recommended to deal with the limitations of the manual setting of
# scope_contract_coeff as described above.
#
# 4) single_contract_coeff (small nonnegative real value)
# This parameter is used in Algorithm 1 and impacts only the CS; it enables an
# artificial extension to ensure a relevant point is found in those inequality
# constraints which for no combination of parameters are active. No value for this
# parameter ensures accuracy beyond the space of parameters and in the more
# general context of the p-dimensional space. Since the CS is presented mostly
# as an extension from mp-QP with limited applicability, this parameter has a very
# low impact overall. It has no impact on the BES and RES, since it is related to
# a set of always inactive constraints, and thus not included in these solutions.
#
# 5) tolerance_edges (small nonnegative real value)
# This parameter is used in Algorithm 3 and controls the accuracy of edges (how
# many additional points added to improve their description). From our experience
# so far, we recommend a smaller value for this parameter than to parameter
# "tolerance_partitions": this promotes than an expanded and accurate set of
# points is used in the initial construction of critical regions and which reduces
# the burden on the construction of critical regions (more expensive step). User
# should also take into account the magnitude of the optimization variables to
# define accordingly what is a suitable tolerance: this value is used to test
# the norm of the difference between two optimizers (absolute comparison).
#
# 6) tolerance_partitions (small nonnegative real value)
# This parameter is used in \textbf{Algorithm 5}. It should be set according to the
# guidelines relating to parameter "tolerance_edges". Low values for this parameter
# will promote that the initial set of critical regions obtained from Algorithm 4 will
# be partitioned sequentially to improve accuracy. Slighly higher values than
# "tolerance_edges" should take full advantage of the extra set of reference points
# calculated in Algorithm 3 to reduce the burden in Algorithm 5.
