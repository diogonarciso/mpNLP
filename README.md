# mpNLP
Algorithms for multiparametric nonlinear programming

To make use of these code files, the following steps are required:

1) Save all files in this project to any folder (do not change the folder structure).
2) Open Python (code developed in Spyder) and install all necessary libraries (numpy, scipy, pyomo, and any other relevant libraries).
3) Open the "main.py" file. Specify the root folder in line 2 in accordance with Step 1 (or equivalent; e.g. update the list of folders read to import libraries)
4) Specify the problem to be solved in line 13. A list of example problems is included in the sub-folders "mp-NLPs" and "comp_study". Any new mp-NLP problem must be defined in accordance with the information provided for these examples: partial derivatives and Jacobian of the objective function; all constant matrices/vectors A, b, F, P_A, P_B and the bounds on the parameters space (from PA theta <= P_b, or setting them very wide).
5) Define the algorithm parameters in the file "par.py", or use the provided defaults.
6) Execute sequentially all cells in "main.py": the first two cells load and set up the problem to be solved. The next cells execute and deliver the compact solution, the basic explicit solution, and the refined explicit solution in sequence. It suffices to execute the corresponding code lines (28, 38 and 47). To query solutions, print selectively the outputs from all functions/algorithms.
