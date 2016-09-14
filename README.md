TbSLAS
======

Tree-based Semi-Lagrangian Advection-Diffusion Solver with added GPU support

view INSTALL for installation instructions

Added files:
examples/src/chebEval.cpp : 
				includes tests for different GPU implementations
				Test 1: NodeFieldFunctor Evaluation comparison on velocity tree (Matrix Multiplication)
				Test 2: NodeFieldFunctor Evaluation comparison on Gaussian tree (Matrix Multiplication)
				Test 3: Matrix Multiplication Evaluation comparison for CPU and GPU 
				Test 4: Compare single Matrix Multiplication
				Test 5: Compare computation of Chebyshev Polynomials on CPU and GPU
				Test 6:  NodeFieldFunctor Evaluation comparison on vorticity field tree (Vector Evaluation)
				Test 7:  NodeFieldFunctor Evaluation comparison on Gaussian tree (Vector Evaluation)
				Test 8:  NodeFieldFunctor Evaluation comparison on Hopf field tree (Vector Evaluation)
				fastest unrolled vector evaluation used in test 6,7 and 8
examples/src/chebEvalCuda.cu :
				includes CUDA kernels for the evaluation
examples/include/chebEval.h :
				header file for CUDA code
examples/include/preprocessordefines.h :
				includes preprocessor directives to create unrolled code on compile time
examples/include/setChebyshevDegree.h :
				Edit this file to change the Chebyshev Degree used for evaluation
				(specifing the degree as command line parameter is still necessary)
src/tree/node_field_functor_cuda.h :
				Contains code for Matrix Multiplication Evaluation on GPU
src/tree/tree_functor_cuda_vec.h :
				Contains code for Vector Evaluation on GPU