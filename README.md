# PyOpenCL-MatMul

This is an OpenCL Matrix Multiplication (GEMM) implementation using PyOpenCL. 

The GEMM cases all perform the following operation:
C = A B

where A is M x N, B is N x P, making C = M x P.

There are several implementations of GEMM:
1. GEMM: NDRange Kernel with Local Memory Tiling where M, N, P must all be multiples of BLOCK_SIZE
2. GEMM_1DREG: NDRange Kernel with Local Memory and 1D Register Tiling where M, N, P must all be multiples of BLOCK_SIZE
3. GEMM_2DREG: NDRange Kernel with Local Memory and 2D Register Tiling where M, N, P must all be multiples of BLOCK_SIZE
4. GEMM_IMITATE_PADDING: NDRange Kernel with Local Memory where M, N, P are arbitrarily sized
5. GEMM_2DREG_IMITATE_PADDING: NDRange Kernel with Local Memory and 2D Register Tiling where M, N, P are arbitrarily sized
-- this case computes the GEMM case imitates padding, i.e. as if it were padded, but no explicit padding is necessary. 
    This dramatically makes GEMM faster and allows for arbitrarily sized matrices.
