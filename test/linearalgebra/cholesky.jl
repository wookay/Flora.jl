module test_linearalgebra_cholesky

using Test
using LinearAlgebra

# https://en.wikipedia.org/wiki/Cholesky_decomposition#Example

A = [  4  12 -16;
      12  37 -43;
     -16 -43  98]

@test UpperTriangular(A) ==
      triu(A) ==
                            [  4  12 -16;
                               0  37 -43;
                               0   0  98]

@test LowerTriangular(A) ==
      tril(A) ==
                            [  4   0   0;
                              12  37   0;
                             -16 -43  98]
@test issymmetric(A)
@test A' == adjoint(A) == transpose(A) == A

C = cholesky(A)
@test C.L * C.U == A

end # module test_linearalgebra_cholesky
