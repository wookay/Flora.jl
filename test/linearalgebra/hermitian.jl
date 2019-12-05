module test_linearalgebra_hermitian

using Test
using LinearAlgebra

# https://en.wikipedia.org/wiki/Hermitian_matrix

A = [2 2+im 4; 2-im 3 im; 4 -im 1]
@test ishermitian(A)
@test A' == adjoint(A) == (conj ∘ transpose)(A) == A
@test transpose(A) == [2 2-im 4; 2+im 3 -im; 4 im 1] != A

end # module test_linearalgebra_hermitian
