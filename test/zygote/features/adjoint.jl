module test_zygote_features_adjoint

using Test
using Zygote

@test (+)'(1) == 1


g(x) = 2x^3
@test gradient((f, x) -> f(x), g, 1) == (nothing, 6)

Zygote.@adjoint g(x) = g(x), Δ -> (g(x)+100, nothing)

@test g(1) == 2
@test g'(1) == 102
@test gradient((f, x) -> f(x), g, 1) == (nothing, 102)

end # module test_zygote_features_adjoint
