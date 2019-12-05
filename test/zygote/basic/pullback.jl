module test_zygote_basic_pullback

using Test
using Zygote # Params

# Zygote.pullback
f(x) = 2x^3
y, back = Zygote.pullback(f, 3)
@test back(1) == (y,) == (54,)

# Zygote.forwarddiff
@test Zygote.forwarddiff(f, 3) == y

# code from https://github.com/FluxML/Zygote.jl/blob/master/test/features.jl
W = [1 0; 0 1]
x = [1, 2]
@test W * x == [1, 2]

p = Params([W])
@test p.order == [[1 0; 0 1]]

y, back = Zygote.pullback(() -> W * x, Params([W]))
@test y == [1, 2]
@test back([1, 1]) isa Zygote.Grads
@test back([1, 1])[W] == [1 2; 1 2]

end # module test_zygote_basic_pullback
