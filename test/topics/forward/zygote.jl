module test_flora_forward_zygote

using Test
using Zygote

Zygote.forward
f(x) = 2x^3
y, back = Zygote.forward(f, 3)
@test back(1) == (y,) == (54,)

Zygote.forwarddiff
@test Zygote.forwarddiff(f, 3) == y

end # module test_flora_forward_zygote
