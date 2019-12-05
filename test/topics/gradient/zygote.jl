module test_topics_gradient_zygote

using Test
using Zygote

Zygote.gradient
@test Zygote.gradient(sin, 2pi) == (1.0, )  # cos(2pi)

Zygote.forward_jacobian
@test Zygote.forward_jacobian(identity, 5:7) == ([5, 6, 7], [1 0 0; 0 1 0; 0 0 1])
@test Zygote.forward_jacobian(x->x[1], 5:7) == (5, [1 0 0]')
@test Zygote.forward_jacobian(x->x[2], 5:7) == (6, [0 1 0]')
@test Zygote.forward_jacobian(x->x[3], 5:7) == (7, [0 0 1]')

Zygote.hessian
@test Zygote.hessian(x->x[1], 5:7) == [0 0 0; 0 0 0; 0 0 0]

end # module test_topics_gradient_zygote
