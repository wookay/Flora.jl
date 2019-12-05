module test_topics_gradient_forwarddiff

using Test
using ForwardDiff

ForwardDiff.derivative
@test ForwardDiff.derivative(sin, 2pi) == cos(2pi) == 1.0

ForwardDiff.gradient
@test ForwardDiff.gradient(x->x[1], 5:7) == [1, 0, 0]
@test ForwardDiff.gradient(x->x[2], 5:7) == [0, 1, 0]
@test ForwardDiff.gradient(x->x[3], 5:7) == [0, 0, 1]

ForwardDiff.jacobian
@test ForwardDiff.jacobian(identity, 5:7) == [1 0 0;
                                              0 1 0;
                                              0 0 1]

ForwardDiff.hessian
@test ForwardDiff.hessian(x->1, rand(2)) == [0 0; 0 0]
@test ForwardDiff.hessian(x->1.0, rand(2)) == [0 0; 0 0]

J(∇f) = x -> ForwardDiff.jacobian(∇f, x)
H(f) = x -> ForwardDiff.hessian(f, x)

f(x::Vector) = sum(sin, x) + prod(tan, x) * sum(sqrt, x)
∇f = x -> ForwardDiff.gradient(f, x)

x = [0.986403
     0.140913
     0.294963
     0.837125
     0.650451] # rand(5)
@test H(f)(x) == J(∇f)(x)

end # module test_topics_gradient_forwarddiff
